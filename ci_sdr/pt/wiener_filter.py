import numpy as np
import einops

import torch

from padercontrib.pytorch.ops.mir_eval_sdr.toeplitz import toeplitz

if not hasattr(torch, 'solve'):
    torch.solve = torch.gesv


def complex_mul(x, y, conj_x=False):
    """
    >>> x = torch.as_tensor([[2, 3]])
    >>> y = torch.as_tensor([[5, 7]])
    >>> complex_mul(x, y), complex_mul(x, y, conj_x=True)
    (tensor([[-11,  29]]), tensor([[31, -1]]))
    >>> x, y = np.array([[2+3j], [5+7j]])
    >>> x*y, x.conj() * y
    (array([-11.+29.j]), array([31.-1.j]))
    """
    assert x.shape[-1] == 2, x.shape
    assert y.shape[-1] == 2, y.shape
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    if conj_x:
        return torch.stack([a * c + b * d, -b * c + a * d], -1)
    else:
        return torch.stack([a * c - b * d, b * c + a * d], -1)


def rfft(signal, n_fft):
    signal = torch.nn.functional.pad(
        signal, [0, n_fft - signal.shape[-1]])
    return torch.rfft(signal, 1)


def irfft(signal, n_fft):
    # signal_sizes shouldn't be nessesary, but torch.irfft has a bug,
    # that it calculates only the correct inverse, when signal_sizes is
    # given. Since this strange behaviour is documented in the pytorch
    # documentation I do not open a PR.
    # Assumption: In image processing you have not optimal sizes, so they
    #             introduced signal_sizes. And since they cannot use
    #             fft without signal_sizes, they didn't cared about the
    #             default behaviour.
    return torch.irfft(signal, 1, signal_sizes=(n_fft,))


def pt_wiener_filter_predict(observation, desired, filter_length, return_w=False):
    """
    Also known as projection of observation to desired
    (mir_eval.separation._project)

    w = argmin_w ( sum( |x * w - d|^2 ) )
    return x * w

    >>> from paderbox.notebook import pprint
    >>> from padercontrib.pytorch.ops.mir_eval_sdr.np_wiener_filter import np_wiener_filter_predict
    >>> x = np.array([1, 2, 3, 4, 5.])
    >>> y = np.array([1, 2, 1, 2, 1.])
    >>> from mir_eval.separation import _project
    >>> _project(x[None], y, 2)
    array([ 0.41754386,  0.78596491,  1.15438596,  1.52280702,  1.89122807,
           -0.24561404])
    >>> np_wiener_filter_predict(x, y, 2)
    array([ 0.41754386,  0.78596491,  1.15438596,  1.52280702,  1.89122807,
           -0.24561404])
    >>> np_wiener_filter_predict(x, y, 2, return_w=True)
    array([[ 0.41754386, -0.04912281]])

    >>> pt_wiener_filter_predict(torch.as_tensor(x), torch.as_tensor(y), 2, return_w=True)
    tensor([[ 0.4175, -0.0491]], dtype=torch.float64)
    >>> np.asarray(pt_wiener_filter_predict(torch.as_tensor(x), torch.as_tensor(y), 2))
    array([ 0.41754386,  0.78596491,  1.15438596,  1.52280702,  1.89122807,
           -0.24561404])

    >>> x = np.array([[1, 2, 3, 4, 5.], [1, 2, 1, 2, 1.]])
    >>> y = np.array([1, 2, 1, 2, 1.])
    >>> pprint(_project(x, y, 2))
    array([ 1.,  2.,  1.,  2.,  1., -0.])
    >>> pprint(np_wiener_filter_predict(torch.as_tensor(x), torch.as_tensor(y), 2))
    array([1., 2., 1., 2., 1., 0.])
    >>> pprint(np_wiener_filter_predict(torch.as_tensor(x), torch.as_tensor(y), 2, return_w=True))
    array([[0., 0.],
           [1., 0.]])
    >>> pprint(np.asarray(pt_wiener_filter_predict(torch.as_tensor(x), torch.as_tensor(y), 2, return_w=True)))
    array([[0., 0.],
           [1., 0.]])
    >>> pprint(np.asarray(pt_wiener_filter_predict(torch.as_tensor(x), torch.as_tensor(y), 2)))
    array([ 1.,  2.,  1.,  2.,  1., -0.])

    """
    if len(observation.shape) == 1:
        observation = observation[None, :]

    observation_length = observation.shape[-1]

    n_fft = int(2 ** np.ceil(np.log2(
        observation_length + desired.shape[-1] - 1.
    )))


    Observation = rfft(observation, n_fft=n_fft)
    Desired = rfft(desired, n_fft=n_fft)

    assert Observation.shape[-1] == 2, Observation.shape
    assert len(Observation.shape[:-1]) == len(observation.shape), (Observation.shape, observation.shape)

    # Autocorr = np.einsum('KT,kT->KkT', Observation.conj(), Observation)
    Autocorr = complex_mul(Observation[:, None, :], Observation[None, :, :], conj_x=True)

    autocorr = irfft(Autocorr, n_fft=n_fft)
    R = toeplitz(autocorr[..., :filter_length])
    R = einops.rearrange(
        R, 'source1 source2 filter1 filter2 -> (source1 filter1) (source2 filter2)',
        filter1=filter_length,
        filter2=filter_length,
        source1=observation.shape[-2],
        source2=observation.shape[-2],
    )

    # Crosscorr = np.einsum('KT,T->KT', Observation.conj(), Desired)
    Crosscorr = complex_mul(Observation, Desired, conj_x=True)
    crosscorr = irfft(Crosscorr, n_fft=n_fft)
    p = crosscorr[..., :filter_length]
    p = einops.rearrange(p, 'source filter -> (source filter)')

    # Note: The solve arguments are swapped in pytorch
    try:
        w, _ = torch.solve(p[..., None], R)
    except Exception:
        raise Exception(p.shape, R.shape, crosscorr.shape)
    assert w.shape[-1] == 1, w.shape
    w = w[..., 0]
    w = einops.rearrange(w, '(source filter) -> source filter', filter=filter_length)

    if return_w:
        return w
    else:
        # This pads to much, but it allows us to reuse the fft of observation
        W = rfft(w, n_fft=n_fft)

        return irfft(
            torch.sum(complex_mul(Observation, W), dim=0), n_fft=n_fft
        )[..., :observation_length + filter_length - 1]


def pt_wiener_filter_predict_single_input(
        observation, desired, filter_length, return_w=False):
    """
    Also known as projection of observation to desired
    (mir_eval.separation._project)

    w = argmin_w ( sum( |x * w - d|^2 ) )
    return x * w

    >>> from padercontrib.pytorch.ops.mir_eval_sdr.np_wiener_filter import np_wiener_filter_predict
    >>> x = np.array([1, 2, 3, 4, 5.])
    >>> y = np.array([1, 2, 1, 2, 1.])
    >>> from mir_eval.separation import _project
    >>> _project(x[None], y, 2)
    array([ 0.41754386,  0.78596491,  1.15438596,  1.52280702,  1.89122807,
           -0.24561404])
    >>> np_wiener_filter_predict(x, y, 2)
    array([ 0.41754386,  0.78596491,  1.15438596,  1.52280702,  1.89122807,
           -0.24561404])
    >>> np_wiener_filter_predict(x, y, 2, return_w=True)
    array([[ 0.41754386, -0.04912281]])

    >>> pt_wiener_filter_predict_single_input(torch.as_tensor(x), torch.as_tensor(y), 2, return_w=True)
    tensor([ 0.4175, -0.0491], dtype=torch.float64)
    >>> np.asarray(pt_wiener_filter_predict_single_input(torch.as_tensor(x), torch.as_tensor(y), 2))
    array([ 0.41754386,  0.78596491,  1.15438596,  1.52280702,  1.89122807,
           -0.24561404])
    """
    import torch
    from padercontrib.pytorch.ops.mir_eval_sdr.toeplitz import toeplitz

    assert len(observation.shape) == 1, observation.shape

    observation_length = observation.shape[-1]

    n_fft = int(2 ** np.ceil(np.log2(
        observation_length + desired.shape[-1] - 1.
    )))

    Observation = rfft(observation, n_fft=n_fft)
    Desired = rfft(desired, n_fft=n_fft)

    assert Observation.shape[-1] == 2, Observation.shape
    assert len(Observation.shape[:-1]) == len(observation.shape), (Observation.shape, observation.shape)

    Autocorr = complex_mul(Observation, Observation, conj_x=True)
    autocorr = irfft(Autocorr, n_fft=n_fft)
    R = toeplitz(autocorr[..., :filter_length])

    Crosscorr = complex_mul(Observation, Desired, conj_x=True)
    crosscorr = irfft(Crosscorr, n_fft=n_fft)
    p = crosscorr[..., :filter_length]

    # Note: The solve arguments are swapped in pytorch
    w, _ = torch.solve(p[..., None], R)
    assert w.shape[-1] == 1, w.shape
    w = w[..., 0]

    if return_w:
        return w
    else:
        # This pads to much, but it allows us to reuse the fft of observation
        W = rfft(w, n_fft=n_fft)

        return irfft(
            complex_mul(Observation, W), n_fft=n_fft
        )[..., :observation_length + filter_length - 1]
