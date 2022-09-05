from distutils.version import LooseVersion

import numpy as np
import einops

import torch
import torch.nn.functional

from ci_sdr.pt.toeplitz import toeplitz


# torch 1.7 introduces numpy compatible torch.fft module, while
# it was earlier a function.
# The old functions expect a real tensor as input, where the last dimension
# has 2 entries for the real and imag part.
# The new functions use the native complex support from torch.
loose_torch_version = LooseVersion(torch.__version__)
_native_complex = loose_torch_version >= "1.7.0"

if loose_torch_version >= '1.9.0':
    torch_linalg_solve = torch.linalg.solve
elif loose_torch_version <= '1.0.0':
    # if not hasattr(torch, 'solve'):  # torch <= 1.0.0
    #     torch.solve = torch.gesv
    def torch_linalg_solve(A, B):
        return torch.gesv(B, A)[0]
else:
    def torch_linalg_solve(A, B):
        return torch.solve(B, A)[0]


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


def rfft(signal, n_fft, _native_complex=_native_complex):
    """
    >>> import pytest
    >>> if not torch.__version__.startswith('1.7.'):
    ...     pytest.skip(f'This doctest only works with torch 1.7, but your version is: {torch.__version__}')

    >>> t = torch.tensor([1., 2, 3, 4])
    >>> rfft(t, 4, False)
    tensor([[10.,  0.],
            [-2.,  2.],
            [-2.,  0.]])
    >>> rfft(t, 8, False)
    tensor([[10.0000,  0.0000],
            [-0.4142, -7.2426],
            [-2.0000,  2.0000],
            [ 2.4142, -1.2426],
            [-2.0000,  0.0000]])
    >>> rfft(t, 4, True)
    tensor([10.+0.j, -2.+2.j, -2.+0.j])
    >>> rfft(t, 8, True)
    tensor([10.0000+0.0000j, -0.4142-7.2426j, -2.0000+2.0000j,  2.4142-1.2426j,
            -2.0000+0.0000j])

    >>> irfft(rfft(t, 4), 4)
    tensor([1., 2., 3., 4.])
    >>> irfft(rfft(t, 8), 8)
    tensor([1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00, 0.0000e+00, 1.1921e-07,
            0.0000e+00, 1.1921e-07])
    >>> irfft(rfft(t, 4, True), 4, True)
    tensor([1., 2., 3., 4.])
    >>> irfft(rfft(t, 8, True), 8, True)
    tensor([1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00, 0.0000e+00, 1.1921e-07,
            0.0000e+00, 1.1921e-07])

    >>> irfft(rfft(torch.stack([t, t + 10], 0), 4, True), 4, True)
    tensor([[ 1.,  2.,  3.,  4.],
            [11., 12., 13., 14.]])
    >>> irfft(rfft(torch.stack([t, t + 10], 0), 4, False), 4, False)
    tensor([[ 1.,  2.,  3.,  4.],
            [11., 12., 13., 14.]])
    """
    import torch
    if _native_complex:
        import torch.fft
        return torch.fft.rfft(signal, n_fft)
    else:
        signal = torch.nn.functional.pad(
            signal, [0, n_fft - signal.shape[-1]])
        return torch.rfft(signal, 1)


def irfft(signal, n_fft, _native_complex=_native_complex):
    import torch
    if _native_complex:
        import torch.fft
        return torch.fft.irfft(signal, n_fft)
    else:
        # signal_sizes shouldn't be nessesary, but torch.irfft has a bug,
        # that it calculates only the correct inverse, when signal_sizes is
        # given. Since this strange behaviour is documented in the pytorch
        # documentation I do not open a PR.
        # Assumption: In image processing you have not optimal sizes, so they
        #             introduced signal_sizes. And since they cannot use
        #             fft without signal_sizes, they didn't cared about the
        #             default behaviour.
        return torch.irfft(signal, 1, signal_sizes=(n_fft,))


def wiener_filter_predict(
        observation,
        desired,
        filter_length,
        return_w=False,
        _native_complex=_native_complex,
):
    """
    Also known as projection of observation to desired
    (mir_eval.separation._project)

    Args:
        observation: multi dimensional input with shape: [dim, time]
        desired: the desired signal with shape [time]
        filter_length: Filter legth of "w"
        return_w: If True, return the filter coefficients instead of the
            filtered signal.
        _native_complex:
            Implementation detail: Whether to use naitive complex support from
            torch.
            torch < 1.7: Has no complex support
            torch == 1.7: Supports old style and complex
            torch > 1.7: Droppt support for non native call

    Returns:
        observation convolved with w, where w is:
            w = argmin_t ( sum_t( |x_t * w - d_t|^2 ) )

    >>> from paderbox.utils.pretty import pprint
    >>> from ci_sdr.np.wiener_filter import wiener_filter_predict as np_wiener_filter_predict
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

    >>> wiener_filter_predict(torch.as_tensor(x), torch.as_tensor(y), 2, return_w=True)
    tensor([[ 0.4175, -0.0491]], dtype=torch.float64)
    >>> np.asarray(wiener_filter_predict(torch.as_tensor(x), torch.as_tensor(y), 2))
    array([ 0.41754386,  0.78596491,  1.15438596,  1.52280702,  1.89122807,
           -0.24561404])

    >>> x = np.array([[1, 2, 3, 4, 5.], [1, 2, 1, 2, 1.]])
    >>> y = np.array([1, 2, 1, 2, 1.])
    >>> pprint(_project(x, y, 2))
    array([1., 2., 1., 2., 1., 0.])
    >>> pprint(np_wiener_filter_predict(torch.as_tensor(x), torch.as_tensor(y), 2))
    array([1., 2., 1., 2., 1., 0.])
    >>> np.testing.assert_allclose(
    ...     np_wiener_filter_predict(torch.as_tensor(x), torch.as_tensor(y), 2, return_w=True),
    ...     [[0., 0.],
    ...      [1., 0.]], atol=1e-15, rtol=1e-15)
    >>> np.testing.assert_allclose(
    ...     np.asarray(wiener_filter_predict(torch.as_tensor(x), torch.as_tensor(y), 2, return_w=True)),
    ...     [[0., 0.],
    ...      [1., 0.]], atol=1e-15, rtol=1e-15)
    >>> pprint(np.asarray(wiener_filter_predict(torch.as_tensor(x), torch.as_tensor(y), 2)))
    array([ 1.,  2.,  1.,  2.,  1., -0.])

    """
    if len(observation.shape) == 1:
        observation = observation[None, :]

    observation_length = observation.shape[-1]

    n_fft = int(2 ** np.ceil(np.log2(
        observation_length + desired.shape[-1] - 1.
    )))

    Observation = rfft(observation, n_fft=n_fft, _native_complex=_native_complex)
    Desired = rfft(desired, n_fft=n_fft, _native_complex=_native_complex)

    if not _native_complex:
        assert Observation.shape[-1] == 2, Observation.shape
        assert len(Observation.shape[:-1]) == len(observation.shape), (Observation.shape, observation.shape)

    # Autocorr = np.einsum('KT,kT->KkT', Observation.conj(), Observation)
    if _native_complex:
        Autocorr = Observation[:, None, :].conj() * Observation[None, :, :]
    else:
        Autocorr = complex_mul(Observation[:, None, :], Observation[None, :, :], conj_x=True)

    autocorr = irfft(Autocorr, n_fft=n_fft, _native_complex=_native_complex)
    R = toeplitz(autocorr[..., :filter_length])
    R = einops.rearrange(
        R, 'source1 source2 filter1 filter2 -> (source1 filter1) (source2 filter2)',
        filter1=filter_length,
        filter2=filter_length,
        source1=observation.shape[-2],
        source2=observation.shape[-2],
    )

    # Crosscorr = np.einsum('KT,T->KT', Observation.conj(), Desired)
    if _native_complex:
        Crosscorr = Observation.conj() * Desired
    else:
        Crosscorr = complex_mul(Observation, Desired, conj_x=True)
    crosscorr = irfft(Crosscorr, n_fft=n_fft, _native_complex=_native_complex)

    p = crosscorr[..., :filter_length]
    p = einops.rearrange(p, 'source filter -> (source filter)')

    try:
        w = torch_linalg_solve(R, p[..., None])
    except Exception:
        raise Exception(p.shape, R.shape, crosscorr.shape)
    assert w.shape[-1] == 1, w.shape
    w = w[..., 0]
    w = einops.rearrange(w, '(source filter) -> source filter', filter=filter_length)

    if return_w:
        return w
    else:
        # This pads to much, but it allows us to reuse the fft of observation
        W = rfft(w, n_fft=n_fft, _native_complex=_native_complex)

        if _native_complex:
            estimate = Observation * W
        else:
            estimate = complex_mul(Observation, W)

        return irfft(
            torch.sum(estimate, dim=0), n_fft=n_fft, _native_complex=_native_complex,
        )[..., :observation_length + filter_length - 1]

        # return irfft(
        #     torch.sum(complex_mul(Observation, W), dim=0), n_fft=n_fft
        # )[..., :observation_length + filter_length - 1]


def wiener_filter_predict_single_input(
        observation, desired, filter_length,
        *,
        first_filter_index=0,
        return_w=False,
        _native_complex=_native_complex,
        return_locals=False,
):
    """

    Also known as projection of observation to desired
    (mir_eval.separation._project)

                      ┌────────┐
     x (observation)  │        │ y (returned)   ╭───╮       -d (desired)
    ─────────────────>│   w    │───────────────>│ + │<──────────────────
                      │        │                ╰───╯
                      └────────┘                  │
                                                  │ error (minimized)
                                                  v

    Args:
        observation: signle dimensional input with shape: [time]
        desired: the desired signal with shape [time]
        filter_length: The length of the filter.
        first_filter_index:
            The first index of the filter.
            0: Classical causal filter
            negative: Non causal filter
            positive: Not implemented
        return_w:
        _native_complex:
            Implementation detail: Whether to use naitive complex support from
            torch.
            torch < 1.7: Has no complex support
            torch == 1.7: Supports old style and complex
            torch > 1.7: Droppt support for non native call

    Returns: filterd signal with shape [time + filter_length - 1]

        x: observation
        d: desired
        L: filter_length
        S: first_filter_index

        (x * w)_t := sum_{l=S}^{L-S-1} x_{t - l} * w_l

        w = argmin_w ( sum( |(x * w)_t - d_t|^2 ) )
        return x * w

    >>> from ci_sdr.np.wiener_filter import wiener_filter_predict as np_wiener_filter_predict
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

    >>> wiener_filter_predict_single_input(torch.as_tensor(x), torch.as_tensor(y), 2, return_w=True)
    tensor([ 0.4175, -0.0491], dtype=torch.float64)
    >>> np.asarray(wiener_filter_predict_single_input(torch.as_tensor(x), torch.as_tensor(y), 2))
    array([ 0.41754386,  0.78596491,  1.15438596,  1.52280702,  1.89122807,
           -0.24561404])

    Test filter estimation

    >>> x = np.random.RandomState(0).randn(400).astype(dtype=np.float64)
    >>> filter = [1, -2]
    >>> y = np.convolve(x, filter)[:1-len(filter)]
    >>> filter_est = wiener_filter_predict_single_input(torch.as_tensor(x), torch.as_tensor(y), 2, return_w=True).numpy()
    >>> np.testing.assert_allclose(filter_est, [1.000079, -1.996279], atol=1e-6)

    Test concurent filter estimation

    >>> x1, x2 = np.random.RandomState(0).randn(2, 400).astype(dtype=np.float64)
    >>> filter1 = [1, -2]
    >>> filter2 = [3, -2]
    >>> y1 = np.convolve(x1, filter1)[:1-len(filter1)]
    >>> y2 = np.convolve(x2, filter2)[:1-len(filter2)]
    >>> filter_est = wiener_filter_predict_single_input(torch.as_tensor([x1, x2]), torch.as_tensor([y1, y2]), 2, return_w=True).numpy()
    >>> np.testing.assert_allclose(filter_est[0], [1.000079, -1.996279], atol=1e-6)
    >>> np.testing.assert_allclose(filter_est[1], [3.000067, -1.99662], atol=1e-6)

    Test non causal filters

    >>> x = np.random.RandomState(0).randn(400).astype(dtype=np.float64)
    >>> filter = [1, -2]
    >>> y = np.convolve(x, filter)[len(filter)-1:]
    >>> filter_est = wiener_filter_predict_single_input(torch.as_tensor(x), torch.as_tensor(y), 2, return_w=True).numpy()
    >>> np.testing.assert_allclose(filter_est, [-2.019528,  0.077869], atol=1e-6)  # Fail
    >>> filter_est = wiener_filter_predict_single_input(torch.as_tensor(x), torch.as_tensor(y), 2, first_filter_index=-1, return_w=True).numpy()
    >>> np.testing.assert_allclose(filter_est, [0.992035, -2.000169], atol=1e-6)   # Work

    >>> x = np.random.RandomState(0).randn(400).astype(dtype=np.float64)
    >>> filter = [1, -2, 1, -3]
    >>> y = np.convolve(x, filter)[2:-1]
    >>> filter_est = wiener_filter_predict_single_input(torch.as_tensor(x), torch.as_tensor(y), 4, first_filter_index=-2, return_w=True).numpy()
    >>> np.testing.assert_allclose(filter_est, [0.995203, -1.986345,  1.000766, -2.995206], atol=1e-6)   # Work
    """
    assert len(observation.shape) >= 1, observation.shape

    observation_length = observation.shape[-1]

    n_fft = int(2 ** np.ceil(np.log2(
        observation_length + desired.shape[-1] - 1.
    )))

    if first_filter_index == 0:
        pass
    elif first_filter_index < 0:
        if first_filter_index + filter_length < 0:
            raise NotImplementedError()
        desired = torch.nn.functional.pad(
            desired, [-first_filter_index, 0])
    else:
        observation = torch.nn.functional.pad(
            observation, [first_filter_index, 0])
        # raise NotImplementedError(first_filter_index)

    Observation = rfft(observation, n_fft=n_fft, _native_complex=_native_complex)
    Desired = rfft(desired, n_fft=n_fft, _native_complex=_native_complex)

    if not _native_complex:
        assert Observation.shape[-1] == 2, Observation.shape
        assert len(Observation.shape[:-1]) == len(observation.shape), (Observation.shape, observation.shape)

    if _native_complex:
        Autocorr = Observation.conj() * Observation
    else:
        Autocorr = complex_mul(Observation, Observation, conj_x=True)
    autocorr = irfft(Autocorr, n_fft=n_fft, _native_complex=_native_complex)
    R = toeplitz(autocorr[..., :filter_length])

    if _native_complex:
        Crosscorr = Observation.conj() * Desired
    else:
        Crosscorr = complex_mul(Observation, Desired, conj_x=True)
    crosscorr = irfft(Crosscorr, n_fft=n_fft, _native_complex=_native_complex)
    p = crosscorr[..., :filter_length]

    w = torch_linalg_solve(R, p[..., None])
    w = torch.squeeze(w, -1)
    # assert w.shape[-1] == 1, w.shape
    # w = w[..., 0]

    if return_locals:
        return locals()

    if return_w:
        return w
    else:
        # This pads to much, but it allows us to reuse the fft of observation
        W = rfft(w, n_fft=n_fft, _native_complex=_native_complex)

        if _native_complex:
            Est = Observation * W
        else:
            Est = complex_mul(Observation, W)

        est = irfft(
            Est, n_fft=n_fft, _native_complex=_native_complex
        )
        if first_filter_index <= 0:
            return est[..., :observation_length + filter_length - 1]
        else:
            return est[..., first_filter_index: first_filter_index+observation_length + filter_length - 1]
