"""
This file is just to verify that the pytorch implementation is correct
"""


import numpy as np
import paderbox as pb
from scipy.linalg import toeplitz
import scipy.signal
import einops


def stack_history(x, length, cut=True, flip=False):
    out = pb.array.segment_axis(x, length, 1, end='conv_pad')
    if cut:
        assert length > 1, length
        out = out[..., :1-length, :]
    if flip:
        out = np.flip(out, axis=-1)
    return out


def brute_force_autocorrelate(x, length):
    """
    >>> a = np.linspace(0, 1)
    >>> brute_force_autocorrelate(a, 4)
    array([16.83673469, 16.32653061, 15.81674302, 15.30778842])
    """
    assert length > 0, length
    return np.flip(np.sum(x[..., None] * stack_history(x, length).conj(), axis=-2), axis=-1)


def brute_force_crosscorrelate(x, y, length):
    """
    >>> x = np.linspace(0, 1)
    >>> y = np.logspace(0, 1)
    >>> brute_force_crosscorrelate(x, y, 4)
    array([134.68100035, 130.67979405, 126.69997781, 122.74258078])
    """
    assert length > 1, length
    return np.flip(
        np.sum(y[..., None] * stack_history(x, length, cut=True).conj(),
               axis=-2), axis=-1)


def fft_autocorrelate(x, length):
    """

    >>> a = np.linspace(0, 1)
    >>> fft_autocorrelate(a, 4)
    array([16.83673469, 16.32653061, 15.81674302, 15.30778842])
    """
    len1 = x.shape[-1]
    len2 = x.shape[-1]

    n_fft = int(2**np.ceil(np.log2(len1 + len2 - 1.)))
    X = np.fft.rfft(x, n=n_fft, axis=-1)
    return np.fft.irfft(X.conj() * X)[:length]


def fft_crosscorrelate(x, y, length):
    """
    >>> x = np.linspace(0, 1)
    >>> y = np.logspace(0, 1)
    >>> fft_crosscorrelate(x, y, 4)
    array([134.68100035, 130.67979405, 126.69997781, 122.74258078])
    """
    len1 = x.shape[-1]
    len2 = y.shape[-1]

    n_fft = int(2**np.ceil(np.log2(len1 + len2 - 1.)))
    X = np.fft.rfft(x, n=n_fft, axis=-1)
    Y = np.fft.rfft(y, n=n_fft, axis=-1)
    return np.fft.irfft(X.conj() * Y)[:length]


def wiener_filter_predict(observation, desired, filter_order, return_w=False):
    """
    Also known as projection of observation to desired
    (mir_eval.separation._project)

    w = argmin_w ( sum( |x * w - d|^2 ) )
    return x * w

    >>> from paderbox.utils.pretty import pprint
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([1, 2, 1, 2, 1])
    >>> from mir_eval.separation import _project
    >>> _project(x[None], y, 2)
    array([ 0.41754386,  0.78596491,  1.15438596,  1.52280702,  1.89122807,
           -0.24561404])
    >>> wiener_filter_predict(x, y, 2)
    array([ 0.41754386,  0.78596491,  1.15438596,  1.52280702,  1.89122807,
           -0.24561404])
    >>> wiener_filter_predict(np.array([x]), y, 2)
    array([ 0.41754386,  0.78596491,  1.15438596,  1.52280702,  1.89122807,
           -0.24561404])
    >>> wiener_filter_predict(np.array([x, -x]), y, 2)
    array([ 0.41754386,  0.78596491,  1.15438596,  1.52280702,  1.89122807,
           -0.24561404])
    >>> _project(np.array([x, -x]), y, 2)
    array([ 0.41754386,  0.78596491,  1.15438596,  1.52280702,  1.89122807,
           -0.24561404])
    >>> pprint(wiener_filter_predict(np.array([x, y]), y, 2))
    array([1., 2., 1., 2., 1., 0.])
    >>> pprint(_project(np.array([x, y]), y, 2))
    array([1., 2., 1., 2., 1., 0.])

    """
    n_fft = int(2**np.ceil(np.log2(
        observation.shape[-1] + desired.shape[-1] - 1.
    )))

    if observation.ndim == 1:
        observation = observation[None, :]

    Observation = np.fft.rfft(observation, n=n_fft, axis=-1)
    Desired = np.fft.rfft(desired, n=n_fft, axis=-1)

    Autocorr = np.einsum('KT,kT->KkT', Observation.conj(), Observation)
    Crosscorr = np.einsum('KT,T->KT', Observation.conj(), Desired)

    autocorr = np.fft.irfft(Autocorr)
    crosscorr = np.fft.irfft(Crosscorr)

    R = np.array([
        [
            scipy.linalg.toeplitz(a[:filter_order])
            for a in aa
        ]
        for aa in autocorr
    ])
    R = einops.rearrange(R, 'source1 source2 filter1 filter2 -> (source1 filter1) (source2 filter2)')

    p = crosscorr[..., :filter_order]
    p = einops.rearrange(p, 'source filter -> (source filter)')

    from paderbox.math.solve import stable_solve

    w = np.squeeze(stable_solve(R, p[..., None]), axis=-1)
    w = einops.rearrange(w, '(source filter) -> source filter', filter=filter_order)

    if return_w:
        return w
    else:
        return np.sum([
            scipy.signal.fftconvolve(o, filter, axes=(-1))
            for o, filter in zip(observation, w)
        ], axis=0)
