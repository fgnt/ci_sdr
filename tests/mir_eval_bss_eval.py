"""BSS Eval v3 reference implementation, copied from mir_eval for the tests.

The doctests and tests of ci_sdr check its results against
``mir_eval.separation`` (``_project`` and ``bss_eval_sources``). That module
is deprecated since mir_eval 0.8 and will be removed in 0.9, and mir_eval
0.8.2 fails on numpy>=2.4 (it catches ``np.linalg.linalg.LinAlgError``). So
the functions used are copied here, unchanged apart from catching
``np.linalg.LinAlgError``, dropping the deprecation decorator, and dropping
the Examples section of ``bss_eval_sources``, which pytest would collect as
a doctest.

Source: mir_eval/separation.py at
https://github.com/mir-evaluation/mir_eval/blob/bca552d1e59e22a2d453998a319ef5804d806574/mir_eval/separation.py

The MIT License (MIT)

Copyright (c) 2014 Colin Raffel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import itertools
import warnings

import numpy as np
import scipy.fftpack
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve

MAX_SOURCES = 100


def validate(reference_sources, estimated_sources):
    """Check that the input data to a metric are valid, and throws helpful
    errors if not.

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing true sources
    estimated_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing estimated sources

    """
    if reference_sources.shape != estimated_sources.shape:
        raise ValueError(
            "The shape of estimated sources and the true "
            "sources should match.  reference_sources.shape "
            "= {}, estimated_sources.shape "
            "= {}".format(reference_sources.shape, estimated_sources.shape)
        )

    if reference_sources.ndim > 3 or estimated_sources.ndim > 3:
        raise ValueError(
            "The number of dimensions is too high (must be less "
            "than 3). reference_sources.ndim = {}, "
            "estimated_sources.ndim "
            "= {}".format(reference_sources.ndim, estimated_sources.ndim)
        )

    if reference_sources.size == 0:
        warnings.warn(
            "reference_sources is empty, should be of size "
            "(nsrc, nsample).  sdr, sir, sar, and perm will all "
            "be empty np.ndarrays"
        )
    elif _any_source_silent(reference_sources):
        raise ValueError(
            "All the reference sources should be non-silent (not "
            "all-zeros), but at least one of the reference "
            "sources is all 0s, which introduces ambiguity to the"
            " evaluation. (Otherwise we can add infinitely many "
            "all-zero sources.)"
        )

    if estimated_sources.size == 0:
        warnings.warn(
            "estimated_sources is empty, should be of size "
            "(nsrc, nsample).  sdr, sir, sar, and perm will all "
            "be empty np.ndarrays"
        )
    elif _any_source_silent(estimated_sources):
        raise ValueError(
            "All the estimated sources should be non-silent (not "
            "all-zeros), but at least one of the estimated "
            "sources is all 0s. Since we require each reference "
            "source to be non-silent, having a silent estimated "
            "source will result in an underdetermined system."
        )

    if (
        estimated_sources.shape[0] > MAX_SOURCES
        or reference_sources.shape[0] > MAX_SOURCES
    ):
        raise ValueError(
            "The supplied matrices should be of shape (nsrc,"
            " nsampl) but reference_sources.shape[0] = {} and "
            "estimated_sources.shape[0] = {} which is greater "
            "than mir_eval.separation.MAX_SOURCES = {}.  To "
            "override this check, set "
            "mir_eval.separation.MAX_SOURCES to a "
            "larger value.".format(
                reference_sources.shape[0], estimated_sources.shape[0], MAX_SOURCES
            )
        )


def _any_source_silent(sources):
    """Return true if the parameter sources has any silent first dimensions"""
    return np.any(
        np.all(np.sum(sources, axis=tuple(range(2, sources.ndim))) == 0, axis=1)
    )


def bss_eval_sources(reference_sources, estimated_sources, compute_permutation=True):
    """
    Ordering and measurement of the separation quality for estimated source
    signals in terms of filtered true source, interference and artifacts.

    The decomposition allows a time-invariant filter distortion of length
    512, as described in Section III.B of [#vincent2006performance]_.

    Passing ``False`` for ``compute_permutation`` will improve the computation
    performance of the evaluation; however, it is not always appropriate and
    is not the way that the BSS_EVAL Matlab toolbox computes bss_eval_sources.

    Parameters
    ----------
    reference_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing true sources (must have same shape as
        estimated_sources)
    estimated_sources : np.ndarray, shape=(nsrc, nsampl)
        matrix containing estimated sources (must have same shape as
        reference_sources)
    compute_permutation : bool, optional
        compute permutation of estimate/source combinations (True by default)

    Returns
    -------
    sdr : np.ndarray, shape=(nsrc,)
        vector of Signal to Distortion Ratios (SDR)
    sir : np.ndarray, shape=(nsrc,)
        vector of Source to Interference Ratios (SIR)
    sar : np.ndarray, shape=(nsrc,)
        vector of Sources to Artifacts Ratios (SAR)
    perm : np.ndarray, shape=(nsrc,)
        vector containing the best ordering of estimated sources in
        the mean SIR sense (estimated source number ``perm[j]`` corresponds to
        true source number ``j``). Note: ``perm`` will be ``[0, 1, ...,
        nsrc-1]`` if ``compute_permutation`` is ``False``.

    References
    ----------
    .. [#] Emmanuel Vincent, Shoko Araki, Fabian J. Theis, Guido Nolte, Pau
        Bofill, Hiroshi Sawada, Alexey Ozerov, B. Vikrham Gowreesunker, Dominik
        Lutter and Ngoc Q.K. Duong, "The Signal Separation Evaluation Campaign
        (2007-2010): Achievements and remaining challenges", Signal Processing,
        92, pp. 1928-1936, 2012.

    """
    # make sure the input is of shape (nsrc, nsampl)
    if estimated_sources.ndim == 1:
        estimated_sources = estimated_sources[np.newaxis, :]
    if reference_sources.ndim == 1:
        reference_sources = reference_sources[np.newaxis, :]

    validate(reference_sources, estimated_sources)
    # If empty matrices were supplied, return empty lists (special case)
    if reference_sources.size == 0 or estimated_sources.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    nsrc = estimated_sources.shape[0]

    # does user desire permutations?
    if compute_permutation:
        # compute criteria for all possible pair matches
        sdr = np.empty((nsrc, nsrc))
        sir = np.empty((nsrc, nsrc))
        sar = np.empty((nsrc, nsrc))
        for jest in range(nsrc):
            for jtrue in range(nsrc):
                s_true, e_spat, e_interf, e_artif = _bss_decomp_mtifilt(
                    reference_sources, estimated_sources[jest], jtrue, 512
                )
                sdr[jest, jtrue], sir[jest, jtrue], sar[jest, jtrue] = _bss_source_crit(
                    s_true, e_spat, e_interf, e_artif
                )

        # select the best ordering
        perms = list(itertools.permutations(list(range(nsrc))))
        mean_sir = np.empty(len(perms))
        dum = np.arange(nsrc)
        for i, perm in enumerate(perms):
            mean_sir[i] = np.mean(sir[perm, dum])
        popt = perms[np.argmax(mean_sir)]
        idx = (popt, dum)
        return (sdr[idx], sir[idx], sar[idx], np.asarray(popt))
    else:
        # compute criteria for only the simple correspondence
        # (estimate 1 is estimate corresponding to reference source 1, etc.)
        sdr = np.empty(nsrc)
        sir = np.empty(nsrc)
        sar = np.empty(nsrc)
        for j in range(nsrc):
            s_true, e_spat, e_interf, e_artif = _bss_decomp_mtifilt(
                reference_sources, estimated_sources[j], j, 512
            )
            sdr[j], sir[j], sar[j] = _bss_source_crit(s_true, e_spat, e_interf, e_artif)

        # return the default permutation for compatibility
        popt = np.arange(nsrc)
        return (sdr, sir, sar, popt)


def _bss_decomp_mtifilt(reference_sources, estimated_source, j, flen):
    """Decomposition of an estimated source image into four components
    representing respectively the true source image, spatial (or filtering)
    distortion, interference and artifacts, derived from the true source
    images using multichannel time-invariant filters.
    """
    nsampl = estimated_source.size
    # decomposition
    # true source image
    s_true = np.hstack((reference_sources[j], np.zeros(flen - 1)))
    # spatial (or filtering) distortion
    e_spat = (
        _project(reference_sources[j, np.newaxis, :], estimated_source, flen) - s_true
    )
    # interference
    e_interf = _project(reference_sources, estimated_source, flen) - s_true - e_spat
    # artifacts
    e_artif = -s_true - e_spat - e_interf
    e_artif[:nsampl] += estimated_source
    return (s_true, e_spat, e_interf, e_artif)


def _project(reference_sources, estimated_source, flen):
    """Least-squares projection of estimated source on the subspace spanned by
    delayed versions of reference sources, with delays between 0 and flen-1
    """
    nsrc = reference_sources.shape[0]
    nsampl = reference_sources.shape[1]

    # computing coefficients of least squares problem via FFT ##
    # zero padding and FFT of input data
    reference_sources = np.hstack((reference_sources, np.zeros((nsrc, flen - 1))))
    estimated_source = np.hstack((estimated_source, np.zeros(flen - 1)))
    n_fft = int(2 ** np.ceil(np.log2(nsampl + flen - 1.0)))
    sf = scipy.fftpack.fft(reference_sources, n=n_fft, axis=1)
    sef = scipy.fftpack.fft(estimated_source, n=n_fft)
    # inner products between delayed versions of reference_sources
    G = np.zeros((nsrc * flen, nsrc * flen))
    for i in range(nsrc):
        for j in range(nsrc):
            ssf = sf[i] * np.conj(sf[j])
            ssf = np.real(scipy.fftpack.ifft(ssf))
            ss = toeplitz(np.hstack((ssf[0], ssf[-1:-flen:-1])), r=ssf[:flen])
            G[i * flen : (i + 1) * flen, j * flen : (j + 1) * flen] = ss
            G[j * flen : (j + 1) * flen, i * flen : (i + 1) * flen] = ss.T
    # inner products between estimated_source and delayed versions of
    # reference_sources
    D = np.zeros(nsrc * flen)
    for i in range(nsrc):
        ssef = sf[i] * np.conj(sef)
        ssef = np.real(scipy.fftpack.ifft(ssef))
        D[i * flen : (i + 1) * flen] = np.hstack((ssef[0], ssef[-1:-flen:-1]))

    # Computing projection
    # Distortion filters
    try:
        C = np.linalg.solve(G, D).reshape(flen, nsrc, order="F")
    except np.linalg.LinAlgError:
        C = np.linalg.lstsq(G, D)[0].reshape(flen, nsrc, order="F")
    # Filtering
    sproj = np.zeros(nsampl + flen - 1)
    for i in range(nsrc):
        sproj += fftconvolve(C[:, i], reference_sources[i])[: nsampl + flen - 1]
    return sproj


def _bss_source_crit(s_true, e_spat, e_interf, e_artif):
    """Measurement of the separation quality for a given source in terms of
    filtered true source, interference and artifacts.
    """
    # energy ratios
    s_filt = s_true + e_spat
    sdr = _safe_db(np.sum(s_filt**2), np.sum((e_interf + e_artif) ** 2))
    sir = _safe_db(np.sum(s_filt**2), np.sum(e_interf**2))
    sar = _safe_db(np.sum((s_filt + e_interf) ** 2), np.sum(e_artif**2))
    return (sdr, sir, sar)


def _safe_db(num, den):
    """Properly handle the potential +Inf db SIR, instead of raising a
    RuntimeWarning. Only denominator is checked because the numerator can never
    be 0.
    """
    if den == 0:
        return np.inf
    return 10 * np.log10(num / den)
