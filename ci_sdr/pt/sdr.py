import itertools
import torch
import einops
from ci_sdr.pt.wiener_filter import wiener_filter_predict_single_input


def soft_max_SDR_to_eps(soft_max_SDR):
    if soft_max_SDR is None:
        return None
    else:
        assert soft_max_SDR > 5, soft_max_SDR
        return 10 ** (-soft_max_SDR / 10)


def linear_to_db(numerator, denominator, eps=None):
    """
    >>> linear_to_db(torch.tensor(100.), torch.tensor(1.))
    tensor(20.)
    >>> linear_to_db(torch.tensor(100.), torch.tensor(1.), soft_max_SDR_to_eps(10))
    tensor(9.5861)
    >>> linear_to_db(torch.tensor(100.), torch.tensor(1.), soft_max_SDR_to_eps(20))
    tensor(16.9897)
    >>> linear_to_db(torch.tensor(100.), torch.tensor(1.), soft_max_SDR_to_eps(100))
    tensor(20.)

    >>> import numpy as np
    >>> linear_to_db(torch.tensor(1.), torch.tensor(np.finfo(np.float64).eps))
    tensor(156.5356, dtype=torch.float64)
    >>> linear_to_db(torch.tensor(1.), torch.tensor(np.finfo(np.float32).eps))
    tensor(69.2369)

    """
    # ToDo: add an assert that works for tensors
    # assert denominator > 0, denominator
    if eps is None:
        return 10 * torch.log10(numerator / denominator)
    else:
        return -10 * torch.log10(denominator / numerator + eps)


def ci_sdr_loss(
        estimation,  # K x T
        reference,  # K x T
        *,
        compute_permutation=True,
        filter_length=512,
        soft_max_SDR=None,
):
    """Convolutive transfer function Invariant Signal-to-Distortion Ratio loss

    Note:
        To follow the pytorch convention, this function has as first argument
        the estimation, while `ci_sdr` follows the convention for many metrics,
        that use as first argument the reference.

    The difference to ci_sdr are:
     - Change the sign, so this function can be minimized by an NN to reach the
       optimum.

    Args:
        estimation: ... x source x samples
        reference: ... x source x samples
        compute_permutation: If true, assume estimation source index is
            permuted. Note mir_eval.separation.bss_eval_sources computes
            the permutation based on the SIR, while this function computes the
            permutation based on the SDR.
        filter_length:
        soft_max_SDR:

    Returns:

    """
    return ci_sdr(
        **locals(),
        change_sign=True
    )


def ci_sdr_loss_hungarian(
        estimation,  # K x T
        reference,  # K x T
        *,
        # compute_permutation=True,
        filter_length=512,
        soft_max_SDR=None,
        algorithm='optimal',  # 'optimal', 'greedy'
):
    """Convolutive transfer function Invariant Signal-to-Distortion Ratio loss

    The `ci_sdr_loss` function is more efficient for low number of speakers,
    while this function has advantages for larger number of speakers.
    The optimization of this function is, that instead of triing each
    permutation, it uses the idea of the Hungarian algorithm to decompose the
    permutation problem in the calculation of a loss/score matrix and a
    "linear sum assignment" problem (scipy.optimize.linear_sum_assignment).

    Args:
        estimation: ... x source x samples
        reference: ... x source x samples
        filter_length:
        soft_max_SDR:
        algorithm: Either 'optimal' or 'greedy'
         - 'optimal': Use scipy.optimize.linear_sum_assignment to find the
                      optimal assignment
         - 'greedy' : Use a gready approach to find a good assignment.
                      ToDo: Check thesis: greedy is better for large number of
                                          speakers.

    Returns:

    Example:
        >>> reference = torch.tensor([[1., 2, 1, 2], [4, 3, 2, 1], [1, 2, 3, 4]])
        >>> estimation = torch.tensor([[1., 2, 3, 4], [1, 2, 1, 2], [4, 3, 2, 1]])
        >>> ci_sdr_loss_hungarian(estimation, reference, filter_length=2)
        tensor([-144.0805, -145.4635, -143.0331])
        >>> ci_sdr_loss(estimation, reference, filter_length=2)
        tensor([-144.0805, -145.4635, -143.0331])
        >>> estimation = torch.tensor([[1., 2, 2, 4], [1, 1, 1, 2], [4, 2, 2, 1]])
        >>> ci_sdr_loss_hungarian(estimation, reference, filter_length=2)
        tensor([-10.3300, -17.2712, -15.4051])
        >>> ci_sdr_loss(estimation, reference, filter_length=2)
        tensor([-10.3300, -17.2712, -15.4051])
        >>> ci_sdr_loss_hungarian(estimation[None], reference[None], filter_length=2)
        tensor([[-10.3300, -17.2712, -15.4051]])

    """
    from padertorch.ops.losses.source_separation import pit_loss_from_loss_matrix

    estimation = estimation[..., None, :, :]
    reference = reference[..., :, None, :]

    loss_matrix = ci_sdr_loss(
        estimation=estimation,
        reference=reference,
        filter_length=filter_length,
        soft_max_SDR=soft_max_SDR,
        compute_permutation=False,
    )

    if len(loss_matrix.shape) == 2:
        return pit_loss_from_loss_matrix(
            loss_matrix,
            reduction=None,
            algorithm=algorithm  # 'optimal', 'greedy'
        )
    else:
        loss_matrix_flat = einops.rearrange(loss_matrix, '... k1 k2 -> (...) k1 k2')
        loss = torch.stack([pit_loss_from_loss_matrix(
            l,
            reduction=None,
            algorithm=algorithm  # 'optimal', 'greedy'
        ) for l in loss_matrix_flat])
        return loss.reshape(*loss_matrix.shape[:-1])


def _is_broadcastable(shape1, shape2):
    # https://stackoverflow.com/a/24769712/5766934
    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def ci_sdr(
        reference,  # ... x K x T
        estimation,  # ... x K x T
        *,
        compute_permutation=True,
        change_sign=False,
        filter_length=512,
        soft_max_SDR=None,
):
    """Convolutive transfer function Invariant Signal-to-Distortion Ratio

    With the default arguments, this functions returns the same value as the
    SDR from `mir_eval.separation.bss_eval_sources`.

    Args:
        reference: ... x source x samples
        estimation: ... x source x samples
        compute_permutation: If true, assume estimation source index is
            permuted. Note mir_eval.separation.bss_eval_sources computes
            the permutation based on the SIR, while this function computes the
            permutation based on the SDR.
        change_sign:
            When True, assume this function is used as loss and return `-SDR`
            instead of `SDR`.
        filter_length:
        soft_max_SDR: ToDo: Was it first proposed in mixture of mixture?

    Returns:
        SDR values for each source

    >>> import numpy as np
    >>> import paderbox as pb

    >>> from paderbox.testing.testfile_fetcher import fetch_file_from_url

    >>> prefix = 'https://github.com/fgnt/pb_test_data/raw/master/bss_data/reverberation/'
    >>> audio_data = {
    ...     file.split('.')[0]: pb.io.load(fetch_file_from_url(prefix + file))
    ...     for file in [
    ...         'speech_source_0.wav',  # speaker 0
    ...         'speech_source_1.wav',  # speaker 1
    ...         'speech_reverberation_early_0.wav',  # reverberated signal, speaker 0
    ...         'speech_reverberation_early_1.wav',  # reverberated signal, speaker 1
    ...         'speech_image_0.wav',  # reverberated signal, speaker 0
    ...         'speech_image_1.wav',  # reverberated signal, speaker 1
    ...         'observation.wav',
    ...     ]
    ... }
    >>> ref_channel = 0
    >>> reference = np.array([audio_data['speech_source_0'], audio_data['speech_source_1']])
    >>> estimation = np.array([audio_data['speech_image_0'][ref_channel, :], audio_data['speech_image_1'][ref_channel, :]])

    >>> reference.shape, estimation.shape
    ((2, 38520), (2, 38520))
    >>> reference_pt = torch.as_tensor(reference)
    >>> estimation_pt = torch.as_tensor(estimation)

    >>> import pb_bss
    >>> pb_bss.evaluation.mir_eval_sources(reference, estimation)[0]
    array([12.60576235, 12.45027328])
    >>> ci_sdr(reference_pt, estimation_pt)
    tensor([12.6058, 12.4503], dtype=torch.float64)
    >>> ci_sdr(reference_pt, estimation_pt, compute_permutation=True)
    tensor([12.6058, 12.4503], dtype=torch.float64)
    >>> ci_sdr(reference_pt, estimation_pt[[0, 1]], compute_permutation=False)
    tensor([12.6058, 12.4503], dtype=torch.float64)
    >>> ci_sdr(reference_pt, estimation_pt[[1, 0]], compute_permutation=False)
    tensor([-23.5670, -25.1648], dtype=torch.float64)
    >>> ci_sdr(reference_pt, estimation_pt[[0, 1]], compute_permutation=True)
    tensor([12.6058, 12.4503], dtype=torch.float64)
    >>> ci_sdr(reference_pt, estimation_pt[[1, 0]], compute_permutation=True)
    tensor([12.6058, 12.4503], dtype=torch.float64)
    >>> ci_sdr(reference_pt, estimation_pt[[0, 1]], compute_permutation=True, change_sign=True)
    tensor([-12.6058, -12.4503], dtype=torch.float64)

    >>> ci_sdr(reference_pt, estimation_pt, soft_max_SDR=20)
    tensor([11.8788, 11.7469], dtype=torch.float64)
    >>> ci_sdr(reference_pt, reference_pt, soft_max_SDR=20)
    tensor([20., 20.], dtype=torch.float64)
    >>> sdrs = ci_sdr(reference_pt, reference_pt, soft_max_SDR=None)
    ... # tensor([245.8194, 282.1901], dtype=torch.float64)  # old pytorch
    ... # tensor([253.4707, 279.4020], dtype=torch.float64)  # new pytorch
    >>> sdrs > 200, sdrs < 300
    (tensor([True, True]), tensor([True, True]))


    >>> estimation = audio_data['observation'][:2]
    >>> pb_bss.evaluation.mir_eval_sources(reference, estimation)[0]
    array([ 1.83304215, -2.79861495])
    >>> e = torch.tensor(estimation, requires_grad=True)
    >>> sdr = ci_sdr(reference_pt, e)
    >>> sdr  # doctest: +ELLIPSIS
    tensor([ 1.8330, -2.7986], dtype=torch.float64, grad_fn=<SelectBackward...>)
    >>> sdr.sum().backward()
    >>> e.grad
    tensor([[-2.7294e-06, -5.2814e-06, -3.2224e-05,  ..., -8.6633e-05,
             -1.3574e-04, -7.0333e-06],
            [-3.0444e-06,  3.5137e-06,  4.8881e-06,  ...,  3.4590e-06,
              4.3444e-05,  6.0922e-06]], dtype=torch.float64)

    Comparison with si_sdr and sdr. Note, we must change reference to a
    reverberated signal (speech_image), otherwise we get very bad values for
    both objectives.

    >>> reference = np.array([audio_data['speech_reverberation_early_0'][ref_channel, :], audio_data['speech_reverberation_early_1'][ref_channel, :]])
    >>> estimation = np.array([audio_data['speech_image_0'][ref_channel, :], audio_data['speech_image_1'][ref_channel, :]])

    # >>> reference = audio_data['speech_image'][(0, 1), (0, 1), :]
    # >>> estimation = audio_data['observation'][:2, :] + reference
    >>> reference.shape, estimation.shape
    ((2, 38520), (2, 38520))
    >>> reference_pt = torch.as_tensor(reference)
    >>> estimation_pt = torch.as_tensor(estimation)

    >>> from padertorch.ops.losses.regression import sdr_loss, si_sdr_loss
    >>> si_sdr_loss(estimation_pt, reference_pt, reduction=None)
    tensor([-9.9188, -9.5530], dtype=torch.float64)
    >>> ci_sdr(reference_pt, estimation_pt, soft_max_SDR=None, filter_length=1, change_sign=True)
    tensor([-9.9188, -9.5530], dtype=torch.float64)
    >>> sdr_loss(estimation_pt, reference_pt, reduction=None)
    tensor([-9.9814, -9.4490], dtype=torch.float64)
    >>> ci_sdr(reference_pt, estimation_pt, soft_max_SDR=None, filter_length=0, change_sign=True)
    tensor([-9.9814, -9.4490], dtype=torch.float64)
    """
    if len(reference.shape) == 1:
        assert len(estimation.shape) == 1, (reference.shape, estimation.shape)
        single_source = True
        reference = reference[None, :]
        estimation = estimation[None, :]
    else:
        single_source = False

    *_, K, num_samples = reference.shape

    if K > 1 and compute_permutation:
        assert reference.shape == estimation.shape, (reference.shape, estimation.shape)
        # ToDo: Add option to use hungarian algorithm
        #       Note:
        #        - The hungarian algorithm cannot release intermediate tensors
        #        - The hungarien algorithm has advantages for K > 3

        axis = -2
        candidates = []
        indexer = [slice(None), ] * estimation.ndim
        permutations = list(itertools.permutations(range(K)))
        # Does torch drop a tensor after `torch.minimum(a, b)`, when both are
        # scalars? (i.e. release the memory).
        #  - If yes, optimize the following loop
        #  - If no, min(a, b) would be a option, but it would introduce a
        #    cuda sync point, hence it is a tradeoff
        for permutation in permutations:
            indexer[axis] = permutation
            candidates.append(ci_sdr(
                reference,
                estimation[tuple(indexer)],
                change_sign=False,
                filter_length=filter_length, soft_max_SDR=soft_max_SDR,
                compute_permutation=False,
            ))
        if len(reference.shape) == 2:  # No batch axis
            candidates = torch.stack(candidates)
            _, idx = torch.max(torch.sum(candidates, axis=-1), dim=0)
            sdr = candidates[idx]
        else:
            candidates = torch.stack(candidates)
            candidates_flat = einops.rearrange(
                candidates,
                'permutations ... k -> permutations (...) k'
            )
            _, idx = torch.max(torch.sum(candidates_flat, axis=-1), dim=0)
            batch_size = candidates_flat.shape[1]
            sdr = candidates_flat[idx, range(batch_size), :]
            sdr = sdr.reshape(*candidates.shape[1:])
        if change_sign:
            return -sdr
        else:
            return sdr

    assert _is_broadcastable(reference.shape, estimation.shape), (reference.shape, estimation.shape)
    assert reference.shape[-1] == estimation.shape[-1], (reference.shape, estimation.shape)

    est = estimation
    if filter_length != 0:
        reverberated = wiener_filter_predict_single_input(
            reference, estimation, filter_length=filter_length)
        est = torch.nn.functional.pad(est, [0, filter_length - 1])
    else:
        reverberated = reference
    num = torch.sum(reverberated**2, dim=-1)
    den = torch.sum((reverberated - est)**2, dim=-1)
    scores = linear_to_db(num, den, eps=soft_max_SDR_to_eps(soft_max_SDR))
    if change_sign:
        scores = -scores

    if single_source:
        scores = torch.squeeze(scores, dim=-1)

    return scores
