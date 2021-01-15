import itertools
import torch
from ci_sdr.pt.wiener_filter import wiener_filter_predict_single_input


def soft_max_SDR_to_eps(soft_max_SDR):
    if soft_max_SDR is None:
        return None
    else:
        assert soft_max_SDR > 5, soft_max_SDR
        return 10 ** (-soft_max_SDR / 10)


def linear_to_db(numerator, denominator, eps=None):
    """
    >>> linear_to_db(torch.tensor(100), torch.tensor(1))
    tensor(20.)
    >>> linear_to_db(torch.tensor(100), torch.tensor(1), soft_max_SDR_to_eps(10))
    tensor(9.5861)
    >>> linear_to_db(torch.tensor(100), torch.tensor(1), soft_max_SDR_to_eps(20))
    tensor(16.9897)
    >>> linear_to_db(torch.tensor(100), torch.tensor(1), soft_max_SDR_to_eps(100))
    tensor(20.)

    """
    assert denominator > 0, denominator
    if eps is None:
        return 10 * torch.log10(numerator / denominator)
    else:
        return -10 * torch.log10(denominator / numerator + eps)


def ci_sdr(
        reference,  # K x T
        estimation,  # K x T
        compute_permutation=False,
        change_sign=False,
        filter_length=512,
        soft_max_SDR=None,
):
    """Convolutive transfer function Invariant Signal-to-Distortion Ratio

    With the default arguments, this functions returns the same value as the
    SDR from `mir_eval.separation.bss_eval_sources`.

    Args:
        reference: source x samples
        estimation: source x samples
        compute_permutation: If true, assume estimation source index is
            permuted. Note mir_eval.separation.bss_eval_sources computes
            the permutation based on the SIR, while this function computes the
            permutation based on the SDR.
        change_sign:
            When True, assume this function is used as loss and return `-SDR`
            instead of `SDR`.

    Returns:
        SDR values for each source

    >>> import numpy as np
    >>> import paderbox as pb
    >>> import sms_wsj.database

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
    >>> ci_sdr(reference_pt, reference_pt, soft_max_SDR=None)
    tensor([253.4707, 279.4020], dtype=torch.float64)

    >>> estimation = audio_data['observation'][:2]
    >>> pb_bss.evaluation.mir_eval_sources(reference, estimation)[0]
    array([ 1.83304215, -2.79861495])
    >>> e = torch.tensor(estimation, requires_grad=True)
    >>> sdr = ci_sdr(reference_pt, e)
    >>> sdr
    tensor([ 1.8330, -2.7986], dtype=torch.float64, grad_fn=<StackBackward>)
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
    K, num_samples = reference.shape

    if compute_permutation:
        # ToDo: Add option to use hungarian algorithm
        #       Note:
        #        - The hungarian algorithm cannot release intermediate tensors
        #        - The hungarien algorithm has advantages for K > 3

        axis = 0
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
                filter_length=filter_length, soft_max_SDR=soft_max_SDR
            ))
        sdr, idx = torch.max(torch.stack(candidates), dim=0)
        if change_sign:
            return -sdr
        else:
            return sdr

    scores = []
    for k in range(K):
        est = estimation[k]

        if filter_length != 0:
            reverberated = wiener_filter_predict_single_input(
                reference[k], estimation[k], filter_length=filter_length)
            est = torch.nn.functional.pad(est, [0, filter_length-1])
        else:
            reverberated = reference[k]

        num = torch.sum(reverberated**2)
        den = torch.sum((reverberated - est)**2)

        scores.append(linear_to_db(num, den, eps=soft_max_SDR_to_eps(soft_max_SDR)))

    if change_sign:
        return -torch.stack(scores)
    else:
        return torch.stack(scores)
