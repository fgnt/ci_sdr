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
    """

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

    >>> import padercontrib as pc
    >>> import paderbox as pb
    >>> import sms_wsj.database
    >>> db = pc.database.sms_wsj.SmsWsj()
    >>> ds = db.get_dataset_train().map(sms_wsj.database.AudioReader(keys=['speech_source', 'speech_image', 'observation']))
    >>> ex = ds[0]
    >>> reference = ex['audio_data']['speech_source']
    >>> estimation = ex['audio_data']['speech_image'][:, 0, :]
    >>> reference.shape, estimation.shape
    ((2, 87663), (2, 87663))

    >>> import pb_bss
    >>> pb_bss.evaluation.mir_eval_sources(reference, estimation)[0]
    array([11.21430422, 12.10953126])
    >>> ci_sdr(torch.as_tensor(reference), torch.as_tensor(estimation))
    tensor([11.2143, 12.1095], dtype=torch.float64)
    >>> ci_sdr(torch.as_tensor(reference), torch.as_tensor(estimation), compute_permutation=True)
    tensor([11.2143, 12.1095], dtype=torch.float64)
    >>> ci_sdr(torch.as_tensor(reference), torch.as_tensor(estimation[[0, 1]]), compute_permutation=False)
    tensor([11.2143, 12.1095], dtype=torch.float64)
    >>> ci_sdr(torch.as_tensor(reference), torch.as_tensor(estimation[[0, 1]]), compute_permutation=True)
    tensor([11.2143, 12.1095], dtype=torch.float64)
    >>> ci_sdr(torch.as_tensor(reference), torch.as_tensor(estimation[[0, 1]]), compute_permutation=True, change_sign=True)
    tensor([-11.2143, -12.1095], dtype=torch.float64)

    >>> ci_sdr(torch.as_tensor(reference), torch.as_tensor(estimation), soft_max_SDR=20)
    tensor([10.6748, 11.4555], dtype=torch.float64)
    >>> ci_sdr(torch.as_tensor(reference), torch.as_tensor(reference), soft_max_SDR=20)
    tensor([20., 20.], dtype=torch.float64)
    >>> ci_sdr(torch.as_tensor(reference), torch.as_tensor(reference), soft_max_SDR=None)
    tensor([295.1350, 269.7644], dtype=torch.float64)

    >>> estimation = ex['audio_data']['observation'][:2]
    >>> pb_bss.evaluation.mir_eval_sources(reference, estimation)[0]
    array([-0.26370129, -0.57722483])
    >>> e = torch.tensor(estimation, requires_grad=True)
    >>> sdr = ci_sdr(torch.as_tensor(reference), e)
    >>> sdr
    tensor([-0.2637, -0.5772], dtype=torch.float64, grad_fn=<StackBackward>)
    >>> sdr.sum().backward()
    >>> e.grad
    tensor([[-1.0439e-05, -2.4280e-04, -2.2361e-04,  ...,  6.3655e-04,
             -9.6265e-04,  3.0609e-04],
            [ 1.7144e-05,  6.1831e-04, -3.0613e-04,  ...,  5.1454e-05,
             -6.0106e-05,  2.1169e-04]], dtype=torch.float64)

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
        reverberated = wiener_filter_predict_single_input(
            reference[k], estimation[k], filter_length=filter_length)

        est = estimation[k]
        est = torch.nn.functional.pad(est, [0, filter_length-1])
        # [...,: reference.shape[-1]]

        num = torch.sum(reverberated**2)
        den = torch.sum((reverberated - est)**2)

        scores.append(linear_to_db(num, den, eps=soft_max_SDR_to_eps(soft_max_SDR)))
    if change_sign:
        return -torch.stack(scores)
    else:
        return torch.stack(scores)
