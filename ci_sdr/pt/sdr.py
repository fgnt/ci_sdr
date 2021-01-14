import torch


def linear_to_db(numerator, denominator):
    assert denominator > 0, denominator
    return 10 * torch.log10(numerator / denominator)


def py_mir_eval_sdr(
        reference,  # K x T
        estimation,  # K x T
        compute_permutation=False,
        change_sign=False,
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
            instead of `SDR`. This can save intermediate tensors.

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
    >>> py_mir_eval_sdr(torch.as_tensor(reference), torch.as_tensor(estimation))
    tensor([11.2143, 12.1095], dtype=torch.float64)
    >>> py_mir_eval_sdr(torch.as_tensor(reference), torch.as_tensor(estimation), compute_permutation=True)
    tensor([11.2143, 12.1095], dtype=torch.float64)
    >>> py_mir_eval_sdr(torch.as_tensor(reference), torch.as_tensor(estimation[[0, 1]]), compute_permutation=False)
    tensor([11.2143, 12.1095], dtype=torch.float64)
    >>> py_mir_eval_sdr(torch.as_tensor(reference), torch.as_tensor(estimation[[0, 1]]), compute_permutation=True)
    tensor([11.2143, 12.1095], dtype=torch.float64)
    >>> py_mir_eval_sdr(torch.as_tensor(reference), torch.as_tensor(estimation[[0, 1]]), compute_permutation=True, change_sign=True)
    tensor([-11.2143, -12.1095], dtype=torch.float64)

    >>> estimation = ex['audio_data']['observation'][:2]
    >>> pb_bss.evaluation.mir_eval_sources(reference, estimation)[0]
    array([-0.26370129, -0.57722483])
    >>> e = torch.tensor(estimation, requires_grad=True)
    >>> sdr = py_mir_eval_sdr(torch.as_tensor(reference), e)
    >>> sdr
    tensor([-0.2637, -0.5772], dtype=torch.float64, grad_fn=<StackBackward>)
    >>> sdr.sum().backward()
    >>> e.grad
    tensor([[-1.0439e-05, -2.4280e-04, -2.2361e-04,  ...,  6.3655e-04,
             -9.6265e-04,  3.0609e-04],
            [ 1.7144e-05,  6.1831e-04, -3.0613e-04,  ...,  5.1454e-05,
             -6.0106e-05,  2.1169e-04]], dtype=torch.float64)

    """
    from padercontrib.pytorch.ops.mir_eval_sdr.wiener_filter import pt_wiener_filter_predict_single_input

    if compute_permutation:
        from padertorch.ops.losses import pit_loss

        def loss_fn(estimation, reference):
            return py_mir_eval_sdr(reference, estimation, change_sign=True)

        if change_sign:
            return pit_loss(estimation, reference, loss_fn=loss_fn, axis=0)
        else:
            return -pit_loss(estimation, reference, loss_fn=loss_fn, axis=0)

    K = len(reference)

    filter_length = 512

    scores = []
    for k in range(K):
        reverberated = pt_wiener_filter_predict_single_input(
            reference[k], estimation[k], filter_length=filter_length)

        est = estimation[k]
        est = torch.nn.functional.pad(est, [0, filter_length-1])
        # [...,: reference.shape[-1]]

        num = torch.sum(reverberated**2)
        den = torch.sum((reverberated - est)**2)

        scores.append(linear_to_db(num, den))
    if change_sign:
        return -torch.stack(scores)
    else:
        return torch.stack(scores)