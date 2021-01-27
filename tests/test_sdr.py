from dataclasses import dataclass

import einops
import numpy as np
import torch

import pb_bss
import paderbox as pb
from paderbox.testing.testfile_fetcher import fetch_file_from_url

import ci_sdr


def get_data():
    prefix = 'https://github.com/fgnt/pb_test_data/raw/master/bss_data/reverberation/'

    def load(file):
        if isinstance(file, (tuple, list)):
            return np.array([load(f) for f in file])
        else:
            return pb.io.load(fetch_file_from_url(prefix + file))

    @dataclass
    class Data:
        speech_source: np.array
        speech_reverberation_early: np.array
        speech_image: np.array
        observation: np.array

    return Data(
        speech_source=load(['speech_source_0.wav', 'speech_source_1.wav']),
        speech_reverberation_early=load([
            'speech_reverberation_early_0.wav',
            'speech_reverberation_early_1.wav']),
        speech_image=load(['speech_image_0.wav', 'speech_image_1.wav']),
        observation=load('observation.wav'),
    )


def test_vs_mir_eval():
    data = get_data()
    ref_channel = 0
    speakers = 2

    reference = data.speech_source
    estimation = data.speech_reverberation_early[:, ref_channel, :]
    mir_eval_sdr = pb_bss.evaluation.mir_eval_sources(reference, estimation)[0]
    sdr = ci_sdr.pt.ci_sdr(torch.tensor(reference), torch.tensor(estimation)).numpy()
    np.testing.assert_allclose(mir_eval_sdr, sdr)
    np.testing.assert_allclose(sdr, [61.984948, 44.553148])

    reference = data.speech_source
    estimation = data.speech_image[:, ref_channel, :]
    mir_eval_sdr = pb_bss.evaluation.mir_eval_sources(reference, estimation)[0]
    sdr = ci_sdr.pt.ci_sdr(torch.tensor(reference), torch.tensor(estimation)).numpy()
    np.testing.assert_allclose(mir_eval_sdr, sdr)
    np.testing.assert_allclose(sdr, [12.605762, 12.450273])

    reference = data.speech_source
    estimation = data.observation[:speakers, :]
    mir_eval_sdr = pb_bss.evaluation.mir_eval_sources(reference, estimation, compute_permutation=True)[0]
    sdr = ci_sdr.pt.ci_sdr(torch.tensor(reference), torch.tensor(estimation), compute_permutation=True).numpy()
    np.testing.assert_allclose(mir_eval_sdr, sdr)
    np.testing.assert_allclose(sdr, [1.833042, -2.798615])
    mir_eval_sdr = pb_bss.evaluation.mir_eval_sources(reference, estimation, compute_permutation=False)[0]
    sdr = ci_sdr.pt.ci_sdr(torch.tensor(reference), torch.tensor(estimation), compute_permutation=False).numpy()
    np.testing.assert_allclose(mir_eval_sdr, sdr)
    np.testing.assert_allclose(sdr, [1.833042, -2.798615])

    reference = data.speech_source
    estimation = data.observation[(1, 0), :]
    mir_eval_sdr = pb_bss.evaluation.mir_eval_sources(reference, estimation, compute_permutation=True)[0]
    sdr = ci_sdr.pt.ci_sdr(torch.tensor(reference), torch.tensor(estimation), compute_permutation=True).numpy()
    np.testing.assert_allclose(mir_eval_sdr, sdr)
    np.testing.assert_allclose(sdr, [1.833042, -2.798615])
    mir_eval_sdr = pb_bss.evaluation.mir_eval_sources(reference, estimation, compute_permutation=False)[0]
    sdr = ci_sdr.pt.ci_sdr(torch.tensor(reference), torch.tensor(estimation), compute_permutation=False).numpy()
    np.testing.assert_allclose(mir_eval_sdr, sdr)
    np.testing.assert_allclose(sdr, [1.829307, -2.931903], rtol=1e-6)

    # For a perfect estimate, they are different, but in that region, it does
    # not matter, 200 dB is pretty large.
    reference = data.speech_source
    estimation = data.speech_source
    mir_eval_sdr = pb_bss.evaluation.mir_eval_sources(reference, estimation)[0]
    sdr = ci_sdr.pt.ci_sdr(torch.tensor(reference), torch.tensor(estimation)).numpy()
    # These numbers aren't reproducible across different package versions.
    # Note: An error of `np.finfo(np.float64).eps` means 156.5356 dB and
    #       an error of `np.finfo(np.float32).eps` means 69.2369 dB
    np.testing.assert_allclose(mir_eval_sdr, [245.530192, 266.500027], atol=10)
    np.testing.assert_allclose(sdr, [253.470667, 279.402013], atol=10)


def test_vs_sdr_si_sdr():
    from padertorch.ops.losses.regression import sdr_loss, si_sdr_loss
    from padertorch.ops.losses.source_separation import pit_loss

    data = get_data()
    ref_channel = 0

    # si_sdr and sdr need aligned reference signal, hence use
    # speech_reverberation_early as reference

    reference = torch.tensor(data.speech_reverberation_early[:, ref_channel, :])
    estimation = torch.tensor(data.speech_image[:, ref_channel, :])
    ref_sdr = si_sdr_loss(estimation, reference, reduction=None).numpy()
    sdr = ci_sdr.pt.ci_sdr_loss(estimation, reference, filter_length=1).numpy()
    np.testing.assert_allclose(ref_sdr, sdr)
    np.testing.assert_allclose(sdr, [-9.918769, -9.55302], rtol=1e-6)

    # Permutation test:
    #     si_sdr(...) -> pit_loss(..., axis=0, loss_fn=si_sdr_loss)
    reference = torch.tensor(data.speech_reverberation_early[(1, 0), ref_channel, :])
    estimation = torch.tensor(data.speech_image[(0, 1), ref_channel, :])
    ref_sdr = pit_loss(estimation, reference, axis=0, loss_fn=si_sdr_loss).numpy()
    sdr = ci_sdr.pt.ci_sdr_loss(estimation, reference, filter_length=1).numpy()
    np.testing.assert_allclose(ref_sdr, np.mean(sdr), rtol=1e-6)
    np.testing.assert_allclose(ref_sdr, -9.735894, rtol=1e-6)
    np.testing.assert_allclose(sdr, [-9.55302, -9.918769], rtol=1e-6)

    reference = torch.tensor(data.speech_reverberation_early[:, ref_channel, :])
    estimation = torch.tensor(data.speech_image[:, ref_channel, :])
    ref_sdr = sdr_loss(estimation, reference, reduction=None).numpy()
    sdr = ci_sdr.pt.ci_sdr_loss(estimation, reference, filter_length=0, compute_permutation=False).numpy()
    np.testing.assert_allclose(ref_sdr, sdr)
    np.testing.assert_allclose(sdr, [-9.981357, -9.448952], rtol=1e-6)

    # Permutation test:
    #     sdr_loss(...) -> pit_loss(..., axis=0, loss_fn=sdr_loss)
    reference = torch.tensor(data.speech_reverberation_early[(1, 0), ref_channel, :])
    estimation = torch.tensor(data.speech_image[(0, 1), ref_channel, :])
    ref_sdr = pit_loss(estimation, reference, axis=0, loss_fn=sdr_loss).numpy()
    sdr = ci_sdr.pt.ci_sdr_loss(estimation, reference, filter_length=0).numpy()
    np.testing.assert_allclose(ref_sdr, np.mean(sdr))
    np.testing.assert_allclose(ref_sdr, -9.715154, rtol=1e-6)
    np.testing.assert_allclose(sdr, [-9.448952, -9.981357], rtol=1e-6)


def test_batched():
    data = get_data()
    ref_channel = 0

    permutation = [1, 0]

    reference = np.array([data.speech_source, data.speech_source[permutation]])
    estimation = np.array([
        data.speech_reverberation_early[:, ref_channel, :],
        data.speech_image[permutation, ref_channel, :]])
    sdr = ci_sdr.pt.ci_sdr(torch.tensor(reference), torch.tensor(estimation), compute_permutation=False).numpy()
    sdr_loopy = [
        ci_sdr.pt.ci_sdr(torch.tensor(reference[0]), torch.tensor(estimation[0]), compute_permutation=False).numpy(),
        ci_sdr.pt.ci_sdr(torch.tensor(reference[1]), torch.tensor(estimation[1]), compute_permutation=False).numpy(),
    ]
    np.testing.assert_allclose(sdr_loopy, sdr)
    np.testing.assert_allclose(
        sdr, [[61.984948, 44.553148], [12.450273, 12.605762]])

    sdr_perm = ci_sdr.pt.ci_sdr(torch.tensor(reference), torch.tensor(estimation), compute_permutation=True).numpy()
    np.testing.assert_allclose(sdr_loopy, sdr_perm)

    estimation = np.array([
        data.speech_reverberation_early[(1, 0), ref_channel, :],
        data.speech_image[(1, 0), ref_channel, :]])
    sdr_perm = ci_sdr.pt.ci_sdr(torch.tensor(reference), torch.tensor(estimation), compute_permutation=True).numpy()
    np.testing.assert_allclose(sdr_loopy, sdr_perm)

    estimation = np.array([
        data.speech_reverberation_early[(1, 0), ref_channel, :],
        data.speech_image[(0, 1), ref_channel, :]])
    sdr_perm = ci_sdr.pt.ci_sdr(torch.tensor(reference), torch.tensor(estimation), compute_permutation=True).numpy()
    np.testing.assert_allclose(sdr_loopy, sdr_perm)

    estimation = np.array([
        data.speech_reverberation_early[(0, 1), ref_channel, :],
        data.speech_image[(0, 1), ref_channel, :]])
    sdr_perm = ci_sdr.pt.ci_sdr(torch.tensor(reference), torch.tensor(estimation), compute_permutation=True).numpy()
    np.testing.assert_allclose(sdr_loopy, sdr_perm)

    sdr_perm = ci_sdr.pt.ci_sdr(torch.tensor(reference)[None], torch.tensor(estimation)[None], compute_permutation=True).numpy()
    np.testing.assert_allclose(sdr_loopy, np.squeeze(sdr_perm, axis=0))

    rng = np.random.RandomState(0)
    reference = torch.tensor(rng.normal(size=[3, 5, 7, 2, 100]))
    estimation = torch.tensor(rng.normal(size=[3, 5, 7, 2, 100]))

    sdr = ci_sdr.pt.ci_sdr(reference, estimation, compute_permutation=False, filter_length=16).numpy()
    sdr_loopy = [
        ci_sdr.pt.ci_sdr(r, e, compute_permutation=False, filter_length=16).numpy()
        for r, e in zip(
            einops.rearrange(reference, '... sources time -> (...) sources time'),
            einops.rearrange(estimation, '... sources time -> (...) sources time'),
        )
    ]
    sdr_loopy = einops.rearrange(
        sdr_loopy, '(a b c) sources -> a b c sources', a=3, b=5, c=7)
    np.testing.assert_allclose(sdr, sdr_loopy, rtol=1e-6, atol=1e-6)

    sdr = ci_sdr.pt.ci_sdr(reference, estimation, compute_permutation=True, filter_length=16).numpy()
    # With permutation solving, the solution is different
    np.testing.assert_allclose(np.sqrt(np.mean((sdr - sdr_loopy)**2)), 1.781779, rtol=1e-6, atol=1e-6)

    sdr_loopy = [
        ci_sdr.pt.ci_sdr(r, e, compute_permutation=True, filter_length=16).numpy()
        for r, e in zip(
            einops.rearrange(reference, '... sources time -> (...) sources time'),
            einops.rearrange(estimation, '... sources time -> (...) sources time'),
        )
    ]
    sdr_loopy = einops.rearrange(
        sdr_loopy, '(a b c) sources -> a b c sources', a=3, b=5, c=7)
    np.testing.assert_allclose(sdr, sdr_loopy, rtol=1e-6, atol=1e-6)
