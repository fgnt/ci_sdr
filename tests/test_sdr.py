from dataclasses import dataclass

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
