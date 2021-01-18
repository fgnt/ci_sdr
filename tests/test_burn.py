import torch
import ci_sdr


def test_burn_single_source():
    t1 = torch.tensor([1., 2, 4, 7, 1, 3, 7, 8, 0, 3, 4])
    t2 = torch.clone(t1)
    t2[:4] += 2

    sdr = ci_sdr.pt.ci_sdr(t1, t2, filter_length=3)
    assert sdr.shape == (), sdr.shape
    torch.testing.assert_allclose(sdr, 13.592828750610352)


def test_burn_multi_source():
    t1 = torch.tensor([
        [1., 2, 4, 7, 1, 3, 7, 8, 0, 3, 4],
        [5., 2, 7, 9, 3, 8, 4, 2, 9, 4, 5],
    ])
    t2 = torch.clone(t1)
    t2[:, :4] += 2

    sdr = ci_sdr.pt.ci_sdr(t1, t2, filter_length=3, compute_permutation=False)
    assert sdr.shape == (2,), sdr.shape
    torch.testing.assert_allclose(sdr, [13.592828750610352, 17.48115348815918])

    sdr = ci_sdr.pt.ci_sdr(
        t1, t2[(1, 0), :], filter_length=3, compute_permutation=True)
    assert sdr.shape == (2,), sdr.shape
    torch.testing.assert_allclose(sdr, [13.592828750610352, 17.48115348815918])
