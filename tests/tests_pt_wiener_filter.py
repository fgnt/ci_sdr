import numpy as np
import scipy.signal
import torch

from mir_eval.separation import _project
from ci_sdr.pt.wiener_filter import wiener_filter_predict_single_input


def get_filter_and_estimate(
        x,
        y,
        filter_length,
        first_filter_index,
):
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)

    b_est = wiener_filter_predict_single_input(
        torch.tensor(x).to(torch.float32),
        torch.tensor(y).to(torch.float32),
        filter_length,
        return_w=True,
        _native_complex=True,
        first_filter_index=first_filter_index,
    )
    b_est = b_est.numpy()

    y_est = wiener_filter_predict_single_input(
        torch.tensor(x).to(torch.float32),
        torch.tensor(y).to(torch.float32),
        filter_length,
        return_w=False,
        _native_complex=True,
        first_filter_index=first_filter_index,
    )
    y_est = y_est.numpy()
    return b_est, y_est


def test_causal():
    b = [1, 1]
    a = [1]
    x = [1., -1, 1, -1, 1, -1, 0]
    filter_length = 2
    first_filter_index = 0

    x = np.array(x)
    y = scipy.signal.lfilter(b, a, x)

    np.testing.assert_equal(y, [1, 0, 0, 0, 0, 0, -1])

    b_est, y_est = get_filter_and_estimate(
        x=x, y=y, filter_length=filter_length,
        first_filter_index=first_filter_index,
    )

    y_est_mir_eval = _project(x[None, :], y, filter_length)

    np.testing.assert_allclose(b_est, b, atol=1e-6)

    # The returned signal used full mode convolution, while lfilter
    # use some kind of same mode.
    y = scipy.signal.convolve(x, b, mode='full')
    np.testing.assert_allclose(y_est_mir_eval, y, atol=1e-6)
    np.testing.assert_allclose(y_est, y, atol=1e-6)


def test_filtfilt():
    b = [1, 1]
    a = [1]
    x = [0, 0, 1., -1, 1, -1, 1, -1, 0, 0]
    filter_length = 3
    first_filter_index = -1

    x = np.array(x)
    y = scipy.signal.filtfilt(b, a, x)

    effective_b = [1, 2, 1]

    np.testing.assert_equal(y, [0, 1, 1, 0, 0, 0, 0, -1, -1, 0])

    b_est, y_est = get_filter_and_estimate(
        x=x, y=y, filter_length=filter_length,
        first_filter_index=first_filter_index,
    )

    np.testing.assert_allclose(b_est, effective_b, atol=1e-6)

    y = scipy.signal.convolve(
            scipy.signal.convolve(x, b, mode='full')[::-1], b
    )[::-1]
    np.testing.assert_allclose(y_est, y, atol=1e-6)


def test_delay():
    b = [0, 1, 1]
    a = [1]
    x = [1., -1, 1, -1, 1, -1, 0, 0]
    filter_length = 2
    first_filter_index = 1

    x = np.array(x)
    y = scipy.signal.lfilter(b, a, x)

    np.testing.assert_equal(y, [0, 1, 0, 0, 0, 0, 0, -1])

    b_est, y_est = get_filter_and_estimate(
        x=x, y=y, filter_length=filter_length,
        first_filter_index=first_filter_index,
    )

    y_est_mir_eval = _project(x[None, :], y, first_filter_index + filter_length)

    np.testing.assert_allclose([0] * first_filter_index + list(b_est), b, atol=1e-6)

    # The returned signal used full mode convolution, while lfilter
    # use some kind of same mode.
    y = scipy.signal.convolve(x, b, mode='full')
    np.testing.assert_allclose(y_est_mir_eval, y, atol=1e-6)
    y = scipy.signal.convolve(x, [1, 1], mode='full')
    np.testing.assert_allclose(y_est, y, atol=1e-6)
