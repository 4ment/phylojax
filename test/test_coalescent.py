import jax.numpy as np
import pytest

from phylojax.coalescent import ConstantCoalescent


def test_constant():
    sampling_times = np.array([0.0, 1.0, 2.0])
    coalescent = ConstantCoalescent(np.array([20.0]))
    log_p = coalescent.log_prob(np.concatenate((sampling_times, np.array([4.0, 5.0]))))
    assert -6.3915 == pytest.approx(float(log_p), 0.001)


def test_constant_batch():
    sampling_times = np.zeros((2, 4))
    thetas = np.array([[3.0], [6.0]])
    heights = np.array([[2.0, 6.0, 12.0], [16, 24, 24]])
    constant = ConstantCoalescent(thetas)
    log_p = constant.log_prob(np.concatenate((sampling_times, heights), -1))
    assert np.allclose(
        np.array([[-13.295836866], [-25.375278407684164]]), log_p, rtol=1e-05
    )
