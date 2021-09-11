import pytest
import jax.numpy as np
from phylojax.coalescent import ConstantCoalescent


def test_constant():
    sampling_times = np.array([0., 1., 2.])
    coalescent = ConstantCoalescent(np.array([20.]))
    print(np.concatenate((sampling_times, np.array([4., 5.]))))
    log_p = coalescent.log_prob(np.concatenate((sampling_times, np.array([4., 5.]))))
    assert -6.3915 == pytest.approx(float(log_p), 0.001)
