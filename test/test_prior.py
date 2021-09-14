import numpy as np

from phylojax.prior import ctmc_scale


def test_ctmc_scale():
    branch_lengths = np.array(
        [
            [1.5000, 0.5000, 2.0000, 3.0000, 4.0000, 2.5000, 2.0000, 10.0000],
            [1.5000, 0.5000, 2.0000, 3.0000, 4.0000, 2.5000, 2.0000, 10.0000],
        ]
    )

    res = ctmc_scale(branch_lengths, np.array([[0.001], [0.001]]))
    assert np.allclose(np.full((2, 1), 4.475351922659342), res, rtol=1e-05)
