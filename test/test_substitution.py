import jax
import jax.numpy as np
import pytest
from jax.config import config

from phylojax.substitution import GTR, JC69

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("t", [0.001, 0.1])
def test_JC69(t):
    ii = 1.0 / 4.0 + 3.0 / 4.0 * np.exp(-4.0 / 3.0 * t)
    ij = 1.0 / 4.0 - 1.0 / 4.0 * np.exp(-4.0 / 3.0 * t)
    subst_model = JC69()
    P = subst_model.p_t(np.expand_dims(np.array([t]), axis=-1))
    assert ii == pytest.approx(P[0, 0, 0].item(), 1.0e-6)
    assert ij == pytest.approx(P[0, 0, 1].item(), 1.0e-6)


def test_GTR():
    rates = np.array([0.060602, 0.402732, 0.028230, 0.047910, 0.407249, 0.053277])
    freqs = np.array([0.479367, 0.172572, 0.140933, 0.207128])
    subst_model = GTR(rates, freqs)
    P = subst_model.p_t(np.array([[0.1]]))
    P_expected = np.array(
        [
            [
                [
                    [0.93717830, 0.009506685, 0.047505899, 0.005809115],
                    [0.02640748, 0.894078744, 0.006448058, 0.073065722],
                    [0.16158572, 0.007895626, 0.820605951, 0.009912704],
                    [0.01344433, 0.060875872, 0.006744752, 0.918935042],
                ]
            ]
        ]
    )
    assert jax.numpy.allclose(P, P_expected, rtol=1e-05)


def test_GTR_batch():
    rates = np.array(
        [
            [0.060602, 0.402732, 0.028230, 0.047910, 0.407249, 0.053277],
            [1.0, 3.0, 1.0, 1.0, 3.0, 1.0],
        ]
    )
    pi = np.array(
        [
            [0.479367, 0.172572, 0.140933, 0.207128],
            [0.479367, 0.172572, 0.140933, 0.207128],
        ]
    )
    subst_model = GTR(rates, pi)
    branch_lengths = np.array([[0.1], [0.001]])
    P = subst_model.p_t(np.expand_dims(branch_lengths, -1))
    P_expected = np.array(
        [
            [
                [
                    [0.93717830, 0.009506685, 0.047505899, 0.005809115],
                    [0.02640748, 0.894078744, 0.006448058, 0.073065722],
                    [0.16158572, 0.007895626, 0.820605951, 0.009912704],
                    [0.01344433, 0.060875872, 0.006744752, 0.918935042],
                ]
            ],
            [
                [
                    [0.9992649548, 0.0001581235, 0.0003871353, 0.0001897863],
                    [0.0004392323, 0.9988625812, 0.0001291335, 0.0005690531],
                    [0.0013167952, 0.0001581235, 0.9983352949, 0.0001897863],
                    [0.0004392323, 0.0004741156, 0.0001291335, 0.9989575186],
                ]
            ],
        ]
    )
    assert jax.numpy.allclose(P, P_expected, rtol=1e-05)
