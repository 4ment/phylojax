import jax
import pytest
import jax.numpy as np

from phylojax.substitution import JC69, GTR
from jax.config import config
config.update("jax_enable_x64", True)

@pytest.mark.parametrize("t", [0.001, 0.1])
def test_JC69(t):
    ii = 1. / 4. + 3. / 4. * np.exp(- 4. / 3. * t)
    ij = 1./4. - 1./4. * np.exp(- 4./3. * t)
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
        [[[
            [0.93717830, 0.009506685, 0.047505899, 0.005809115],
            [0.02640748, 0.894078744, 0.006448058, 0.073065722],
            [0.16158572, 0.007895626, 0.820605951, 0.009912704],
            [0.01344433, 0.060875872, 0.006744752, 0.918935042],
        ]]]
    )
    assert jax.numpy.allclose(P, P_expected, rtol=1e-05)