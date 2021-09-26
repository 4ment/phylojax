import jax.numpy as np
from jax.nn import softplus
from jax.scipy.special import expit


class SigmoidTransform:
    def __call__(self, x):
        finfo = np.finfo(np.result_type(x))
        return np.clip(expit(x), a_min=finfo.tiny, a_max=1.0 - finfo.eps)

    def log_abs_det_jacobian(self, x, y):
        return -softplus(x) - softplus(-x)
