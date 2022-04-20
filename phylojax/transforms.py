import jax.numpy as jnp
from jax.nn import log_sigmoid, softplus
from jax.scipy.special import expit, logit


class SigmoidTransform:
    def __call__(self, x):
        finfo = jnp.finfo(jnp.result_type(x))
        return jnp.clip(expit(x), a_min=finfo.tiny, a_max=1.0 - finfo.eps)

    def inverse(self, y):
        return logit(y)

    def log_abs_det_jacobian(self, x, y):
        return -softplus(x) - softplus(-x)


class StickBreakingTransform:
    """Code from numpyro."""

    def __call__(self, x):
        # we shift x to obtain a balanced mapping
        # (0, 0, ..., 0) -> (1/K, 1/K, ..., 1/K)
        x = x - jnp.log(x.shape[-1] - jnp.arange(x.shape[-1]))
        # convert to probabilities (relative to the remaining)
        # of each fraction of the stick
        z = _clipped_expit(x)
        z1m_cumprod = jnp.cumprod(1 - z, axis=-1)
        pad_width = [(0, 0)] * x.ndim
        pad_width[-1] = (0, 1)
        z_padded = jnp.pad(z, pad_width, mode="constant", constant_values=1.0)
        pad_width = [(0, 0)] * x.ndim
        pad_width[-1] = (1, 0)
        z1m_cumprod_shifted = jnp.pad(
            z1m_cumprod, pad_width, mode="constant", constant_values=1.0
        )
        return z_padded * z1m_cumprod_shifted

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        # https://mc-stan.org/docs/2_19/reference-manual/simplex-transform-section.html
        # |det|(J) = Product(y * (1 - sigmoid(x)))
        #          = Product(y * sigmoid(x) * exp(-x))
        x = x - jnp.log(x.shape[-1] - jnp.arange(x.shape[-1]))
        return jnp.sum(jnp.log(y[..., :-1]) + (log_sigmoid(x) - x), axis=-1)


def _clipped_expit(x):
    finfo = jnp.finfo(jnp.result_type(x))
    return jnp.clip(expit(x), a_min=finfo.tiny, a_max=1.0 - finfo.eps)
