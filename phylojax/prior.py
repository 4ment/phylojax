import jax.numpy as np
from jax.lax import lgamma

_shape = np.array([0.5])
_log_gamma_one_half = lgamma(_shape)


def ctmc_scale(branch_lengths, rate):
    """Implement the CTMC scale prior on substitution rate [ferreira2008]_

    :param branch_lengths: branch length
    :param rate: substitution rate

    .. [ferreira2008] Ferreira and Suchard. Bayesian analysis of elapsed times
     in continuous-time Markov chains. 2008
    """
    total_tree_time = branch_lengths.sum(-1, keepdims=True)
    log_normalization = _shape * np.log(total_tree_time) - _log_gamma_one_half
    return log_normalization - _shape * np.log(rate) - rate * total_tree_time
