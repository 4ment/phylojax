from functools import partial

import jax
import jax.numpy as np
from jax.ops import index, index_update

from .tree import heights_to_branch_lengths, transform_ratios


def calculate_treelikelihood(tip_partials, weights, indices, mats, freqs, props):
    partials = np.concatenate(
        (
            np.broadcast_to(
                np.expand_dims(np.expand_dims(tip_partials, 1), -3),
                (tip_partials.shape[0],)
                + mats.shape[:-4]
                + (props.shape[-3],)
                + tip_partials.shape[-2:],
            ),
            np.empty(
                (tip_partials.shape[0] - 1,)
                + mats.shape[:-4]
                + (props.shape[-3],)
                + tip_partials.shape[-2:]
            ),
        ),
        axis=0,
    )

    def fn(post_indexing, i, partials):
        node, left, right = np.array(post_indexing)[i]
        return index_update(
            partials,
            index[node],
            (mats[..., left, :, :, :] @ partials[left])
            * (mats[..., right, :, :, :] @ partials[right]),
        )

    fn2 = partial(fn, indices)
    partials = jax.lax.fori_loop(0, len(indices), fn2, partials)

    return np.sum(
        np.log(freqs @ np.sum(props * partials[indices[-1][0]], -3)) * weights,
        axis=-1,
    )


def jax_likelihood(
    subst_model,
    partials,
    weights,
    bounds,
    pre_indexing,
    post_indexing,
    indices_for_ratios,
    ratios_root_height,
    clock,
    props,
):
    internal_heights = transform_ratios(ratios_root_height, bounds, indices_for_ratios)
    branch_lengths = heights_to_branch_lengths(internal_heights, bounds, pre_indexing)

    bls = branch_lengths * clock
    mats = subst_model.p_t(np.expand_dims(bls, -1))
    frequencies = np.broadcast_to(subst_model.frequencies, bls.shape[:-1] + (4,))
    log_p = calculate_treelikelihood(
        partials,
        weights,
        post_indexing,
        mats,
        np.expand_dims(frequencies, axis=-2),
        props,
    )
    return log_p, internal_heights
