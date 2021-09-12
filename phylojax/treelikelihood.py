import jax.numpy as np

from .tree import heights_to_branch_lengths, transform_ratios


def calculate_treelikelihood(partials, weights, post_indexing, mats, freqs, props):
    for node, left, right in post_indexing:
        partials[node] = (mats[..., left, :, :, :] @ partials[left]) * (
            mats[..., right, :, :, :] @ partials[right]
        )
    return np.sum(
        np.log(freqs @ np.sum(props * partials[post_indexing[-1][0]], -3)) * weights,
        axis=-1,
    )


def jax_likelihood(
    subst_model,
    partials,
    weights,
    bounds,
    pre_indexing,
    post_indexing,
    root_height,
    ratios,
    clock,
    props,
):
    node_heights = transform_ratios(root_height, ratios, bounds, pre_indexing)
    branch_lengths = heights_to_branch_lengths(node_heights, bounds, pre_indexing)

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
    return log_p, node_heights
