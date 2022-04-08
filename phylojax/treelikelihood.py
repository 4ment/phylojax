from functools import partial

import jax
import jax.numpy as np
from jax.ops import index, index_update

from .tree import heights_to_branch_lengths, transform_ratios


def calculate_partials(tip_partials, indices, mats, props):
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

    def fn(i, partials):
        node, left, right = np.array(indices)[i]
        return index_update(
            partials,
            index[node],
            (mats[..., left, :, :, :] @ partials[left])
            * (mats[..., right, :, :, :] @ partials[right]),
        )

    return jax.lax.fori_loop(0, len(indices), fn, partials)


def calculate_upper_partials(partials, indices, mats):
    uppers = np.empty((partials.shape[0] - 1,) + partials.shape[1:])
    node, sibling, _ = indices[0]
    uppers = index_update(
        uppers,
        index[node],
        mats[..., sibling, :, :, :] @ partials[sibling],
    )
    node, sibling, _ = indices[1]
    uppers = index_update(
        uppers,
        index[node],
        mats[..., sibling, :, :, :] @ partials[sibling],
    )

    def fn(i, uppers):
        node, sibling, parent = np.array(indices)[i]
        return index_update(
            uppers,
            index[node],
            (mats[..., sibling, :, :, :] @ partials[sibling])
            * (mats[..., parent, :, :, :] @ uppers[parent]),
        )

    return jax.lax.fori_loop(2, len(indices), fn, uppers)


def calculate_treelikelihood_upper(partials, uppers, weights, mat, freqs, props):
    return np.sum(
        np.log(freqs @ np.sum(props * (mat @ partials) * uppers, -3)) * weights,
        axis=-1,
    )


def calculate_treelikelihood_gradient(
    likelihoods, partials, uppers, weights, dmats, freqs, props
):
    return np.sum(
        (
            freqs
            @ np.sum(props * (np.swapaxes(dmats, 0, 1) @ partials[:-1]) * uppers, -3)
        )
        / likelihoods
        * weights,
        axis=-1,
    )


@partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3, 4, 5))
def calculate_treelikelihood_custom(
    branch_lengths, tip_partials, weights, indices, subst_model, props
):
    frequencies = np.expand_dims(subst_model.frequencies, axis=-2)
    mats = subst_model.p_t(np.expand_dims(branch_lengths, axis=-1))
    mats = np.expand_dims(mats, -3)
    partials = calculate_partials(tip_partials, indices[0], mats, props)
    return np.sum(
        np.log(frequencies @ np.sum(props * partials[indices[0][-1][0]], -3)) * weights,
        axis=-1,
    )


@calculate_treelikelihood_custom.defjvp
def calculate_treelikelihood_custom_jvp(
    tip_partials, weights, indices, subst_model, props, primals, tangents
):
    (branch_lengths,) = primals
    (branch_lengths_dot,) = tangents
    branch_lengths = np.expand_dims(branch_lengths, axis=-1)
    mats = subst_model.p_t(branch_lengths)
    mats = np.expand_dims(mats, -3)
    partials = calculate_partials(tip_partials, indices[0], mats, props)
    frequencies = np.expand_dims(subst_model.frequencies, axis=-2)
    log_p = np.sum(
        np.log(frequencies @ np.sum(props * partials[indices[0][-1][0]], -3)) * weights,
        axis=-1,
    )
    likelihoods = frequencies @ np.sum(partials[-1], -3)
    dmats = subst_model.dp_dt(branch_lengths)
    dmats = np.expand_dims(dmats, -3)
    uppers = calculate_upper_partials(partials, indices[1], mats)
    gradient = calculate_treelikelihood_gradient(
        likelihoods, partials, uppers, weights, dmats, frequencies, props
    )
    return log_p.squeeze(), np.dot(gradient.squeeze(), branch_lengths_dot.squeeze())


def calculate_treelikelihood(tip_partials, weights, indices, mats, freqs, props):
    partials = calculate_partials(tip_partials, indices, mats, props)
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
