import jax.numpy as np
from .tree import transform_ratios, heights_to_branch_lengths
from jax.api import vmap, jit, partial
import jax
import jax.ops


# @jax.partial(jit, static_argnums=(1, 2, 4))
def calculate_treelikelihood(partials, weights, post_indexing, mats, freqs):
    for node, left, right in post_indexing:
        # partials = jax.ops.index_update(partials, jax.ops.index[node, :, :], np.dot(mats[left], partials[left]) * np.dot(mats[right], partials[right]))
        partials[node] = np.dot(mats[left], partials[left]) * np.dot(mats[right], partials[right])
        # partials[node] = vmap(np.dot, in_axes=(0,None))(mats[left], partials[left]) * vmap(np.dot, in_axes=(0,None))(mats[right], partials[right])
    return np.sum(np.log(np.dot(freqs, partials[-1])) * weights)

    # for node, left, right in post_indexing:
    #     partials[node] = np.matmul(mats[left], partials[left]) * np.matmul(mats[right], partials[right])
    # return np.sum(np.log(np.sum(np.squeeze(partials[-1]) * freqs, 1)) * weights)


#@jax.partial(jit, static_argnums=(0,2,3,4,5))
#@jit
def jax_likelihood(subst_model, partials, weights, bounds, pre_indexing, post_indexing, root_height, ratios, clock):
    # taxa_count = ratios.shape[0] + 2
    node_heights = transform_ratios(root_height, ratios, bounds, pre_indexing)
    branch_lengths = heights_to_branch_lengths(node_heights, bounds, pre_indexing)
    # heights = np.split(node_heights, node_heights.shape[0])
    # branch_lengths = [None]*(len(partials)-1)
    # for idx_parent, idx in pre_indexing:
    #     if idx < taxa_count:
    #         branch_lengths[idx] = heights[idx_parent - taxa_count] - bounds[idx]
    #     else:
    #         branch_lengths[idx] = heights[idx_parent - taxa_count] - heights[idx - taxa_count]

    # bls = np.expand_dims(np.concatenate(branch_lengths), axis=1) * clock
    bls = branch_lengths * clock
    mats = subst_model.p_t(bls)
    # print("log_p")
    log_p = calculate_treelikelihood(partials, weights, post_indexing, mats, subst_model.frequencies)
    # print(log_p)
    return log_p, node_heights