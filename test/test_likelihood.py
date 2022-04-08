import jax.numpy as np
import pytest
from jax import grad, vjp

import phylojax.treelikelihood as treelikelihood
from phylojax.io import read_tree_and_alignment
from phylojax.sitepattern import get_dna_leaves_partials_compressed
from phylojax.tree import postorder_indices, preorder_indices


def extract_branch_lengths(tree):
    nodes = [None] * (len(tree.taxon_namespace) * 2 - 1)
    for node in tree.postorder_node_iter():
        nodes[node.index] = node.edge_length
    nodes.pop()
    return np.array([[float(x) for x in nodes]])


def test_calculate_unrooted_efficient(hello_tree_file, hello_fasta_file, jc69_model):
    tree, dna = read_tree_and_alignment(hello_tree_file, hello_fasta_file, False, False)
    indices = postorder_indices(tree)
    pre_indices = preorder_indices(tree)
    branch_lengths = extract_branch_lengths(tree)
    bls = np.expand_dims(branch_lengths, axis=-1)
    partials, weights = get_dna_leaves_partials_compressed(dna)
    tip_partials = np.array(partials[: len(tree.taxon_namespace)])
    frequencies = np.broadcast_to(
        jc69_model.frequencies, branch_lengths.shape[:-1] + (4,)
    )
    frequencies = np.expand_dims(frequencies, axis=-2)
    mats = jc69_model.p_t(bls)
    mats = np.expand_dims(mats, -3)
    props = np.array([[[1.0]]])
    partials = treelikelihood.calculate_partials(
        tip_partials,
        indices,
        mats,
        props,
    )

    # calculate log likelihood at every node using upper partials
    uppers = treelikelihood.calculate_upper_partials(partials, pre_indices, mats)
    for i in range(uppers.shape[0]):
        log_p2 = treelikelihood.calculate_treelikelihood_upper(
            partials[i],
            uppers[i],
            weights,
            mats[:, i, ...],
            frequencies,
            props,
        )
        assert -84.852358 == pytest.approx(float(log_p2), 0.0001)

    # analytical derivatives
    expected_gradient = (21.0223, -5.34462, -17.7298, -17.7298)
    dmats = jc69_model.dp_dt(bls)
    dmats = np.expand_dims(dmats, -3)
    likelihoods = frequencies @ np.sum(partials[-1], -3)
    gradient = treelikelihood.calculate_treelikelihood_gradient(
        likelihoods, partials, uppers, weights, dmats, frequencies, props
    )
    assert np.allclose(gradient.squeeze(), expected_gradient, rtol=0.0001)

    # analytical derivatives using custom_jvp
    log_p3 = treelikelihood.calculate_treelikelihood_custom(
        branch_lengths, tip_partials, weights, (indices, pre_indices), jc69_model, props
    )
    assert -84.852358 == pytest.approx(float(log_p3), 0.0001)
    g = grad(treelikelihood.calculate_treelikelihood_custom)
    gradient2 = g(
        branch_lengths,
        tip_partials,
        weights,
        (np.array(indices, dtype=np.int32), np.array(pre_indices, dtype=np.int32)),
        jc69_model,
        props,
    )
    assert np.allclose(gradient2.squeeze(), expected_gradient, rtol=0.0001)


def test_calculate_unrooted(hello_tree_file, hello_fasta_file, jc69_model):
    tree, dna = read_tree_and_alignment(hello_tree_file, hello_fasta_file, False, False)
    indices = indices = postorder_indices(tree)
    branch_lengths = extract_branch_lengths(tree)
    bls = np.expand_dims(branch_lengths, axis=-1)
    partials, weights = get_dna_leaves_partials_compressed(dna)
    tip_partials = np.array(partials[: len(tree.taxon_namespace)])
    frequencies = np.broadcast_to(
        jc69_model.frequencies, branch_lengths.shape[:-1] + (4,)
    )

    def fn(bls):
        mats = jc69_model.p_t(bls)
        return treelikelihood.calculate_treelikelihood(
            tip_partials,
            weights,
            indices,
            np.expand_dims(mats, -3),
            np.expand_dims(frequencies, axis=-2),
            np.array([[[1.0]]]),
        )

    y, vjp_fn = vjp(fn, bls)
    gradient = vjp_fn(np.ones(y.shape))[0]
    expected_gradient = (21.0223, -5.34462, -17.7298, -17.7298)
    assert all(
        [
            a == pytest.approx(b, 0.0001)
            for a, b in zip(expected_gradient, np.squeeze(gradient))
        ]
    )

    mats = jc69_model.p_t(bls)
    log_p = treelikelihood.calculate_treelikelihood(
        tip_partials,
        weights,
        indices,
        np.expand_dims(mats, -3),
        np.expand_dims(frequencies, axis=-2),
        np.array([[[1.0]]]),
    )
    assert -84.852358 == pytest.approx(float(log_p), 0.0001)


def test_calculate_likelihood_rooted(flu_a_tree_file, flu_a_fasta_file, jc69_model):
    tree, dna = read_tree_and_alignment(flu_a_tree_file, flu_a_fasta_file, True, True)
    indices = postorder_indices(tree)
    partials, weights = get_dna_leaves_partials_compressed(dna)
    tip_partials = np.array(partials[: len(tree.taxon_namespace)])
    branch_lengths = extract_branch_lengths(tree) * 0.001
    mats = jc69_model.p_t(np.expand_dims(branch_lengths, axis=-1))
    frequencies = np.broadcast_to(
        jc69_model.frequencies, branch_lengths.shape[:-1] + (4,)
    )

    log_p = treelikelihood.calculate_treelikelihood(
        tip_partials,
        weights,
        indices,
        np.expand_dims(mats, -3),
        np.expand_dims(frequencies, axis=-2),
        np.array([[[1.0]]]),
    )
    assert -4777.616349 == pytest.approx(float(log_p), 0.0001)
