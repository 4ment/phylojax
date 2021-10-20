import jax.numpy as np
import pytest
from jax import vjp

import phylojax.treelikelihood as treelikelihood
from phylojax.io import read_tree_and_alignment
from phylojax.sitepattern import get_dna_leaves_partials_compressed


def test_calculate_unrooted(hello_tree_file, hello_fasta_file, jc69_model):
    tree, dna = read_tree_and_alignment(hello_tree_file, hello_fasta_file, False, False)
    nodes = [None] * (len(tree.taxon_namespace) * 2 - 1)
    for node in tree.postorder_node_iter():
        nodes[node.index] = node.edge_length
    nodes.pop()
    indices = []
    for node in tree.postorder_node_iter():
        if not node.is_leaf():
            children = node.child_nodes()
            indices.append((node.index, children[0].index, children[1].index))

    branch_lengths = np.array([[float(x) for x in nodes]])
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
            mats,
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
        mats,
        np.expand_dims(frequencies, axis=-2),
        np.array([[[1.0]]]),
    )
    assert -84.852358 == pytest.approx(float(log_p), 0.0001)


def test_calculate_likelihood_rooted(flu_a_tree_file, flu_a_fasta_file, jc69_model):
    tree, dna = read_tree_and_alignment(flu_a_tree_file, flu_a_fasta_file, True, True)
    nodes = [None] * (len(tree.taxon_namespace) * 2 - 1)
    for node in tree.postorder_node_iter():
        nodes[node.index] = node.edge_length
    nodes.pop()
    indices = []
    for node in tree.postorder_node_iter():
        if not node.is_leaf():
            children = node.child_nodes()
            indices.append((node.index, children[0].index, children[1].index))
    partials, weights = get_dna_leaves_partials_compressed(dna)
    tip_partials = np.array(partials[: len(tree.taxon_namespace)])

    branch_lengths = np.array([[float(x) for x in nodes]]) * 0.001
    mats = jc69_model.p_t(np.expand_dims(branch_lengths, axis=-1))
    frequencies = np.broadcast_to(
        jc69_model.frequencies, branch_lengths.shape[:-1] + (4,)
    )

    log_p = treelikelihood.calculate_treelikelihood(
        tip_partials,
        weights,
        indices,
        mats,
        np.expand_dims(frequencies, axis=-2),
        np.array([[[1.0]]]),
    )
    assert -4777.616349 == pytest.approx(float(log_p), 0.0001)
