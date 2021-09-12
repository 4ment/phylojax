from functools import partial

import jax.numpy as np
import phylojax.treelikelihood as treelikelihood
import pytest
from jax import grad
from phylojax.io import read_tree_and_alignment
from phylojax.sitepattern import get_dna_leaves_partials_compressed
from phylojax.tree import distance_to_ratios


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
    branch_lengths = np.array([float(x) for x in nodes])
    bls = np.expand_dims(branch_lengths, axis=1)
    partials, weights = get_dna_leaves_partials_compressed(dna)

    def grad_fn(bls):
        mats = jc69_model.p_t(bls)
        return treelikelihood.calculate_treelikelihood(
            partials, weights, indices, mats, np.array([0.25] * 4), np.array([[[1.0]]])
        )

    g = grad(grad_fn)
    gradient = g(bls)
    expected_gradient = (21.0223, -5.34462, -17.7298, -17.7298)
    assert all(
        [
            a == pytest.approx(b, 0.0001)
            for a, b in zip(expected_gradient, np.squeeze(gradient))
        ]
    )

    mats = jc69_model.p_t(bls)
    log_p = treelikelihood.calculate_treelikelihood(
        partials, weights, indices, mats, np.array([0.25] * 4), np.array([[[1.0]]])
    )
    assert -84.852358 == pytest.approx(float(log_p), 0.0001)


def test_calculate_likelihood_rooted(flu_a_tree_file, flu_a_fasta_file, jc69_model):
    tree, dna = read_tree_and_alignment(flu_a_tree_file, flu_a_fasta_file, True, True)
    nodes = [None] * (len(tree.taxon_namespace) * 2 - 1)
    for node in tree.postorder_node_iter():
        nodes[node.index] = node.edge_length
    nodes.pop()
    nodes = [edge_length * 0.001 for edge_length in nodes]
    indices = []
    for node in tree.postorder_node_iter():
        if not node.is_leaf():
            children = node.child_nodes()
            indices.append((node.index, children[0].index, children[1].index))
    branch_lengths = np.array([float(x) for x in nodes])
    bls = np.expand_dims(branch_lengths, axis=1)
    partials, weights = get_dna_leaves_partials_compressed(dna)

    mats = jc69_model.p_t(bls)
    log_p = treelikelihood.calculate_treelikelihood(
        partials, weights, indices, mats, np.array([0.25] * 4), np.array([[[1.0]]])
    )
    assert -4777.616349 == pytest.approx(float(log_p), 0.0001)


def test_calculate_rooted(flu_a_tree_file, flu_a_fasta_file, jc69_model):
    tree, dna = read_tree_and_alignment(flu_a_tree_file, flu_a_fasta_file, True, True)
    indices = []
    for node in tree.postorder_node_iter():
        if not node.is_leaf():
            children = node.child_nodes()
            indices.append((node.index, children[0].index, children[1].index))

    partials, weights = get_dna_leaves_partials_compressed(dna)
    ratios, root_height, bounds = distance_to_ratios(tree)

    # postorder for peeling
    post_indexing = []
    for node in tree.postorder_node_iter():
        if not node.is_leaf():
            children = node.child_nodes()
            post_indexing.append((node.index, children[0].index, children[1].index))

    # preoder indexing to go from ratios to heights
    pre_indexing = []
    for node in tree.preorder_node_iter():
        if node != tree.seed_node:
            pre_indexing.append((node.parent_node.index, node.index))

    jax_likelihood_fn = partial(
        treelikelihood.jax_likelihood,
        jc69_model,
        partials,
        weights,
        np.array(bounds),
        np.array(pre_indexing),
        post_indexing,
    )

    def test(a, b, c, d):
        log_p, _ = jax_likelihood_fn(a, b, c, d)
        return log_p[0]

    g = grad(test, (0, 1, 2))
    root_height_grad, ratios_grad, clock_gradient = g(
        np.array([root_height]),
        np.array(ratios),
        np.array([0.001]),
        np.array([[[1.0]]]),
    )
    assert 328018 == pytest.approx(float(clock_gradient), 0.0001)

    log_p, nh = jax_likelihood_fn(
        np.array([root_height]),
        np.array(ratios),
        np.array([0.001]),
        np.array([[[1.0]]]),
    )
    assert -4777.616349 == pytest.approx(float(log_p), 0.0001)
