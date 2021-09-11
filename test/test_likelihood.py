import pytest
import jax.numpy as np
from jax import grad
from functools import partial
from phylojax.sitepattern import get_dna_leaves_partials_compressed
from phylojax.io import read_tree_and_alignment
import phylojax.treelikelihood as treelikelihood
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
    print(weights)
    print(partials)
    print(jc69_model.p_t(bls))
    print('bls', bls)
    def grad_fn(bls):
        mats = jc69_model.p_t(bls)
        return treelikelihood.calculate_treelikelihood(partials, weights, indices, mats, np.array([0.25]*4))

    g = grad(grad_fn)
    gradient = g(bls)
    expected_gradient = (21.0223, -5.34462, -17.7298, -17.7298)
    assert all([a == pytest.approx(b, 0.0001) for a, b in zip(expected_gradient, np.squeeze(gradient))])

    mats = jc69_model.p_t(bls)
    log_p = treelikelihood.calculate_treelikelihood(partials, weights, indices, mats, np.array([0.25]*4))
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
    log_p = treelikelihood.calculate_treelikelihood(partials, weights, indices, mats, np.array([0.25]*4))
    assert -4777.616349 == pytest.approx(float(log_p), 0.0001)


def test_calculate_rooted(flu_a_tree_file, flu_a_fasta_file, jc69_model):
    tree, dna = read_tree_and_alignment(flu_a_tree_file, flu_a_fasta_file, True, True)
    taxa_count = len(tree.taxon_namespace)
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

    jax_likelihood_fn = partial(treelikelihood.jax_likelihood, jc69_model, partials, weights, np.array(bounds),
                                np.array(pre_indexing),
                                post_indexing)

    def test(a,b,c):
        log_p, _ = jax_likelihood_fn(a, b, c)
        return log_p

    expected_gradient = (
        -0.593654, 6.44129, 11.2029, 5.17392, -0.904631, 2.7314, 3.15713, 7.08291, 10.3054, 13.9882, 20.7093, 48.898,
        99.1649, 130.206, 17.314, 21.0333, -1.33633, 12.2598, 22.8873, 27.1766, 47.4874, 3.63728, 12.9552, 15.316,
        83.2546,
        -3.807, 105.385, 4.87402, 22.7545, 6.03653, 25.6515, 29.5352, 29.5988, 1.81725, 10.5987, 76.2592, 56.4814,
        10.6798,
        6.58718, 3.33056, -4.62225, 33.4173, 63.4158, 188.81, 23.5409, 17.4211, 1.22257, 22.372, 34.2395, 3.48611,
        4.09887,
        13.201, 19.7269, 96.8087, 4.24003, 7.41458, 48.8717, 3.48852, 82.9691, 9.00933, 8.03247, 3.98102, 6.54365,
        53.7024,
        37.836, 2.84083, 7.51719)

    g = grad(test, (0, 1, 2))
    root_height_grad, ratios_grad, clock_gradient = g(np.array([root_height]), np.array(ratios), np.array([0.001]))

    # with grad log det jacobian
    # assert 19.9369 == pytest.approx(float(root_height_grad), 0.0001)
    # assert all([a == pytest.approx(b, 0.001) for a, b in zip(expected_gradient, np.squeeze(ratios_grad))])
    assert 328018 == pytest.approx(float(clock_gradient), 0.0001)

    log_p, nh = jax_likelihood_fn(np.array([root_height]), np.array(ratios), np.array([0.001]))
    # with grad log det jacobian
    assert -4786.87 == pytest.approx(float(log_p), 0.0001)
    # without grad log det jacobian
    # assert -4777.616349 == pytest.approx(float(log_p), 0.0001)