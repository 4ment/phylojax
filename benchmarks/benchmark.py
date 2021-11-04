#!/usr/bin/env python

import argparse
from timeit import default_timer as timer

import jax
import jax.numpy as np
from jax import grad, jit, vjp
from jax.ops import index, index_update

import phylojax.treelikelihood as treelikelihood
from phylojax.io import read_tree, read_tree_and_alignment
from phylojax.sitepattern import get_dna_leaves_partials_compressed
from phylojax.substitution import JC69
from phylojax.tree import distance_to_ratios, log_abs_det_jacobian, transform_ratios

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def calculate_treelikelihoodv2(partials, weights, post_indexing, mats, freqs, props):
    for node, left, right in post_indexing:
        partials[node] = (mats[..., left, :, :, :] @ partials[left]) * (
            mats[..., right, :, :, :] @ partials[right]
        )
    return np.sum(
        np.log(freqs @ np.sum(props * partials[post_indexing[-1][0]], -3)) * weights,
        axis=-1,
    )


def calculate_treelikelihoodv3(
    tip_partials, weights, post_indexing, mats, freqs, props
):
    partials = np.concatenate(
        (
            np.expand_dims(np.expand_dims(tip_partials, 1), -3).repeat(
                mats.shape[-5], 1
            ),
            np.empty(
                (tip_partials.shape[0] - 1,)
                + mats.shape[:-4]
                + (mats.shape[-3],)
                + tip_partials.shape[1:]
            ),
        ),
        axis=0,
    )

    for node, left, right in post_indexing:
        partials = index_update(
            partials,
            index[node],
            (mats[..., left, :, :, :] @ partials[left])
            * (mats[..., right, :, :, :] @ partials[right]),
        )
    return np.sum(
        np.log(freqs @ np.sum(props * partials[post_indexing[-1][0]], -3)) * weights,
        axis=-1,
    )


def log_prob_squashed(theta, node_heights, counts, taxa_count):
    indices = np.argsort(node_heights)
    heights_sorted = np.take_along_axis(node_heights, indices, -1)
    counts_sorted = np.take_along_axis(counts, indices, -1)
    lineage_count = counts_sorted.cumsum(-1)[..., :-1]
    durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
    lchoose2 = lineage_count * (lineage_count - 1) / 2.0
    return np.sum(-lchoose2 * durations / theta, axis=-1, keepdims=True) - (
        taxa_count - 1
    ) * np.log(theta)


def log_prob(node_heights, theta):
    taxa_shape = node_heights.shape[:-1] + (int((node_heights.shape[-1] + 1) / 2),)
    node_mask = np.concatenate(
        (
            np.full(taxa_shape, 0),
            np.full(
                taxa_shape[:-1] + (taxa_shape[-1] - 1,),
                1,
            ),
        ),
        axis=-1,
    )
    indices = np.argsort(node_heights)
    heights_sorted = np.take_along_axis(node_heights, indices, -1)
    node_mask_sorted = np.take_along_axis(node_mask, indices, -1)
    lineage_count = np.where(
        node_mask_sorted == 1,
        np.full_like(theta, -1),
        np.full_like(theta, 1),
    ).cumsum(-1)[..., :-1]
    durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
    lchoose2 = lineage_count * (lineage_count - 1) / 2.0
    return np.sum(-lchoose2 * durations / theta, axis=-1, keepdims=True) - (
        taxa_shape[-1] - 1
    ) * np.log(theta)


def fluA_unrooted(args):
    replicates = args.replicates
    tree, dna = read_tree_and_alignment(args.tree, args.input, True, True)
    nodes = [None] * (len(tree.taxon_namespace) * 2 - 1)
    for node in tree.postorder_node_iter():
        nodes[node.index] = node.edge_length
    nodes.pop()
    indices = []
    for node in tree.postorder_node_iter():
        if not node.is_leaf():
            children = node.child_nodes()
            indices.append((node.index, children[0].index, children[1].index))
    indices = tuple(indices)
    branch_lengths = np.array([[float(x) for x in nodes]]) * 0.001
    branch_lengths = jax.lax.clamp(1.0e-6, branch_lengths, np.inf)
    bls = np.expand_dims(branch_lengths, axis=-1)
    partials, weights = get_dna_leaves_partials_compressed(dna)
    jc69_model = JC69()

    proportions = np.array([[[1.0]]])
    tip_partials = np.array(partials[: len(tree.taxon_namespace)])

    def fn(bls):
        mats = jc69_model.p_t(bls)
        return treelikelihood.calculate_treelikelihood(
            tip_partials, weights, indices, mats, jc69_model.frequencies, proportions
        )[0]

    if args.all:
        print("  JIT off")

        t, log_p, grad_log_p = test(fn, grad(fn), bls, replicates, args.separate)

        if args.output:
            args.output.write(
                f"treelikelihood,evaluation,off,{t[0]},{log_p.squeeze().tolist()}\n"
            )
            args.output.write(f"treelikelihood,gradient,off,{t[1]},\n")

    print("  JIT on jit(grad(fn))")
    calculate_treelikelihood_jit = jit(fn)
    t, log_p, grad_log_p = test(
        calculate_treelikelihood_jit, jit(grad(fn)), bls, replicates, args.separate
    )

    if args.output:
        args.output.write(
            f"treelikelihood,evaluation,on,{t[0]},{log_p.squeeze().tolist()}\n"
        )
        args.output.write(f"treelikelihood,gradient,on,{t[1]},\n")

    if args.all:
        print("  JIT on grad(jit(fn))")
        test(
            calculate_treelikelihood_jit,
            grad(calculate_treelikelihood_jit),
            bls,
            replicates,
            args.separate,
        )

        print("  v2 JIT off")

        def fnv2(bls):
            mats = jc69_model.p_t(bls)
            return calculate_treelikelihoodv2(
                partials, weights, indices, mats, jc69_model.frequencies, proportions
            )[0]

        test(fnv2, grad(fnv2), bls, replicates, args.separate)

        print("  v3 JIT off")

        def fnv3(bls):
            mats = jc69_model.p_t(bls)
            return calculate_treelikelihoodv3(
                tip_partials,
                weights,
                indices,
                mats,
                jc69_model.frequencies,
                proportions,
            )[0]

        test(fnv3, grad(fnv3), bls, replicates, args.separate)

        print("  v3 JIT on grad(jit(fn))")
        fnv3_jit = jit(fnv3)
        test(fnv3_jit, grad(fnv3_jit), bls, replicates, args.separate)


def test(fn, g, bls, replicates, separate=False):
    times = []
    if separate:
        replicates = replicates - 1
        start = timer()
        log_p = fn(bls)
        end = timer()
        t0 = end - start
        print(f"  First evaluation: {t0}")

    start = timer()
    for _ in range(replicates):
        log_p = fn(bls)
    end = timer()
    t1 = end - start
    times.append(t1)
    print(f"  {replicates} evaluations: {t1} ({log_p.squeeze().tolist()})")

    if separate:
        start = timer()
        _ = g(bls)
        end = timer()
        t2 = end - start
        print(f"  First gradient evaluation: {t2} ({log_p}")

    start = timer()
    for _ in range(replicates):
        grad_log_p = g(bls)
    end = timer()
    t3 = end - start
    times.append(t3)
    print(f"  {replicates} gradient evaluations: {t3})")

    if separate:
        times.append(t0)
        times.append(t2)
    return times, log_p, grad_log_p


def ratio_transform_jacobian(args):
    replicates = args.replicates
    tree = read_tree(args.tree, True, True)
    taxa_count = len(tree.taxon_namespace)
    ratios, root_height, bounds = distance_to_ratios(tree)

    pre_indexing = []
    for node in tree.preorder_node_iter():
        if node != tree.seed_node:
            pre_indexing.append((node.parent_node.index, node.index))
    pre_indexing = np.array(pre_indexing)
    pre_indexing = pre_indexing[np.argsort(pre_indexing[:, 1])].transpose()

    # indices for jacobian term of heights
    indices_for_jac = pre_indexing[0, taxa_count:] - taxa_count

    # indices for ratios to heights
    indices_for_ratios = []
    for node in tree.preorder_node_iter():
        if node != tree.seed_node and node.index >= taxa_count:
            indices_for_ratios.append((node.parent_node.index, node.index))
    indices_for_ratios = np.array(indices_for_ratios)

    ratios_root_height = np.concatenate((ratios, root_height), axis=-1)

    def fn(ratios_root_height):
        internal_heights = transform_ratios(
            ratios_root_height, bounds, indices_for_ratios
        )
        return log_abs_det_jacobian(
            internal_heights, indices_for_jac, bounds[taxa_count:-1]
        )

    print("  JIT off")
    start = timer()
    for _ in range(replicates):
        log_det_jac = fn(ratios_root_height)

    end = timer()
    print(
        f"  {replicates} evaluations: {end - start} ({log_det_jac.squeeze().tolist()})"
    )

    if args.output:
        args.output.write(
            f"ratio_transform_jacobian,evaluation,off,{end - start},"
            f"{log_det_jac.squeeze().tolist()}\n"
        )

    start = timer()
    y, vjp_fn = vjp(fn, ratios_root_height)
    for _ in range(replicates):
        log_det_jac_gradient = vjp_fn(np.ones(y.shape))[0]
    end = timer()
    print(f"  {replicates} gradient evaluations: {end - start}")

    if args.output:
        args.output.write(f"ratio_transform_jacobian,gradient,off,{end - start},\n")

    if args.debug:
        print(log_det_jac_gradient)

    print("  JIT on")
    fn_jit = jit(fn)
    start = timer()
    for _ in range(replicates):
        log_det_jac = fn_jit(ratios_root_height)
    end = timer()
    print(f"  {replicates} evaluations: {end - start} ({log_det_jac.tolist()})")

    if args.output:
        args.output.write(
            f"ratio_transform_jacobian,evaluation,on,{end - start},"
            f"{log_det_jac.squeeze().tolist()}\n"
        )

    start = timer()
    y, vjp_fn = vjp(fn, ratios_root_height)
    fn_jit_grad = jit(vjp_fn)
    for _ in range(replicates):
        log_det_jac_gradient = fn_jit_grad(np.ones(y.shape))[0]

    end = timer()
    print(f"  {replicates} gradient evaluations: {end - start}")

    if args.output:
        args.output.write(f"ratio_transform_jacobian,gradient,on,{end - start},\n")


def ratio_transform(args, separate=False):
    replicates = args.replicates
    if separate:
        replicates = replicates - 1

    tree = read_tree(args.tree, True, True)
    taxa_count = len(tree.taxon_namespace)
    bounds = [None] * (2 * taxa_count - 1)
    for node in tree.postorder_node_iter():
        if node.is_leaf():
            bounds[node.index] = node.date
        else:
            bounds[node.index] = max([bounds[x.index] for x in node.child_node_iter()])
    bounds = np.array(bounds)

    indices_for_ratios = []
    for node in tree.preorder_node_iter():
        if node != tree.seed_node and node.index >= taxa_count:
            indices_for_ratios.append((node.parent_node.index, node.index))
    indices_for_ratios = np.array(indices_for_ratios)

    ratios = np.array([0.5] * 67)
    root = np.array([20.0])

    ratios_root_height = np.concatenate((ratios, root), axis=-1)

    print("  JIT off")

    if separate:
        start = timer()
        transform_ratios(ratios_root_height, bounds, indices_for_ratios)
        end = timer()
        print(f"  First evaluation: {end - start}")

    start = timer()
    for _ in range(replicates):
        transform_ratios(ratios_root_height, bounds, indices_for_ratios)
    end = timer()
    print(f"  {replicates} evaluations: {end - start}")

    print("  JIT on")
    transform_ratios_jit = jit(transform_ratios)

    if separate:
        start = timer()
        transform_ratios_jit(ratios_root_height, bounds, indices_for_ratios)
        end = timer()
        print(f"  First evaluation: {end - start}")

    start = timer()
    for _ in range(replicates):
        transform_ratios_jit = jit(transform_ratios)
        transform_ratios_jit(ratios_root_height, bounds, indices_for_ratios)
    end = timer()
    print(f"  {replicates} evaluations: {end - start}")


def constant_coalescent(args, separate=False):
    replicates = args.replicates
    tree = read_tree(args.tree, True, True)
    taxa_count = len(tree.taxon_namespace)
    ratios, root_height, bounds = distance_to_ratios(tree)

    indices_for_ratios = []
    for node in tree.preorder_node_iter():
        if node != tree.seed_node and node.index >= taxa_count:
            indices_for_ratios.append((node.parent_node.index, node.index))
    indices_for_ratios = np.array(indices_for_ratios)

    ratios_root_height = np.concatenate((ratios, root_height), axis=-1)
    internal_heights = transform_ratios(ratios_root_height, bounds, indices_for_ratios)
    node_heights = np.concatenate((bounds[:taxa_count], internal_heights))

    theta = np.array([4.0])

    print("  JIT off")
    start = timer()
    for _ in range(replicates):
        log_p = log_prob(node_heights, theta)
    end = timer()
    t1 = end - start
    print(f"  {replicates} evaluations: {t1} ({log_p.squeeze().tolist()})")

    if args.output:
        args.output.write(
            f"coalescent,evaluation,off,{end - start},{log_p.squeeze().tolist()}\n"
        )

    start = timer()
    y, vjp_fn = vjp(log_prob, node_heights, theta)
    for _ in range(replicates):
        _ = vjp_fn(np.ones(y.shape))[0]

    end = timer()
    print(f"  {replicates} gradient evaluations: {end - start}")

    if args.output:
        args.output.write(f"coalescent,gradient,off,{end - start},\n")

    print("  JIT on")

    log_prob_jit = jit(log_prob)

    if separate:
        start = timer()
        log_p = log_prob_jit(node_heights, theta)
        end = timer()
        print(f"  First evaluation: {end-start} ({log_p})")

    start = timer()
    for _ in range(replicates):
        log_prob_jit(node_heights, theta)
    end = timer()
    print(f"  {replicates} evaluations: {end - start} ({log_p.squeeze().tolist()})")

    if args.output:
        args.output.write(
            f"coalescent,evaluation,on,{end - start},{log_p.squeeze().tolist()}\n"
        )

    fn_jit_grad = jit(vjp_fn)
    start = timer()
    for _ in range(replicates):
        _ = fn_jit_grad(np.ones(y.shape))[0]
    end = timer()
    print(f"  {replicates} gradient evaluations: {end - start}")

    if args.output:
        args.output.write(f"coalescent,gradient,on,{end - start},\n")

    if args.all:
        x, counts = np.unique(bounds[:taxa_count], return_counts=True)
        counts = np.concatenate((counts, np.full((68,), -1)), 0)
        start = timer()
        for _ in range(replicates):
            log_prob_squashed(
                theta, np.concatenate((x, internal_heights)), counts, taxa_count
            )
        end = timer()
        t1 = end - start
        print(f"  {replicates} evaluations: {t1}")

        print("\n  version 2")
        log_prob_squashed_script = jit(log_prob_squashed)
        start = timer()
        log_p = log_prob_squashed_script(
            theta, np.concatenate((x, internal_heights)), counts, taxa_count
        )
        end = timer()
        print(f"  First evaluation: {end - start} ({log_p})")

        start = timer()
        log_prob_squashed_script = jit(log_prob_squashed)
        for _ in range(replicates):
            log_prob_squashed_script(
                theta, np.concatenate((x, internal_heights)), counts, taxa_count
            )
        end = timer()
        print(f"  {replicates} evaluations: {end - start}")


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="""alignment file""")
parser.add_argument("-t", "--tree", required=True, help="""tree file""")
parser.add_argument(
    "-r",
    "--replicates",
    required=True,
    type=int,
    help="""Number of replicates""",
)
parser.add_argument(
    "-o",
    "--output",
    type=argparse.FileType("w"),
    default=None,
    help="""csv output file""",
)
parser.add_argument(
    "-s",
    "--scaler",
    type=float,
    default=1.0,
    help="""scale branch lengths""",
)
parser.add_argument(
    "--debug", required=False, action="store_true", help="""Debug mode"""
)
parser.add_argument("--all", required=False, action="store_true", help="""Run all""")
parser.add_argument(
    "--separate", required=False, action="store_true", help="""Separate first call"""
)

args = parser.parse_args()

if args.output:
    args.output.write("function,mode,JIT,time,logprob\n")

print("Tree likelihood unrooted:")
fluA_unrooted(args)

print("Height transform log det Jacobian:")
ratio_transform_jacobian(args)

if args.all:
    print("Node height transform:")
    ratio_transform(args)

print("Constant coalescent:")
constant_coalescent(args)

if args.output:
    args.output.close()
