#!/usr/bin/env python

import argparse
from timeit import default_timer as timer

import jax
import jax.numpy as np
from jax import grad, jit, vjp
from jax.ops import index, index_update

import phylojax.treelikelihood as treelikelihood
from phylojax.coalescent import ConstantCoalescent
from phylojax.io import read_tree, read_tree_and_alignment
from phylojax.sitepattern import get_dna_leaves_partials_compressed
from phylojax.substitution import GTR, JC69
from phylojax.tree import (
    distance_to_ratios,
    log_abs_det_jacobian,
    preorder_indices,
    transform_ratios,
)

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


def fluA_unrooted(args, subst_model_type):
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
    branch_lengths = np.array([[float(x) for x in nodes]]) * args.scaler
    branch_lengths = jax.lax.clamp(1.0e-6, branch_lengths, np.inf)
    bls = np.expand_dims(branch_lengths, axis=-1)
    partials, weights = get_dna_leaves_partials_compressed(dna)

    proportions = np.array([[[1.0]]])
    tip_partials = np.array(partials[: len(tree.taxon_namespace)])

    def fn_JC69(bls):
        subst_model = JC69()
        mats = subst_model.p_t(bls)
        return treelikelihood.calculate_treelikelihood(
            tip_partials,
            weights,
            indices,
            np.expand_dims(mats, -3),
            subst_model.frequencies,
            proportions,
        )[0]

    def fn_GTR(bls, rates, frequencies):
        subst_model = GTR(rates, frequencies)
        mats = subst_model.p_t(bls)
        return treelikelihood.calculate_treelikelihood(
            tip_partials,
            weights,
            indices,
            np.expand_dims(mats, -3),
            subst_model.frequencies,
            proportions,
        )[0]

    if subst_model_type == 'JC69':
        fn = fn_JC69
        params = (bls,)
        argnums = 0
    else:
        fn = fn_GTR
        rates = np.repeat(1 / 6, 6)
        frequencies = np.repeat(1 / 4, 4)
        params = (bls, rates, frequencies)
        argnums = (0, 1, 2)

    print("  JIT off")

    t, log_p, grad_log_p = test(
        fn, grad(fn, argnums=argnums), params, replicates, args.separate
    )

    if args.output:
        args.output.write(
            f"treelikelihood{subst_model_type},evaluation,off,{t[0]},"
            f"{log_p.squeeze().tolist()}\n"
        )
        args.output.write(f"treelikelihood{subst_model_type},gradient,off,{t[1]},\n")

        if len(t) > 2:
            args.output.write(
                f"treelikelihood{subst_model_type},evaluation1,off,{t[2]},"
                f"{log_p.squeeze().tolist()}\n"
            )
            args.output.write(
                f"treelikelihood{subst_model_type},gradient1,off,{t[3]},\n"
            )

    print("  JIT on jit(grad(fn))")
    calculate_treelikelihood_jit = jit(fn)
    t, log_p, grad_log_p = test(
        calculate_treelikelihood_jit,
        jit(grad(fn, argnums)),
        params,
        replicates,
        args.separate,
    )

    if args.output:
        args.output.write(
            f"treelikelihood{subst_model_type},evaluation,on,{t[0]},"
            f"{log_p.squeeze().tolist()}\n"
        )
        args.output.write(f"treelikelihood{subst_model_type},gradient,on,{t[1]},\n")
        if len(t) > 2:
            args.output.write(
                f"treelikelihood{subst_model_type},evaluation1,on,{t[2]},"
                f"{log_p.squeeze().tolist()}\n"
            )
            args.output.write(
                f"treelikelihood{subst_model_type},gradient1,on,{t[3]},\n"
            )

    if subst_model_type == 'JC69':
        pre_indices = preorder_indices(tree)
        indices_tup = (
            np.array(indices, dtype=np.int32),
            np.array(pre_indices, dtype=np.int32),
        )

        def fn_custom(bls):
            return treelikelihood.calculate_treelikelihood_custom(
                bls,
                tip_partials,
                weights,
                indices_tup,
                JC69(),
                proportions,
            )

        print("  Analytic")

        calculate_treelikelihood_custom_jit = jit(fn_custom)
        t, log_p, grad_log_p = test(
            calculate_treelikelihood_custom_jit,
            jit(grad(fn_custom)),
            (branch_lengths,),
            replicates,
            args.separate,
        )

        if args.output:
            args.output.write(
                f"treelikelihoodAnalytic,evaluation,on,{t[0]},"
                f"{log_p.squeeze().tolist()}\n"
            )
            args.output.write(f"treelikelihoodAnalytic,gradient,on,{t[1]},\n")
            if len(t) > 2:
                args.output.write(
                    f"treelikelihoodAnalytic,evaluation1,on,{t[2]},"
                    f"{log_p.squeeze().tolist()}\n"
                )
                args.output.write(f"treelikelihoodAnalytic,gradient1,on,{t[3]},\n")

    if args.all and subst_model_type == 'JC69':
        jc69_model = JC69()

        print("  JIT on grad(jit(fn))")
        test(
            calculate_treelikelihood_jit,
            grad(calculate_treelikelihood_jit),
            params,
            replicates,
            args.separate,
        )

        print("  v2 JIT off")

        def fnv2(bls):
            mats = jc69_model.p_t(bls)
            return calculate_treelikelihoodv2(
                partials, weights, indices, mats, jc69_model.frequencies, proportions
            )

        test(fnv2, grad(fnv2), params, replicates, args.separate)

        print("  v2 JIT on")
        fnv2_jit = jit(fnv2)
        fnv2_jit_grad = jit(grad(fnv2))
        test(fnv2_jit, fnv2_jit_grad, params, replicates, args.separate)

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

        test(fnv3, grad(fnv3), params, replicates, args.separate)

        print("  v3 JIT on grad(jit(fn))")
        fnv3_jit = jit(fnv3)
        test(fnv3_jit, grad(fnv3_jit), params, replicates, args.separate)


def test(fn, g, params, replicates, separate):
    times = []
    if separate:
        replicates = replicates - 1
        start = timer()
        log_p = fn(*params)
        log_p.block_until_ready()
        end = timer()
        t0 = end - start
        print(f"  First evaluation: {t0}")

    start = timer()
    for _ in range(replicates):
        log_p = fn(*params)
        log_p.block_until_ready()
    end = timer()
    t1 = end - start
    times.append(t1)
    print(f"  {replicates} evaluations: {t1} ({log_p.squeeze().tolist()})")

    if len(params) == 1:
        if separate:
            start = timer()
            grad_log_p = g(*params)
            grad_log_p.block_until_ready()
            end = timer()
            t2 = end - start

        start = timer()
        for _ in range(replicates):
            grad_log_p = g(*params)
            grad_log_p.block_until_ready()
        end = timer()
    else:
        if separate:
            start = timer()
            grad_log_p = g(*params)
            jax.tree_map(lambda x: x.block_until_ready(), grad_log_p)
            end = timer()
            t2 = end - start

        start = timer()
        for _ in range(replicates):
            grad_log_p = g(*params)
            jax.tree_map(lambda x: x.block_until_ready(), grad_log_p)
        end = timer()
    t3 = end - start
    times.append(t3)
    if separate:
        print(f"  First gradient evaluation: {t2} ({log_p})")
    print(f"  {replicates} gradient evaluations: {t3}")

    if separate:
        times.append(t0)
        times.append(t2)
    return times, log_p, grad_log_p


def ratio_transform_jacobian(args):
    replicates = args.replicates - 1 if args.separate else args.replicates
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
    internal_heights = transform_ratios(ratios_root_height, bounds, indices_for_ratios)

    def fn(x):
        return log_abs_det_jacobian(x, indices_for_jac, bounds[taxa_count:-1])[0]

    print("  JIT off")

    if args.separate:
        start = timer()
        log_det_jac = fn(internal_heights)
        log_det_jac.block_until_ready()
        end = timer()
        print(f"  First evaluation: {end-start} ({log_det_jac})")
        if args.output:
            args.output.write(
                f"ratio_transform_jacobian,evaluation1,off,"
                f"{end - start},{log_det_jac.squeeze().tolist()}\n"
            )

    start = timer()
    for _ in range(replicates):
        log_det_jac = fn(internal_heights)
        log_det_jac.block_until_ready()
    end = timer()
    print(
        f"  {replicates} evaluations: {end - start} ({log_det_jac.squeeze().tolist()})"
    )

    if args.output:
        args.output.write(
            f"ratio_transform_jacobian,evaluation,off,{end - start},"
            f"{log_det_jac.squeeze().tolist()}\n"
        )

    fn_grad = grad(fn)
    if args.separate:
        start = timer()
        log_det_jac_gradient = fn_grad(internal_heights)
        log_det_jac_gradient.block_until_ready()
        end = timer()
        print(f"  First gradient evaluation: {end-start}")
        if args.output:
            args.output.write(
                f"ratio_transform_jacobian,gradient1,off,{end - start},\n"
            )

    start = timer()
    for _ in range(replicates):
        log_det_jac_gradient = fn_grad(internal_heights)
        log_det_jac_gradient.block_until_ready()
    end = timer()
    print(f"  {replicates} gradient evaluations: {end - start}")

    if args.output:
        args.output.write(f"ratio_transform_jacobian,gradient,off,{end - start},\n")

    if args.debug:
        print(log_det_jac_gradient)

    print("  JIT on")
    fn_jit = jit(fn)
    if args.separate:
        start = timer()
        log_det_jac = fn_jit(internal_heights)
        log_det_jac.block_until_ready()
        end = timer()
        print(f"  First evaluation: {end-start} ({log_det_jac})")
        if args.output:
            args.output.write(
                f"ratio_transform_jacobian,evaluation1,on,"
                f"{end - start},{log_det_jac.squeeze().tolist()}\n"
            )

    start = timer()
    for _ in range(replicates):
        log_det_jac = fn_jit(internal_heights)
        log_det_jac.block_until_ready()
    end = timer()
    print(f"  {replicates} evaluations: {end - start} ({log_det_jac.tolist()})")

    if args.output:
        args.output.write(
            f"ratio_transform_jacobian,evaluation,on,{end - start},"
            f"{log_det_jac.squeeze().tolist()}\n"
        )

    fn_grad_jit = jit(fn_grad)
    if args.separate:
        start = timer()
        log_det_jac_gradient = fn_grad_jit(internal_heights)
        log_det_jac_gradient.block_until_ready()
        end = timer()
        print(f"  First gradient evaluation: {end-start}")
        if args.output:
            args.output.write(f"ratio_transform_jacobian,gradient1,on,{end - start},\n")

    start = timer()
    for _ in range(replicates):
        log_det_jac_gradient = fn_grad_jit(internal_heights)
        log_det_jac_gradient.block_until_ready()
    end = timer()
    print(f"  {replicates} gradient evaluations: {end - start}")

    if args.output:
        args.output.write(f"ratio_transform_jacobian,gradient,on,{end - start},\n")


def ratio_transform(args):
    replicates = args.replicates - 1 if args.separate else args.replicates

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

    if args.separate:
        start = timer()
        transform_ratios(ratios_root_height, bounds, indices_for_ratios)
        end = timer()
        print(f"  First evaluation: {end - start}")
        if args.output:
            args.output.write(f"ratio_transform,evaluation1,off,{end - start},\n")

    start = timer()
    for _ in range(replicates):
        transform_ratios(ratios_root_height, bounds, indices_for_ratios)
    end = timer()
    print(f"  {replicates} evaluations: {end - start}")

    if args.output:
        args.output.write(f"ratio_transform,evaluation,off,{end - start},\n")

    def fn(x):
        return transform_ratios(x, bounds, indices_for_ratios)

    y, fn_vjp = vjp(fn, ratios_root_height)

    if args.separate:
        start = timer()
        log_det_jac_gradient = fn_vjp(np.ones(y.shape))[0]
        log_det_jac_gradient.block_until_ready()
        end = timer()
        print(f"  First gradient evaluation: {end-start}")
        if args.output:
            args.output.write(f"ratio_transform,gradient1,off,{end - start},\n")

    start = timer()
    for _ in range(replicates):
        log_det_jac_gradient = fn_vjp(np.ones(y.shape))[0]
        log_det_jac_gradient.block_until_ready()
    end = timer()
    print(f"  {replicates} gradient evaluations: {end - start}")

    if args.output:
        args.output.write(f"ratio_transform,gradient,off,{end - start},\n")

    print("  JIT on")
    fn_jit = jit(fn)

    if args.separate:
        start = timer()
        fn_jit(ratios_root_height)
        end = timer()
        print(f"  First evaluation: {end - start}")
        if args.output:
            args.output.write(f"ratio_transform,evaluation1,on,{end - start},\n")

    start = timer()
    for _ in range(replicates):
        fn_jit(ratios_root_height)
    end = timer()
    print(f"  {replicates} evaluations: {end - start}")

    if args.output:
        args.output.write(f"ratio_transform,evaluation,on,{end - start},\n")

    fn_vjp_jit = jit(fn_vjp)
    if args.separate:
        start = timer()
        log_det_jac_gradient = fn_vjp_jit(np.ones(y.shape))[0]
        log_det_jac_gradient.block_until_ready()
        end = timer()
        print(f"  First gradient evaluation: {end-start}")
        if args.output:
            args.output.write(f"ratio_transform,gradient1,on,{end - start},\n")

    start = timer()
    for _ in range(replicates):
        log_det_jac_gradient = fn_vjp_jit(np.ones(y.shape))[0]
        log_det_jac_gradient.block_until_ready()
    end = timer()
    print(f"  {replicates} gradient evaluations: {end - start}")

    if args.output:
        args.output.write(f"ratio_transform,gradient,on,{end - start},\n")


def constant_coalescent(args):
    replicates = args.replicates - 1 if args.separate else args.replicates

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

    def log_prob_coalescent(node_heights, theta):
        return ConstantCoalescent(theta).log_prob(node_heights)[0]

    print("  JIT off")

    if args.separate:
        start = timer()
        log_p = log_prob_coalescent(node_heights, theta)
        log_p.block_until_ready()
        end = timer()
        print(f"  First evaluation: {end-start} ({log_p})")
        if args.output:
            args.output.write(
                f"coalescent,evaluation1,off,{end - start},{log_p.squeeze().tolist()}\n"
            )

    start = timer()
    for _ in range(replicates):
        log_p = log_prob_coalescent(node_heights, theta)
        log_p.block_until_ready()
    end = timer()
    t1 = end - start
    print(f"  {replicates} evaluations: {t1} ({log_p.squeeze().tolist()})")

    if args.output:
        args.output.write(
            f"coalescent,evaluation,off,{end - start},{log_p.squeeze().tolist()}\n"
        )

    grad_fn = grad(log_prob_coalescent, (0, 1))

    if args.separate:
        start = timer()
        gradient = grad_fn(node_heights, theta)
        jax.tree_map(lambda x: x.block_until_ready(), gradient)
        end = timer()
        print(f"  First gradient evaluation: {end-start}")
        if args.output:
            args.output.write(f"coalescent,gradient1,off,{end - start},\n")

    start = timer()
    for _ in range(replicates):
        gradient = grad_fn(node_heights, theta)
        jax.tree_map(lambda x: x.block_until_ready(), gradient)
    end = timer()
    print(f"  {replicates} gradient evaluations: {end - start}")

    if args.output:
        args.output.write(f"coalescent,gradient,off,{end - start},\n")

    print("  JIT on")

    log_prob_jit = jit(log_prob_coalescent)

    if args.separate:
        start = timer()
        log_p = log_prob_jit(node_heights, theta)
        log_p.block_until_ready()
        end = timer()
        print(f"  First evaluation: {end-start} ({log_p})")
        if args.output:
            args.output.write(
                f"coalescent,evaluation1,on,{end - start},{log_p.squeeze().tolist()}\n"
            )

    start = timer()
    for _ in range(replicates):
        log_p = log_prob_jit(node_heights, theta)
        log_p.block_until_ready()
    end = timer()
    print(f"  {replicates} evaluations: {end - start} ({log_p.squeeze().tolist()})")

    if args.output:
        args.output.write(
            f"coalescent,evaluation,on,{end - start},{log_p.squeeze().tolist()}\n"
        )

    fn_jit_grad = jit(grad_fn)

    if args.separate:
        start = timer()
        gradient = fn_jit_grad(node_heights, theta)
        jax.tree_map(lambda x: x.block_until_ready(), gradient)
        end = timer()
        print(f"  First gradient evaluation: {end - start}")
        if args.output:
            args.output.write(f"coalescent,gradient1,on,{end - start},\n")

    start = timer()
    for _ in range(replicates):
        gradient = fn_jit_grad(node_heights, theta)
        jax.tree_map(lambda x: x.block_until_ready(), gradient)
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
    "--disable_separate",
    dest="separate",
    required=False,
    action="store_false",
    help="""Disable separate first call""",
)
parser.add_argument(
    '--gtr',
    action='store_true',
    help="""Include gradient calculation of GTR parameters""",
)

args = parser.parse_args()

if args.output:
    args.output.write("function,mode,JIT,time,logprob\n")

print("Tree likelihood unrooted:")
fluA_unrooted(args, 'JC69')

if args.gtr:
    print("Tree likelihood unrooted GTR:")
    fluA_unrooted(args, 'GTR')

print("Height transform log det Jacobian:")
ratio_transform_jacobian(args)

print("Node height transform:")
ratio_transform(args)

print("Constant coalescent:")
constant_coalescent(args)

if args.output:
    args.output.close()
