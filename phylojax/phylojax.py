import argparse
import random
import sys

import jax
import jax.experimental.optimizers
import jax.numpy as np
import jax.ops
import jax.scipy.optimize
from jax import grad, jit
from jax.config import config

from phylojax.transforms import SigmoidTransform, StickBreakingTransform

from .coalescent import ConstantCoalescent
from .io import read_tree_and_alignment
from .prior import ctmc_scale, dirichlet_logpdf
from .sitepattern import get_dna_leaves_partials_compressed
from .substitution import GTR, JC69
from .tree import (
    NodeHeightTransform,
    distance_to_ratios,
    heights_to_branch_lengths,
    transform_ratios,
)
from .treelikelihood import calculate_treelikelihood

config.update("jax_enable_x64", True)


def create_parser():
    parser = argparse.ArgumentParser(
        prog="phylojax", description="Phylogenetic inference using phylojax"
    )
    parser.add_argument("-t", "--tree", required=True, help="""Tree file""")
    parser.add_argument("-i", "--input", required=False, help="""Sequence file""")
    parser.add_argument(
        "-m",
        "--model",
        choices=["JC69", "GTR"],
        default="JC69",
        help="""Substitution model [default: %(default)s]""",
    )
    parser.add_argument(
        '--heights_init',
        choices=['tree'],
        help="""initialize node heights using input tree file""",
    )
    parser.add_argument(
        '--rate_init', type=float, help="""initialize substitution rate"""
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        choices=["advi", "map"],
        default="advi",
        type=str.lower,
        help="""Algorithm [default: %(default)s]""",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=100000,
        help="""Number of iterations""",
    )
    parser.add_argument("--nojit", action="store_true", help="""Disable JIT""")
    parser.add_argument("--seed", type=int, default=None, help="""Initialize seed""")
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        default=0.1,
        help="""Learning rate for variational inference""",
    )
    parser.add_argument(
        "--elbo_samples",
        type=int,
        default=100,
        help="""Number of samples for Monte Carlo estimate of ELBO""",
    )
    parser.add_argument(
        "--grad_samples",
        type=int,
        default=1,
        help="""Number of samples for Monte Carlo estimate of gradients""",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="""Output intermediate results"""
    )
    return parser


def neg_joint_likelihood(
    params,
    partials,
    weights,
    pre_indexing,
    post_indexing,
    indices_for_ratios,
    bounds,
    root_offset,
):
    params = np.expand_dims(params, 0)
    root = params[..., 0:1]
    clock = params[..., 1:2]
    theta = params[..., 2:3]
    ratios = params[..., 3:]

    x_root = np.exp(root) + root_offset
    x_clock = np.exp(clock)
    x_theta = np.exp(theta)
    x_ratios = SigmoidTransform()(ratios)

    ratios_root_height = np.concatenate((x_ratios, x_root), axis=-1)
    node_heights = transform_ratios(ratios_root_height, bounds, indices_for_ratios)

    model = JC69()

    return -joint_likelihood(
        partials,
        weights,
        pre_indexing,
        post_indexing,
        bounds,
        node_heights,
        x_clock,
        x_theta,
        model,
    ).sum()


def joint_likelihood(
    partials,
    weights,
    pre_indexing,
    post_indexing,
    bounds,
    internal_heights,
    clock,
    theta,
    model,
):

    taxa_count = internal_heights.shape[-1] + 1
    sampling_times = bounds[:taxa_count]

    branch_lengths = heights_to_branch_lengths(internal_heights, bounds, pre_indexing)

    bls = branch_lengths * clock
    mats = model.p_t(np.expand_dims(bls, -1))
    frequencies = np.broadcast_to(JC69().frequencies, bls.shape[:-1] + (4,))

    coalescent = ConstantCoalescent(theta)
    heights = np.concatenate(
        (
            np.broadcast_to(
                sampling_times, internal_heights.shape[:-1] + sampling_times.shape[-1:]
            ),
            internal_heights,
        ),
        axis=-1,
    )

    log_prior = coalescent.log_prob(heights) - np.log(theta) + ctmc_scale(bls, clock)

    log_p = calculate_treelikelihood(
        partials,
        weights,
        post_indexing,
        np.expand_dims(mats, -3),
        np.expand_dims(frequencies, axis=-2),
        np.array([[[1.0]]]),
    )
    return log_p + log_prior


def elbo_fn_aux(
    params,
    partials,
    weights,
    pre_indexing,
    post_indexing,
    indices_for_ratios,
    indices_for_jac,
    bounds,
    etas,
    root_offset,
    model,
):
    taxa_count = int((bounds.shape[0] + 1) / 2)
    param_count = taxa_count + 1
    z = jax.vmap(lambda a, b, c: a + np.exp(b) * c, (0, 0, 1), out_axes=1)(
        params[..., :param_count, 0],
        params[..., :param_count, 1],
        etas[..., :param_count],
    )

    sigmoid_transform = SigmoidTransform()
    height_transform = NodeHeightTransform(bounds, indices_for_ratios, indices_for_jac)

    z_root = z[..., 0:1]
    z_ratios = z[..., 1 : (taxa_count - 1)]
    z_clock = z[..., (taxa_count - 1) : taxa_count]
    z_theta = z[..., taxa_count:]

    x_root = np.exp(z_root) + root_offset
    x_ratios = sigmoid_transform(z_ratios)
    x_clock = np.exp(z_clock)
    x_theta = np.exp(z_theta)

    ratios_root_height = np.concatenate((x_ratios, x_root), axis=-1)
    internal_heights = height_transform(ratios_root_height)

    joint_p = joint_likelihood(
        partials,
        weights,
        pre_indexing,
        post_indexing,
        bounds,
        internal_heights,
        x_clock,
        x_theta,
        model,
    )

    entropy = (0.5 + 0.5 * np.log(2 * np.pi) + params[..., :param_count, 1]).sum()

    log_det_jacobians = (
        height_transform.log_abs_det_jacobian(ratios_root_height, internal_heights)
        + sigmoid_transform.log_abs_det_jacobian(z_ratios, x_ratios).sum(
            axis=-1, keepdims=True
        )
        + z_root
        + z_clock
        + z_theta
    )

    return joint_p, log_det_jacobians, entropy


def elbo_fn(
    params,
    partials,
    weights,
    pre_indexing,
    post_indexing,
    indices_for_ratios,
    indices_for_jac,
    bounds,
    etas,
    root_offset,
):
    joint_p, log_det_jacobians, entropy = elbo_fn_aux(
        params,
        partials,
        weights,
        pre_indexing,
        post_indexing,
        indices_for_ratios,
        indices_for_jac,
        bounds,
        etas,
        root_offset,
        JC69(),
    )

    return -(np.mean(joint_p + log_det_jacobians) + entropy)


def elbo_fn_gtr(
    params,
    partials,
    weights,
    pre_indexing,
    post_indexing,
    indices_for_ratios,
    indices_for_jac,
    bounds,
    etas,
    root_offset,
):
    taxa_count = int((bounds.shape[0] + 1) / 2)
    offset = taxa_count + 1  # n-1 heights, 1 clock, 1 theta
    z = jax.vmap(lambda a, b, c: a + np.exp(b) * c, (0, 0, 1), out_axes=1)(
        params[..., offset:, 0], params[..., offset:, 1], etas[..., offset:]
    )

    z_rates = z[..., :5]
    z_freqs = z[..., 5:8]

    stick_transform = StickBreakingTransform()
    x_rates = stick_transform(z_rates)
    x_freqs = stick_transform(z_freqs)

    joint_p, log_det_jacobians, entropy = elbo_fn_aux(
        params,
        partials,
        weights,
        pre_indexing,
        post_indexing,
        indices_for_ratios,
        indices_for_jac,
        bounds,
        etas,
        root_offset,
        GTR(x_rates, x_freqs),
    )

    joint_p += np.expand_dims(
        dirichlet_logpdf(x_rates, np.ones(6)), -1
    ) + np.expand_dims(dirichlet_logpdf(x_freqs, np.ones(4)), -1)

    log_det_jacobians += +np.expand_dims(
        stick_transform.log_abs_det_jacobian(z_rates, x_rates), -1
    ) + np.expand_dims(stick_transform.log_abs_det_jacobian(z_freqs, x_freqs), -1)
    entropy += (0.5 + 0.5 * np.log(2 * np.pi) + params[..., offset:, 1]).sum()
    return -(np.mean(joint_p + log_det_jacobians) + entropy)


def loss(params, size, rng, fn, **kwargs):
    rng, sub_key = jax.random.split(rng)
    etas = jax.random.normal(sub_key, (size, params.shape[0]))

    partials = kwargs["partials"]
    weights = kwargs["weights"]
    pre_indexing = kwargs["pre_indexing"]
    post_indexing = kwargs["post_indexing"]
    indices_for_jac = kwargs["indices_for_jac"]
    indices_for_ratios = kwargs["indices_for_ratios"]
    bounds = kwargs["bounds"]
    root_offset = max(bounds)

    return (
        fn(
            params,
            partials,
            weights,
            pre_indexing,
            post_indexing,
            indices_for_ratios,
            indices_for_jac,
            bounds,
            etas,
            root_offset,
        ),
        rng,
    )


def advi_rooted(x, arg, **kwargs):
    rng = jax.random.PRNGKey(arg.seed)
    taxa_count = kwargs['taxa_count']

    opt_init, opt_update, get_params = jax.experimental.optimizers.adam(arg.eta)
    opt_state = opt_init(x)

    if arg.nojit:
        if arg.model == 'JC69':
            grad_fn = grad(elbo_fn, 0)
            fn = elbo_fn
        else:
            grad_fn = grad(elbo_fn_gtr, 0)
            fn = elbo_fn_gtr
        update = opt_update
    else:
        if arg.model == 'JC69':
            grad_fn = jax.jit(grad(elbo_fn, 0))
            fn = jax.jit(elbo_fn)
        else:
            grad_fn = jax.jit(grad(elbo_fn_gtr, 0))
            fn = jax.jit(elbo_fn_gtr)
        update = jit(opt_update)

    if arg.elbo_samples > 0:
        elbo, rng = loss(x, arg.elbo_samples, rng, fn, **kwargs)
        print("Starting ELBO {}".format(-elbo))

    for epoch in range(1, arg.iter + 1):
        x = get_params(opt_state)
        gradient, rng = loss(x, arg.grad_samples, rng, grad_fn, **kwargs)
        opt_state = update(epoch, gradient, opt_state)

        if epoch % 100 == 0:
            x = get_params(opt_state)
            elbo, rng = loss(x, arg.elbo_samples, rng, fn, **kwargs)
            print(f"{epoch} ELBO {-elbo}")

            if arg.verbose:
                root_offset = max(kwargs["bounds"])
                mean_root = (
                    np.exp(x[0, 0] + np.exp(x[0, 1]) * np.exp(x[0, 1].item() / 2))
                    + root_offset
                )
                mean_clock = np.exp(
                    x[(taxa_count - 1), 0]
                    + np.exp(x[(taxa_count - 1), 1])
                    * np.exp(x[(taxa_count - 1), 1])
                    / 2
                )
                mean_theta = np.exp(
                    x[taxa_count, 0]
                    + np.exp(x[taxa_count, 1]) * np.exp(x[taxa_count, 1]) / 2
                )
                print(
                    "root {} mode {} mu {} sigma {}".format(
                        mean_root,
                        np.exp(x[0, 0]) + root_offset,
                        x[0, 0],
                        np.exp(x[0, 1]),
                    )
                )
                print(
                    "clock {} mode {} mu {} sigma {}".format(
                        mean_clock,
                        np.exp(x[(taxa_count - 1), 0]),
                        x[(taxa_count - 1), 0],
                        np.exp(x[(taxa_count - 1), 1]),
                    )
                )
                print(
                    "theta {} mode {} mu {} sigma {}".format(
                        mean_theta,
                        np.exp(x[taxa_count, 0]),
                        x[taxa_count, 0],
                        np.exp(x[taxa_count, 1]),
                    )
                )


def bfgs_rooted(x0, arg, **kwargs):
    partials = kwargs["partials"]
    weights = kwargs["weights"]
    pre_indexing = kwargs["pre_indexing"]
    post_indexing = kwargs["post_indexing"]
    indices_for_ratios = kwargs["indices_for_ratios"]
    bounds = kwargs["bounds"]
    root_offset = max(bounds)

    options = {"maxiter": arg.iter, "gtol": 1e-9}
    args = (
        partials,
        weights,
        pre_indexing,
        post_indexing,
        indices_for_ratios,
        bounds,
        root_offset,
    )
    res = jax.scipy.optimize.minimize(
        neg_joint_likelihood, x0, args, method="BFGS", options=options
    )

    root = np.exp(res.x[0:1])
    clock = np.exp(res.x[1:2])
    theta = np.exp(res.x[2:3])

    print(f"root {root + root_offset}")
    print(f"theta {theta}")
    print(f"clock {clock}")

    print(
        f"LL: {-res.fun} #iter: {res.nit} #fn: {res.nfev} "
        f"#grad: {res.njev} status: {res.status}"
    )


def run(arg):
    if arg.seed is None:
        arg.seed = random.randrange(sys.maxsize)
    print("seed", arg.seed)

    tree, aln = read_tree_and_alignment(arg.tree, arg.input)

    partials, weights = get_dna_leaves_partials_compressed(aln)
    taxa_count = len(tree.taxon_namespace)
    bounds = [None] * (2 * taxa_count - 1)
    # postorder for peeling
    post_indexing = []
    for node in tree.postorder_node_iter():
        if node.is_leaf():
            bounds[node.index] = node.date
        else:
            bounds[node.index] = max([bounds[x.index] for x in node.child_node_iter()])
            children = node.child_nodes()
            post_indexing.append((node.index, children[0].index, children[1].index))
    post_indexing = tuple(post_indexing)
    bounds = np.array(bounds)

    # indices for heights to branch length
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

    sampling_times = bounds[:taxa_count]

    partials = np.array(partials[:taxa_count])

    if arg.heights_init is not None:
        ratios, root_height, bounds = distance_to_ratios(tree)

    if arg.algorithm == "map":
        if True:
            key = jax.random.PRNGKey(arg.seed)
            x0 = jax.random.uniform(key, (taxa_count - 2 + 3,), minval=-2, maxval=2)
        else:
            root = np.log(np.array([3.0]))
            ratios = np.array([0.1] * (taxa_count - 2))
            clock = np.log(np.array([0.003]))
            theta = np.log(np.array([3.0]))
            x0 = np.concatenate((root, clock, theta, ratios), -1)

        bfgs_rooted(
            x0,
            arg,
            bounds=bounds,
            indices_for_jac=indices_for_jac,
            indices_for_ratios=indices_for_ratios,
            pre_indexing=pre_indexing,
            post_indexing=post_indexing,
            partials=partials,
            weights=weights,
            taxa_count=taxa_count,
        )
    else:
        taxa_count = sampling_times.shape[0]

        root_mu = np.array([1.0])
        root_sigma = np.array([-1.0])
        ratios_mu = np.array([1.0] * (taxa_count - 2))
        ratios_sigma = np.array([1.0] * (taxa_count - 2))
        clock_mu = np.array([-6.0])
        clock_sigma = np.array([-5.0])
        theta_mu = np.array([3.0])
        theta_sigma = np.array([-2.0])

        if arg.heights_init is not None:
            ratios_mu = SigmoidTransform().inverse(ratios)
            ratios_sigma = np.log(np.repeat(1.0e-6, ratios_mu.shape[-1]))
            root_mu, root_sigma = np.log(root_height - max(bounds)), np.log(
                np.array([1.0e-6])
            )

        if arg.rate_init is not None:
            rate = np.array([arg.rate_init])
            clock_mu, clock_sigma = np.log(rate), np.log(rate * 0.01)

        mus = np.concatenate(
            (
                root_mu,
                ratios_mu,
                clock_mu,
                theta_mu,
            ),
            0,
        )
        sigmas = np.concatenate(
            (
                root_sigma,
                ratios_sigma,
                clock_sigma,
                theta_sigma,
            ),
            0,
        )

        if arg.model == "GTR":
            mus = np.concatenate(
                (
                    mus,
                    np.zeros(5),
                    np.zeros(3),
                ),
                0,
            )
            sigmas = np.concatenate(
                (
                    sigmas,
                    np.full([5], -5.0),
                    np.full([3], -5.0),
                ),
                0,
            )
        x = np.vstack((mus, sigmas)).transpose()

        advi_rooted(
            x,
            arg,
            bounds=bounds,
            indices_for_jac=indices_for_jac,
            indices_for_ratios=indices_for_ratios,
            pre_indexing=pre_indexing,
            post_indexing=post_indexing,
            partials=partials,
            weights=weights,
            taxa_count=taxa_count,
        )


def main():
    parser = create_parser()
    arg = parser.parse_args()
    run(arg)


if __name__ == "__main__":
    main()
