import jax.numpy as np
import jax.experimental.optimizers
import jax.scipy.stats.norm as norm
import jax.scipy.stats.beta as beta
from jax import random
from jax.api import jit, grad, vmap
import argparse
from functools import partial

from .substitution import JC69, HKY, GTR
from .sitepattern import get_dna_leaves_partials_compressed
from .io import read_tree_and_alignment
from .treelikelihood import jax_likelihood
from .coalescent import ConstantCoalescent

from jax.config import config
config.update("jax_enable_x64", True)


def create_parser():
    parser = argparse.ArgumentParser(prog='phylotorch', description='Phylogenetic inference using pytorch')
    parser.add_argument('-t', '--tree', required=True, help="""Tree file""")
    parser.add_argument('-i', '--input', required=False, help="""Sequence file""")
    parser.add_argument('-m', '--model', choices=['JC69', 'HKY', 'GTR'], default='JC69',
                        help="""Substitution model [default: %(default)s]""")
    parser.add_argument('--model_rates', required=False, help="""Rate parameters of the substitution model""")
    parser.add_argument('--model_frequencies', required=False,
                        help="""Nucleotide frequencies of the substitution model""")
    # parser.add_argument('--libsbn', action='store_true', required=False, help="""Use libsbn""")
    parser.add_argument('--iter', type=int, default=100000, required=False, help="""Number of iterations""")
    parser.add_argument('--seed', type=int, required=False, default=None, help="""Initialize seed""")
    parser.add_argument('--rescale', type=bool, required=False, help="""Use rescaling""")
    parser.add_argument('-e', '--eta', required=False, type=float, default=0.01,
                        help="""eta for Stan script (variational only)""")
    parser.add_argument('--elbo_samples', required=False, type=int, default=100,
                        help="""Number of samples for Monte Carlo estimate of ELBO (variational only)""")
    parser.add_argument('--grad_samples', required=False, type=int, default=1,
                        help="""Number of samples for Monte Carlo estimate of gradients (variational only)""")
    return parser

# @jit(static_argnums=(0,1,3,4))
# @partial(jit, static_argnums=(1,3,4))
def elbo_fn(epoch, treelike, params, sampling_times, size=1, **kwargs):
    indices_for_jac = kwargs.get('indices_for_jac')
    bounds = kwargs.get('bounds')
    rng = random.PRNGKey(epoch)
    root_offset = max(sampling_times)
    ratios_a, ratios_b, clock_mu, clock_sigma, root_mu, root_sigma, theta_mu, theta_sigma = params
    coalescent = ConstantCoalescent(sampling_times)
    elbo = 0

    for i in range(size):
        rng, *subkeys = random.split(rng, 5)
        z_ratios = random.beta(subkeys[0], np.exp(ratios_a), np.exp(ratios_b))
        # z_root = root_mu + np.exp(root_sigma) * random.normal(subkeys[1])
        # z_clock = clock_mu + np.exp(clock_sigma) * random.normal(subkeys[2])
        # z_theta = theta_mu + np.exp(theta_sigma) * random.normal(subkeys[3])

        eta_root = random.normal(subkeys[1])
        eta_clock = random.normal(subkeys[2])
        eta_theta = random.normal(subkeys[3])
        z_root = root_mu + np.exp(root_sigma) * eta_root
        z_clock = clock_mu + np.exp(clock_sigma) * eta_clock
        z_theta = theta_mu + np.exp(theta_sigma) * eta_theta

        x_root = np.exp(z_root)
        x_clock = np.exp(z_clock)
        x_theta = np.exp(z_theta)

        log_q = np.sum(vmap(beta.logpdf)(z_ratios, np.exp(ratios_a), np.exp(ratios_b))) + \
                norm.logpdf(eta_root) + z_root + \
                norm.logpdf(eta_clock) + z_clock + \
                norm.logpdf(eta_theta) + z_theta

        # log_q = np.sum(vmap(beta.logpdf)(z_ratios, np.exp(ratios_a), np.exp(ratios_b))) + \
        #         norm.logpdf(z_root, root_mu, np.exp(root_sigma)) - z_root + \
        #         norm.logpdf(z_clock, clock_mu, np.exp(clock_sigma)) - z_clock + \
        #         norm.logpdf(z_theta, theta_mu, np.exp(theta_sigma)) - z_theta

        log_p, node_heights = treelike(x_root + root_offset, z_ratios, x_clock)
        log_prior = coalescent.log_prob(x_theta, node_heights) - z_theta
        log_det_jacobian = np.log(node_heights[indices_for_jac] - bounds[sampling_times.shape[0]:-1]).sum()

        # print(log_p, log_prior, log_q)
        elbo += log_p + log_prior - log_q + log_det_jacobian
        # print(log_p)

    elbo = elbo / size
    return np.squeeze(elbo)


# @jit
def objective(epoch, treelike, params, sampling_times, size=1, **kwargs):
    elbo = elbo_fn(epoch, treelike, params, sampling_times, size, **kwargs)
    return -elbo


def run_rooted(treelike, sampling_times, arg, **kwargs):
    if arg.seed is not None:
        rng = random.PRNGKey(arg.seed)
    else:
        rng = random.PRNGKey(1)

    sequence_count = sampling_times.shape[0]
    root_mu = np.array([1.])
    root_sigma = np.array([-1.])
    ratios_a = np.array([1.] * (sequence_count - 2))
    ratios_b = np.array([1.] * (sequence_count - 2))
    clock_mu = np.array([-6.])
    clock_sigma = np.array([-5.])
    theta_mu = np.array([3.])
    theta_sigma = np.array([-2.])

    init_params = (ratios_a, ratios_b,
                   clock_mu, clock_sigma,
                   root_mu, root_sigma,
                   theta_mu, theta_sigma)

    opt_init, opt_update, get_params = jax.experimental.optimizers.adam(arg.eta)
    opt_state = opt_init(init_params)
    root_offset = max(sampling_times)

    if arg.elbo_samples > 0:
        elbo = elbo_fn(0, treelike, init_params, sampling_times, arg.elbo_samples, **kwargs)
        print('Starting ELBO {}'.format(elbo))

    # @jit
    def update(epoch, opt_state, size, **kwargs):
        params = get_params(opt_state)
        grad_fn = grad(objective, 2)
        gradient = grad_fn(epoch, treelike, params, sampling_times, size, **kwargs)
        return opt_update(epoch, gradient, opt_state)

    for epoch in range(arg.iter):
        opt_state = update(epoch, opt_state, arg.grad_samples, **kwargs)

        if epoch % 100 == 0:
            params = get_params(opt_state)
            ratios_a, ratios_b, clock_mu, clock_sigma, root_mu, root_sigma, theta_mu, theta_sigma = params
            if arg.elbo_samples > 0:
                print('{} ELBO {}'.format(epoch, elbo_fn(epoch, treelike, params, sampling_times, arg.elbo_samples, **kwargs)))
            mean_root = np.exp(root_mu + np.exp(root_sigma)*np.exp(root_sigma.item()/2)) + root_offset
            mean_theta = np.exp(theta_mu + np.exp(theta_sigma) * np.exp(theta_sigma) / 2)
            mean_clock = np.exp(clock_mu + np.exp(clock_sigma)*np.exp(clock_sigma) / 2)
            print('root {} mode {} mu {} sigma {}'.format(mean_root, np.exp(root_mu) + root_offset, root_mu[0], np.exp(root_sigma)[0]))
            print('theta {} mode {} mu {} sigma {}'.format(mean_theta, np.exp(theta_mu)[0], theta_mu[0], np.exp(theta_sigma)[0]))
            print('clock {} mode {} mu {} sigma {}'.format(mean_clock, np.exp(clock_mu)[0], clock_mu[0], np.exp(clock_sigma)[0]))


def run(arg):

    tree, aln = read_tree_and_alignment(arg.tree, arg.input)
    if arg.model == 'JC69':
        subst_model = JC69()
    elif arg.model == 'GTR':
        if arg.model_rates:
            rates = np.array([float(x) for x in arg.model_rates.split(',')])
        else:
            rates = np.full(6, 1./6.)
        if arg.model_frequencies:
            frequencies = np.array([float(x) for x in arg.model_frequencies.split(',')])
        else:
            frequencies = np.full(4, 0.25)
        subst_model = GTR(rates, frequencies)
    elif arg.model == 'HKY':
        kappa = float(arg.model_rates) if arg.model_rates else 1.0
        if arg.model_frequencies:
            frequencies = np.array([float(x) for x in arg.model_frequencies.split(',')])
        else:
            frequencies = np.full(4, 0.25)
        subst_model = HKY(kappa, frequencies)
    else:
        exit(2)

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
    bounds = np.array(bounds)

    # preoder indexing to go from ratios to heights
    pre_indexing = []
    for node in tree.preorder_node_iter():
        if node != tree.seed_node:
            pre_indexing.append((node.parent_node.index, node.index))
    pre_indexing = np.array(pre_indexing)

    # indices for jacobian
    indices_for_jac = [None] * (taxa_count - 2)
    for idx_parent, idx in pre_indexing:
        if idx >= taxa_count:
            indices_for_jac[idx - taxa_count] = idx_parent - taxa_count

    sampling_times = bounds[:taxa_count]

    jax_likelihood_fn = partial(jax_likelihood, subst_model, partials, weights, bounds,
                                    pre_indexing,
                                    post_indexing)

    run_rooted(jax_likelihood_fn, sampling_times, arg, bounds=bounds, indices_for_jac=indices_for_jac)


def main():
    parser = create_parser()
    arg = parser.parse_args()
    run(arg)


if __name__ == "__main__":
    # import cProfile
    # import pstats
    # import io
    # # cProfile.run("main()")
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # pr.print_stats()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    # ps.print_stats()
    # pr.dump_stats('test.prof')
    # with open("prof.out", "w") as f:
    #     ps = pstats.Stats("profilingResults.cprof", stream=f)
    #     ps.sort_stats('cumulative')
    #     ps.print_stats()

    # with open('prof.out', 'w+') as f:
    #     f.write(s.getvalue())