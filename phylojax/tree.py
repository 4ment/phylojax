import jax.numpy as np
import jax.ops

from phylojax.transforms import SigmoidTransform


def postorder_indices(tree):
    indices = []
    for node in tree.postorder_node_iter():
        if not node.is_leaf():
            children = node.child_nodes()
            indices.append((node.index, children[0].index, children[1].index))
    return indices


def preorder_indices(tree):
    indices = []
    children = tree.seed_node.child_nodes()
    indices.append((children[0].index, children[1].index, tree.seed_node.index))
    indices.append((children[1].index, children[0].index, tree.seed_node.index))
    for node in tree.preorder_node_iter(
        lambda n: n.index not in (children[0].index, children[1].index)
    ):
        if node != tree.seed_node:
            sibling = node.sibling_nodes()[0]
            indices.append((node.index, sibling.index, node.parent_node.index))
    return indices


def heights_to_ratios(tree, internal_heights, bounds):
    preorder = np.array(
        [
            (node.parent_node.index, node.index)
            for node in tree.preorder_node_iter()
            if node != tree.seed_node
        ]
    )
    ratios_root_height = transform_ratios_inv(internal_heights, bounds, preorder)
    ratios = SigmoidTransform().inverse(ratios_root_height[..., :-1])
    return ratios, ratios_root_height[..., -1]


def distance_to_ratios(tree, eps=1.0e-6):
    taxa_count = len(tree.taxon_namespace)
    bounds = [None] * (2 * taxa_count - 1)
    heights = [None] * (2 * taxa_count - 1)
    ratios = [None] * (taxa_count - 2)

    for node in tree.postorder_node_iter():
        if node.is_leaf():
            bounds[node.index] = node.date
            heights[node.index] = node.date
        else:
            bounds[node.index] = max([bounds[x.index] for x in node.child_node_iter()])
            heights[node.index] = max(
                [
                    heights[c.index] + jax.lax.clamp(eps, c.edge_length, np.inf)
                    for c in node.child_node_iter()
                ]
            )

    for node in tree.preorder_node_iter():
        if node != tree.seed_node:
            if not node.is_leaf():
                ratios[node.index - taxa_count] = (
                    heights[node.index] - bounds[node.index]
                ) / (heights[node.parent_node.index] - bounds[node.index])

    return np.array(ratios), np.array(heights[-1:]), np.array(bounds)


def heights_to_branch_lengths(node_heights, bounds, indices_sorted):
    taxa_count = int((bounds.shape[0] + 1) / 2)
    return np.concatenate(
        (
            node_heights[..., indices_sorted[0, :taxa_count] - taxa_count]
            - bounds[:taxa_count],
            node_heights[..., indices_sorted[0, taxa_count:] - taxa_count]
            - node_heights[..., indices_sorted[1, taxa_count:] - taxa_count],
        ),
        -1,
    )


def transform_ratios(ratios_root_height, bounds, indexing):
    taxa_count = ratios_root_height.shape[-1] + 1
    heights = np.empty_like(ratios_root_height)
    heights = jax.ops.index_update(
        heights, jax.ops.index[..., -1], ratios_root_height[..., -1]
    )

    def f(i, heights):
        parent_id, id_ = indexing[i]
        return jax.ops.index_update(
            heights,
            jax.ops.index[..., id_ - taxa_count],
            bounds[id_]
            + ratios_root_height[..., id_ - taxa_count]
            * (heights[..., parent_id - taxa_count] - bounds[id_]),
        )

    return jax.lax.fori_loop(0, len(indexing), f, heights)


def transform_ratios_inv(internal_heights, bounds, indices):
    taxa_count = internal_heights.shape[-1] + 1
    bounds = bounds[indices[1, taxa_count:]]
    return np.concatenate(
        (
            (
                internal_heights[
                    ...,
                    indices[1, taxa_count:] - taxa_count,
                ]
                - bounds
            )
            / (
                internal_heights[
                    ...,
                    indices[0, taxa_count:] - taxa_count,
                ]
                - bounds
            ),
            internal_heights[..., -1:],
        )
    )


def log_abs_det_jacobian(node_heights, indices, bounds):
    return np.log(node_heights[..., indices] - bounds).sum(axis=-1, keepdims=True)


def setup_indexes(tree):
    for node in tree.postorder_node_iter():
        node.index = -1
        node.annotations.add_bound_attribute("index")

    s = len(tree.taxon_namespace)
    taxa_dict = {taxon.label: idx for idx, taxon in enumerate(tree.taxon_namespace)}

    for node in tree.postorder_node_iter():
        if not node.is_leaf():
            node.index = s
            s += 1
        else:
            node.index = taxa_dict[node.taxon.label]


def setup_dates(tree, heterochronous=False):
    # parse dates
    if heterochronous:
        dates = {}
        for node in tree.leaf_node_iter():
            dates[str(node.taxon)] = float(str(node.taxon).split("_")[-1][:-1])

        max_date = max(dates.values())
        min_date = min(dates.values())

        # time starts at 0
        if min_date == 0:
            for node in tree.leaf_node_iter():
                node.date = dates[str(node.taxon)]
            oldest = max_date
        # time is a year
        else:
            for node in tree.leaf_node_iter():
                node.date = max_date - dates[str(node.taxon)]
            oldest = max_date - min_date
    else:
        for node in tree.postorder_node_iter():
            node.date = 0.0
        oldest = None

    return oldest


class NodeHeightTransform:
    def __init__(self, bounds, indices, indices_for_jac):
        self.bounds = bounds
        self.indices = indices
        self.indices_for_jac = indices_for_jac
        self.taxa_count = int((self.bounds.shape[-1] + 1) / 2)

    def __call__(self, ratios_root_height):
        return transform_ratios(ratios_root_height, self.bounds, self.indices)

    def log_abs_det_jacobian(self, x, y):
        return log_abs_det_jacobian(
            y, self.indices_for_jac, self.bounds[self.taxa_count : -1]
        )
