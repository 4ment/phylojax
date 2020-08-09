import jax.numpy as np
import jax
from jax import jit

def distance_to_ratios(tree):
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
            heights[node.index] = max([heights[c.index] + c.edge_length for c in node.child_node_iter()])

    for node in tree.preorder_node_iter():
        if node != tree.seed_node:
            if not node.is_leaf():
                ratios[node.index - taxa_count] = (heights[node.index] - bounds[node.index])/(heights[node.parent_node.index] - bounds[node.index])

    return ratios, heights[tree.seed_node.index], bounds



# @jax.partial(jit, static_argnums=(1,2))
# @jit
def heights_to_branch_lengths(node_heights, bounds, indexing):
    taxa_count = int((bounds.shape[0] + 1)/2)
    indices_sorted = indexing[np.argsort(indexing[:, 1])].transpose()
    return np.concatenate((node_heights[indices_sorted[0, :taxa_count] - taxa_count] - bounds[:taxa_count],
                      node_heights[indices_sorted[0, taxa_count:] - taxa_count] - node_heights[indices_sorted[1, taxa_count:] - taxa_count]))


def transform_ratios(root_height, ratios, bounds, indexing):
    ratios2 = np.split(ratios, ratios.shape[0])
    taxa_count = ratios.shape[0] + 2
    heights = [None]*(taxa_count-1)
    heights[-1] = root_height

    for parent_id, id in indexing:
        if id >= taxa_count:
            heights[id - taxa_count] = bounds[id] + ratios2[id - taxa_count] * (
                    heights[parent_id - taxa_count] - bounds[id])

    return np.concatenate(heights)


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
            dates[str(node.taxon)] = float(str(node.taxon).split('_')[-1][:-1])

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

