import jax.numpy as np


class ConstantCoalescent:
    def __init__(self, theta):
        self.theta = theta

    def log_prob(self, node_heights):
        taxa_shape = node_heights.shape[:-1] + (int((node_heights.shape[-1] + 1) / 2),)
        node_mask = np.concatenate(
            (
                np.full(taxa_shape, False),
                np.full(
                    taxa_shape[:-1] + (taxa_shape[-1] - 1,),
                    True,
                ),
            ),
            axis=-1,
        )
        indices = np.argsort(node_heights)
        heights_sorted = np.take_along_axis(node_heights, indices, 0)
        node_mask_sorted = np.take_along_axis(node_mask, indices, 0)
        lineage_count = np.where(
            node_mask_sorted,
            np.full_like(self.theta, -1),
            np.full_like(self.theta, 1),
        ).cumsum(-1)[..., :-1]
        durations = heights_sorted[..., 1:] - heights_sorted[..., :-1]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0
        return np.sum(-lchoose2 * durations / self.theta, axis=-1, keepdims=True) - (
            taxa_shape[-1] - 1
        ) * np.log(self.theta)
