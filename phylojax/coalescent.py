import jax.numpy as np


class ConstantCoalescent(object):
    def __init__(self, sampling_times):
        self.sampling_times = sampling_times
        self.taxon_count = sampling_times.shape[0]

    def log_prob(self, theta, node_heights):
        heights = np.concatenate([self.sampling_times, node_heights], 0)
        node_mask = np.array([1.]*self.taxon_count + [-1.]*(self.taxon_count-1))
        indices = np.argsort(heights)
        heights_sorted = np.take_along_axis(heights, indices, 0)
        node_mask_sorted = np.take_along_axis(node_mask, indices, 0)
        lineage_count = node_mask_sorted.cumsum(0)[:-1]
        durations = heights_sorted[1:] - heights_sorted[:-1]
        lchoose2 = lineage_count * (lineage_count - 1) / 2.0
        return np.sum(-lchoose2 * durations / theta) - (self.taxon_count - 1)*np.log(theta)
