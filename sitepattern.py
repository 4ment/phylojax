import jax.numpy as np
from typing import List, Union, Any
import jax.ops

def get_dna_leaves_partials_compressed(alignment):
    weights = []
    keep = [True] * alignment.sequence_size

    patterns = {}
    indexes = {}
    for i in range(alignment.sequence_size):
        pat = tuple(alignment[name][i] for name in alignment)
        if pat in patterns:
            keep[i] = False
            patterns[pat] += 1.0
        else:
            patterns[pat] = 1.0
            indexes[i] = pat

    for i in range(alignment.sequence_size):
        if keep[i]:
            weights.append(patterns[indexes[i]])

    partials = []  # type: List[Union[List[None], Any]]
    # partials = np.empty((len(alignment)*2-1, 4, len(weights)))
    dna_map = {'a': [1.0, 0.0, 0.0, 0.0],
               'c': [0.0, 1.0, 0.0, 0.0],
               'g': [0.0, 0.0, 1.0, 0.0],
               't': [0.0, 0.0, 0.0, 1.0]}
    for idx, name in enumerate(alignment):
        temp = []
        for i, c in enumerate(alignment[name].symbols_as_string()):
            if keep[i]:
                temp.append(dna_map.get(c.lower(), [1., 1., 1., 1.]))
        # partials = jax.ops.index_update(partials, jax.ops.index[idx, :, :], np.transpose(np.array(temp)))
        partials.append(np.transpose(np.array(temp)))

    for i in range(len(alignment) - 1):
        partials.append([None] * len(patterns.keys()))

    return partials, np.array(weights)
