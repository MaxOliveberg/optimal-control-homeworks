import dataclasses

import numpy as np


@dataclasses.dataclass
class KnapsackSettings:
    max_weight: int = 10
    # values in first row, weights in second
    items: np.ndarray = np.array([[0, 1, 1, 1, 2, 3, 5],
                                  [0, 3, 4, 10, 7, 8, 11]])


def solve_knapsack(settings: KnapsackSettings = KnapsackSettings()):
    # init ret array, a 2 x max_weight matrix, first row item index, second value to go
    ret = np.zeros((2, settings.max_weight))
    for k in range(settings.max_weight):
        # check weight post-pick at current weight
        post_action_weights = settings.max_weight - (k + 1) + settings.items[1, :]
        # Replace invalid picks with -inf, otherwise V(x) = R + V(x').
        values = np.where(post_action_weights < settings.max_weight,
                          settings.items[0] + ret[1][
                              np.minimum(post_action_weights,
                                         (settings.max_weight - 1) *
                                         np.ones(len(settings.items[0]))).astype("int32")],
                          - np.infty)
        # Pick best
        value, index = values.max(), values.argmax()
        # Update array
        ret[:, settings.max_weight - (k + 1)] = np.array([index, value])
    return ret
