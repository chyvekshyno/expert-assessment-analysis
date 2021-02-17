from typing import List, Any

import numpy as np


class SimpleNPCluster:
    """A simple class describes cluster using numpy"""

    def __init__(self):
        self.values = []

    def __init__(self, a: List[np.ndarray]):
        self.__init__()
        for o in a:
            self.values.append(o)

    def tolist(self):
        return self.values

    def vshape(self):
        return self.values[0].shape

    def mid(self) -> np.ndarray:
        res = np.zeros(self.values[0].shape)
        for o in self.values:
            res += o
        return res / len(self.values)

    def __append(self, o: np.ndarray):
        if self.vshape() != o.shape:
            assert 0
        self.values.append(o)


def combine(*clusters: SimpleNPCluster) -> SimpleNPCluster:
    allvalues: List[np.ndarray] = []
    for cluster in clusters:
        for o in cluster.tolist():
            allvalues.append(o)
    return SimpleNPCluster(allvalues)
