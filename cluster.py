from typing import List, Any

import numpy as np


class SimpleNPCluster:
    """A simple class describes cluster using numpy"""

    def __init__(self, names: List[str], data: List[np.ndarray]):
        self.names = []
        for name in names:
            self.names.append(name)
        self.values = []
        for o in data:
            self.values.append(o)

    def data(self) -> List[np.ndarray]:
        return self.values

    def allnames(self) -> List[str]:
        return self.names

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
    # combine data
    names = []
    data = []
    for cluster in clusters:
        for o in cluster.data():
            data.append(o)
        for name in cluster.allnames():
            names.append(name)

    return SimpleNPCluster(names, data)
