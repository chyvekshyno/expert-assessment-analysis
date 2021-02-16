import numpy as np


class SimpleNPCluster:
    """A simple class describes cluster using numpy"""

    def __init__(self, o: np.ndarray):
        self.values = []
        self.values.append(o)

    def members(self):
        return self.values

    def vshape(self):
        return self.values[0].shape

    def append(self, o: np.ndarray):
        if self.vshape() != o.shape:
            assert 0
        self.values.append(o)

    def vmean(self) -> np.ndarray:
        res = np.zeros(self.values[0].shape)
        for o in self.values:
            res += o

        return res / len(self.values)
