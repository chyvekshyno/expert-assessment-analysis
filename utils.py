from typing import List, Any, Tuple

from cluster import SimpleNPCluster as npcluster
import numpy as np


def distance(cl1: npcluster, cl2: npcluster) -> float:
    res = 0
    v1 = cl1.mid()
    v2 = cl2.mid()

    for i in range(cl1.vshape()[0]):
        for j in range(cl1.vshape()[1]):
            res += abs(v1[i][j] - v2[i][j])

    return res


def distance_matrix(morph_table: np.ndarray) -> np.ndarray:
    """
    calculate matrix if distances between clusters
    :param morph_table: list of SimpleNPCluster s
    :return: np.array (dim=2)
    """
    mdim = len(morph_table)
    res = np.ndarray(shape=(mdim, mdim)
                     , dtype=float)
    for i in range(mdim):
        res[i][i] = float('nan')
        for j in range(i + 1, mdim):
            dist = distance(morph_table[i], morph_table[j])
            res[i][j] = dist
            res[j][i] = dist
    return res


def symmetric_matrix_dim_sums(matrix: np.ndarray) -> np.ndarray:
    return np.nansum(matrix, axis=0).squeeze()


def argmediana(matrix: np.ndarray) -> int:
    """
    :return: column number with min value
    """
    res = np.nanargmin(symmetric_matrix_dim_sums(matrix))
    return res.take(0)


def trust_radius(matrix: np.ndarray) -> int:
    mediana = matrix[argmediana(matrix)]
    trust_count = matrix.shape[0] // 2
    trust_ind = np.argsort(mediana)[:trust_count]
    res = mediana[trust_ind[-1]] - mediana[trust_ind[0]]
    return res


def find_cluster(matrix: np.ndarray) -> Tuple[int, int]:
    flat_index = np.nanargmin(matrix)
    npcoord = np.unravel_index(flat_index, matrix.shape)
    coord = npcoord[0].take(0), npcoord[1].take(0)
    return coord
