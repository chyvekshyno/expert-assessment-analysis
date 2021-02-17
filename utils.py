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


def distance_matrix(morph_table: List[npcluster]) -> np.ndarray:
    mdim = len(morph_table)
    res = np.ndarray(shape=(mdim, mdim)
                     , dtype=float)
    for i in range(mdim):
        res[i][i] = 0.
        for j in range(i + 1, mdim):
            dist = distance(morph_table[i], morph_table[j])
            res[i][j] = dist
            res[j][i] = dist
    return res


def distance_sums(matrix: np.ndarray) -> np.ndarray:
    return np.sum(matrix, axis=0).squeeze()


def argmediana(matrix: np.ndarray) -> int:
    res = np.argmin(distance_sums(matrix)).squeeze()[0]
    return res


def trust_radius(matrix: np.ndarray) -> int:
    mediana = matrix[argmediana(matrix)]
    trust_count = matrix.shape[0] / 2
    trust_ind = np.argsort(mediana)[:trust_count + 1]  # +1 cause of diagonal zero
    np.delete(np.min(trust_ind))  # delete index of diagonal zero
    res = abs(mediana[np.max(trust_ind)] - mediana[np.min(trust_ind)])
    return res


def find_cluster(matrix: np.ndarray) -> Tuple[int, int]:
    flat_index = np.argmin(matrix)
    npcoord = np.unravel_index(flat_index, matrix.shape)
    coord = npcoord[0].take(0), npcoord[1].take(0)
    return coord


def create_cluster(morph_table: List[npcluster], indecies: Tuple[int, int]) -> npcluster:
    return npcluster(morph_table[indecies[0]].tolist()
                     .append(morph_table[indecies[1]].tolist()))
