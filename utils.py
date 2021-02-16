from cluster import SimpleNPCluster as npcluster


def distance(cl1: npcluster, cl2: npcluster) -> float:
    res = 0
    v1 = cl1.vmean()
    v2 = cl2.vmean()

    for i in range(cl1.vshape()[0]):
        for j in range(cl1.vshape()[1]):
            res += abs(v1[i][j]-v2[i][j])

    return res


def mediana(matrix):
    return 0


def find_cluster(matr):
    pass