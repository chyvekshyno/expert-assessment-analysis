import os

from utils import *
from cluster import SimpleNPCluster as npcluster, combine

file_name = "f_mod.txt"
dir_path = "out"
comp_path = os.path.join(dir_path, file_name)

if __name__ == '__main__':
    # init parameters
    expert_assessment = np.array([
        [  # expert_1
            [0.653, 0.223, 0.124],  # param_1
            [0.321, 0.475, 0.204],  # param_2
            [0.219, 0.188, 0.593]  # param_3
        ], [  # expert_2
            [0.299, 0.346, 0.355],  # param_1
            [0.413, 0.248, 0.339],  # param_2
            [0.136, 0.579, 0.285]  # param_3
        ], [  # expert_3
            [0.392, 0.208, 0.400],  # param_1
            [0.717, 0.199, 0.084],  # param_2
            [0.125, 0.594, 0.281]  # param_3
        ], [  # expert_4
            [0.382, 0.401, 0.217],  # param_1
            [0.336, 0.462, 0.202],  # param_2
            [0.094, 0.410, 0.496]  # param_3
        ], [  # expert_5
            [0.555, 0.169, 0.276],  # param_1
            [0.292, 0.327, 0.381],  # param_2
            [0.124, 0.435, 0.441]  # param_3
        ], [  # expert_6
            [0.310, 0.411, 0.279],  # param_1
            [0.068, 0.316, 0.616],  # param_2
            [0.317, 0.534, 0.149]  # param_3
        ], [  # expert_7
            [0.409, 0.443, 0.148],  # param_1
            [0.402, 0.461, 0.137],  # param_2
            [0.345, 0.201, 0.454]  # param_3
        ], [  # expert_8
            [0.729, 0.187, 0.084],  # param_1
            [0.364, 0.500, 0.136],  # param_2
            [0.327, 0.190, 0.483]  # param_3
        ], [  # expert_9
            [0.276, 0.322, 0.402],  # param_1
            [0.530, 0.217, 0.253],  # param_2
            [0.198, 0.389, 0.413]  # param_3
        ]
    ])

    # import into clusters
    expert_clusters = np.array([], dtype=npcluster)
    for i in range(expert_assessment.shape[0]):
        expert_clusters = np.insert(expert_clusters,
                                    len(expert_clusters),
                                    [npcluster([str(i + 1)], [expert_assessment[i]])])

    # find first distance matrix
    distM = distance_matrix(expert_clusters)
    with open(comp_path, 'w') as f:
        f.write("\nDistance Matrix: \n"
                + "\n".join(" \t\t".join(map(str, o))
                            for o in np.around(distM, 3)))

    # trust radius
    distSUM = symmetric_matrix_dim_sums(distM)
    mediana_ind = argmediana(distM)
    trustRadius = trust_radius(distM)

    with open(comp_path, 'a') as f:
        f.write("\nDistance SUMS:\n" + " \t\t".join(str(o)
                                                    for o in np.around(distSUM, 3))
                + "\nMediana Index: "
                + str(mediana_ind + 1)
                + "\nTrust Radius: "
                + str(round(trustRadius, 4)))

    # clustering
    for i in range(5):
        cluster_ind1, cluster_ind2 = find_cluster_mediana(distM[mediana_ind], mediana_ind)
        cluster1 = expert_clusters[cluster_ind1]
        cluster2 = expert_clusters[cluster_ind2]

        cnew = combine(cluster1, cluster2)
        expert_clusters = np.insert(np.delete(expert_clusters, [cluster_ind1, cluster_ind2]), 0, [cnew])
        distM = distance_matrix(expert_clusters)
        distSUM = symmetric_matrix_dim_sums(distM)
        mediana_ind = argmediana(distM)

        with open(comp_path, 'a') as f:
            f.write("\n\n Iteration {}:".format(i + 1)
                    + "\nMediana min value ind:\t" + str(cluster_ind2 + 1)
                    + "\nNew cluster members: {}, {}".format(cluster1.allnames(), cluster2.allnames())
                    + "\nNew cluster assessment: "
                    + "\nCluster Members: " + ", ".join(cnew.names)
                    + "\nCluster Mid: \n"
                    + "\n".join(" \t\t".join(map(str, o)) for o in np.around(cnew.mid(), 3))
                    + "\nNew Distance Matrix: \n"
                    + "\n".join(" \t\t".join(map(str, o)) for o in np.around(distM, 3))
                    + "\nDistance SUMS:\n" + " \t\t".join(str(o)
                                                          for o in np.around(distSUM, 3))
                    + "\nMediana Index:\t" + str(mediana_ind + 1))

    with open(comp_path, 'a') as f:
        f.write("\n\n FINAL CLUSTERS: \n"
                + "\n".join("Cluster: " + str(cl.allnames())
                            for cl in expert_clusters))
