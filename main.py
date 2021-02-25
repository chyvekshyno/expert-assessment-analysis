import os

from utils import *
from cluster import SimpleNPCluster as npcluster, combine
from data import expert_assessment

file_name = "f.txt"
dir_path = "out"
comp_path = os.path.join(dir_path, file_name)

if __name__ == '__main__':
    # init parameters


    # import into clusters
    expert_clusters = np.array([], dtype=npcluster)
    for i in range(norm(expert_assessment).shape[0]):
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
        cluster_ind1, cluster_ind2 = find_cluster(distM)
        cluster1 = expert_clusters[cluster_ind1]
        cluster2 = expert_clusters[cluster_ind2]

        cnew = combine(cluster1, cluster2)
        expert_clusters = np.insert(np.delete(expert_clusters, [cluster_ind1, cluster_ind2]), 0, [cnew])
        distM = distance_matrix(expert_clusters)

        with open(comp_path, 'a') as f:
            f.write("\n\n Iteration {}:".format(i + 1)
                    + "\nNew cluster members: {}, {}".format(cluster1.allnames(), cluster2.allnames())
                    + "\nNew cluster assessment: "
                    + "\nCluster Members: " + ", ".join(cnew.names)
                    + "\nCluster Mid: \n"
                    + "\n".join(" \t\t".join(map(str, o)) for o in np.around(cnew.mid(), 3))
                    + "\nNew Distance Matrix: \n"
                    + "\n".join(" \t\t".join(map(str, o)) for o in np.around(distM, 3)))

    with open(comp_path, 'a') as f:
        f.write("\n\n FINAL CLUSTERS: \n"
                + "\n".join("Cluster: " + str(cl.allnames())
                            for cl in expert_clusters))
