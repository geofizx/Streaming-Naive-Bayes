# Rough Clustering

###Description
    This algorithm clusters multi-dimensional feature sets with N number of instances (entities) based on an
    absolute integer-distance measure between all entities (sum of all absolute feature differences between any 2 entities).

    The basic objective is to divide a set of entities into discernible (uniquely clustered) entities and
    indiscernible (shared) entities by assigning these entities to subsets. Clusters are based on entity attributes
    and not statistical metrics.

    It also makes use of three properties of rough sets to enumerate these clusters (x_i) from the input feature set for entities:

        Upper Approximation - A_sup(x_i) - Set of all entities in a cluster that may be shared with other clusters.

        Lower Approximation - A_sub(x_i) - Subset of Upper Approximation with entities unique to that cluster, i.e., discernible entities

        Boundary Region - A_sup(x_i) - A_sub(x_i) - Difference between Upper and Lower Approximation which contain strictly
        non-unique (shared) entites, i.e., indiscernible and entities belongs to two or more upper approximations

    This code is an implementation of rough clustering as outlined by Voges, Pope & Brown, 2002, "Cluster Analysis of Marketing
    Data Examining On-line Shopping Orientation: A Comparison of k-means and Rough Clustering Approaches"

    This algorithm takes as input a feature set with integer features only

####Options
    max_clusters - integer corresponding to number of clusters to return
    objective (default="lower") - return max_clusters at distance D that maximizes this property of clusters
    max_d - Maximum intra-entity distance to consider before stopping further clustering

    if max_d is not specified, then algorithm determines max_d based on intra-entity distance
    statistics (25th percentile)

####Optimized Clusters
    The algorithm determines the optimal distance D for final clustering based on option 'objective' which maximizes :
    "lower" : sum of lower approximations (default) - maximum entity uniqueness across all clusters at distance D
    "coverage" : total # of entites covered by all clusters - maximum number of entities across all clusters at distance D
    "ratio" : ratio of lower/coverage - maximum ratio of unique entities to total entities across all clusters at distance D
    "all" : return clusters at every distance D from [0 - self.total_entities]
