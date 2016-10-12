# Rough Clustering

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


