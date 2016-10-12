# Rough Clustering

This algorithm clusters multi-dimensional feature sets with N number of instances (entities) based on an
absolute integer-distance measure between all entities (sum of all absolute feature differences between any 2 entities).

The basic objective is to divide a set of entities into discernible (uniquely clustered) entities and
indiscernible (shared) entities by assigning these entities to subsets. Clusters are based on entity attributes
and not statistical metrics.

It also makes use of three properties of rough sets to enumerate these clusters from the input feature set:

    Upper Approximation - A^*(xi) Set of all entities in a cluster that may be shared with other clusters.
     (p ∈ A(xi) → p ∈ A(xi))
    Lower Approximation - Subset of Upper Approximation with entities unique to that cluster, i.e., discernible entities
     (A(Xi) ⊆ A(Xi))
    Boundary Region - Difference between Upper and Lower Approximation which contain strictly non-unique (shared) entites, i.e., indiscernible

These
1. An entity can be part of at most one lower approximation. This implies that any two lower approximations do not overlap.
2. An entitiy that is member of a lower approximation of a set is also part of its upper approximation (v ∈ A(xi) → v ∈ A(xi)).
This implies that a lower approximation of a set is a subset of its corresponding upper approximation (A(Xi) ⊆ A(Xi)).
3. If an entity is not part of any lower approximation it belongs to two or more upper approximations. This implies that
an object cannot belong to only a single boundary region.