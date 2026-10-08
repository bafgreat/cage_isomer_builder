---
description: >-
  What each CageStatistics field means, how endo/exo sites are decided, and
  how to load the FG-FG distance statistics into pandas and plot them.
---

# Statistics

`get_statistics()` summarises the single-group isomers of a structure: the
isomer count and the distance between every pair of functional-group sites
on different linkers, split by where the sites point.

```python
from cage_isomer_builder import example_data
from cage_isomer_builder.cage import Tri4Di6CageBuilder

cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
                          linker=example_data.path("bdc"), scale_multiplier=0.225)
cage.build()
cage.optimise(include_charges=False, logfile=None)
cage.functionalise()

stats = cage.get_statistics(n_kde_bins=200)
print(stats.n_linkers, stats.n_fg_slots, stats.n_unique_isomers)   # 6 24 176
print(sum(stats.total_fg_pairs))      # 240 site pairs on different linkers
print(sum(stats.other_pairs))         # 240: every site of this pore is tangential
```

| Field | Meaning |
| --- | --- |
| `n_linkers` | Number of linkers carrying sites. |
| `n_fg_slots` | Total number of functional-group sites. |
| `n_unique_isomers` | Symmetry-unique isomers with one group per linker (Burnside's lemma). |
| `distance_keys` | Sorted distinct site-site distances (A, two decimals). |
| `total_fg_pairs` | Number of site pairs at each distance. |
| `endo_endo_pairs` | Of those, pairs where both sites point into the pore. |
| `endo_exo_pairs` | Pairs with one site pointing in and one out. |
| `exo_exo_pairs` | Pairs where both sites point out. |
| `other_pairs` | Pairs involving a tangential (or unclassified) site. The four classes add up to `total_fg_pairs`. |
| `kde_x` | Distance grid of the smoothed curve (`n_kde_bins` points). |
| `kde_y` | Smoothed pair counts on `kde_x`. |

## Endo, exo and tangential

A site is *endo* when its C-site bond points within 60 degrees of the
direction to the pore centre, *exo* when it points within 60 degrees of the
opposite direction, and *tangential* otherwise. In the optimised UiO-66 pore
every phenylene ring lies face-on to the pore, so all its sites are
tangential. `get_statistics(threshold=0.5)` sets the cosine limit.
Periodic structures have no single pore centre: their sites are unclassified
and pairs are counted per unit cell over all periodic images up to
`max_distance` (default 20 A).

```python
slots = cage.slot_features()
print(sorted(set(slots.orientation.tolist())))   # ['tangential']
```

## Counts and the smoothed curve

`distance_keys` and the four class columns are an exact histogram. `kde_y`
smooths it: each pair adds a narrow Gaussian (0.1 A wide), and the curve is
scaled so that its values add up to the number of pairs, so it can be drawn
on the same axis as the counts. For several groups, ratios, missing linkers
or ring rotations, use [Descriptors](descriptors.md) and
[Multivariate pores & defects](multivariate.md).

The per-pair counts are proportional to the average over all placements of
one group per linker: every pair of sites on different linkers is occupied
in the same fraction of placements.

## In pandas

```python
import pandas as pd

table = pd.DataFrame({
    "distance": stats.distance_keys,
    "total": stats.total_fg_pairs,
    "endo_endo": stats.endo_endo_pairs,
    "endo_exo": stats.endo_exo_pairs,
    "exo_exo": stats.exo_exo_pairs,
    "other": stats.other_pairs,
})
curve = pd.DataFrame({"distance": stats.kde_x, "pairs": stats.kde_y})
print(table.shape)          # one row per distinct distance
```

## Plotting

With matplotlib:

```python
import matplotlib
matplotlib.use("Agg")       # no window needed; leave out in a notebook
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
bottom = 0
for column in ("endo_endo", "endo_exo", "exo_exo", "other"):
    ax.bar(table["distance"], table[column], width=0.08, bottom=bottom, label=column)
    bottom = bottom + table[column]
ax.plot(curve["distance"], curve["pairs"], color="black", label="smoothed")
ax.set_xlabel("FG-FG distance (A)")
ax.set_ylabel("Pairs")
ax.legend()
fig.savefig("fg_distances.png", dpi=150)
```
