---
description: >-
  What each CageStatistics field means, and how to load them into pandas
  and plot the FG-FG distance distribution.
---

# Statistics

Summary statistics can be calculated after isomer enumeration:

```python
stats = cage.get_statistics(n_kde_bins=200)
```

`stats` is a [`CageStatistics`](../reference/cage.md) named tuple with these
fields:

| Field | Meaning |
| --- | --- |
| `n_linkers` | Number of linkers in the cage. |
| `n_fg_slots` | Total functional-group anchor slots (`n_linkers × 4`). |
| `n_unique_isomers` | Symmetry-unique isomer count (an `int` - see below). |
| `distance_keys` | Sorted unique inter-linker FG-to-FG distances (Å). |
| `total_fg_pairs` | Total active FG pair count at each distance in `distance_keys`. |
| `endo_endo_pairs` | Of those, pairs where both anchors point inward. |
| `endo_exo_pairs` | Pairs with one inward and one outward anchor. |
| `exo_exo_pairs` | Pairs where both anchors point outward. |
| `kde_x` | Distance grid the KDE was evaluated on (`n_kde_bins` points). |
| `kde_y` | Smoothed pair-count curve at each `kde_x` - see below. |

`distance_keys`, `total_fg_pairs`, `endo_endo_pairs`, `endo_exo_pairs`, and
`exo_exo_pairs` are parallel arrays - same length, one entry per distance
bin. `kde_x`/`kde_y` are a separate pair at their own resolution
(`n_kde_bins`, independent of how many distinct distances actually occur),
not aligned with the others.

## What `kde_x`/`kde_y` actually are

`distance_keys`/`total_fg_pairs` is already an exact histogram (e.g. "6
pairs at 5.2 Å, 9 pairs at 8.1 Å"), which as a bar chart gets spiky and hard
to read once there are many close-together distances. The KDE curve is the
same information smoothed into one continuous line: each individual pair
contributes a narrow Gaussian bump (`bandwidth=0.1` Å - narrow enough that
distinct distances don't blur into each other) centered on its own
distance, and all the bumps are summed. The result is then rescaled so its
height matches actual pair counts rather than integrating to 1 (the usual
KDE convention) - that's why `kde_y` can be plotted on the same axis as
`total_fg_pairs`, as in the example below.

## `n_unique_isomers` vs. `enumerate_isomers()`

These compute the same quantity two different ways, and will always agree:

- `cage.enumerate_isomers()` exhaustively walks the raw combinatorial space,
  groups colorings into symmetry orbits, and returns one representative
  structure per orbit - so `len(cage.enumerate_isomers())` is a count you
  get by actually generating every isomer.
- `stats.n_unique_isomers` (and `cage.count_unique_isomers_burnside()`, which
  `get_statistics()` calls internally) reaches the same number via Burnside's
  lemma - a closed-form counting formula - without generating a single
  structure. Use this when you only need the count and the cage is too large
  to fully enumerate.

## Loading it into pandas

The distance-binned fields are already a tidy table shape - one row per
distance:

```python
import pandas as pd

df = pd.DataFrame({
    "distance": stats.distance_keys,
    "total": stats.total_fg_pairs,
    "endo_endo": stats.endo_endo_pairs,
    "endo_exo": stats.endo_exo_pairs,
    "exo_exo": stats.exo_exo_pairs,
})
```

The KDE curve gets its own frame:

```python
kde_df = pd.DataFrame({"distance": stats.kde_x, "density": stats.kde_y})
```

## Plotting

`cage_isomer_builder` doesn't ship a plotting function - once it's a
DataFrame, use whatever you'd normally reach for. For example, a stacked bar
of the endo/exo breakdown with the KDE curve overlaid:

```python
import matplotlib.pyplot as plt

ax = df.set_index("distance")[["endo_endo", "endo_exo", "exo_exo"]].plot.bar(
    stacked=True, figsize=(8, 4),
)
ax.plot(kde_df["distance"], kde_df["density"], color="black", label="KDE")
ax.set_xlabel("FG-FG distance (Å)")
ax.set_ylabel("Pair count")
ax.legend()
plt.tight_layout()
plt.show()
```
