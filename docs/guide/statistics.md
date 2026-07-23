# Statistics

Summary statistics can be calculated after isomer enumeration. These include
the number of unique isomers, functional-group pair counts, distance
distributions, and endo/exo classifications.

```python
stats = cage.get_statistics(n_kde_bins=200)
print(f"{stats.n_unique_isomers} unique isomers, "
      f"{stats.total_fg_pairs} FG-FG pairs "
      f"({stats.endo_endo_pairs} endo-endo)")
```

See [`CageStatistics`](../reference/cage.md) for every field returned.
