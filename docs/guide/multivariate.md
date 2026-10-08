---
description: >-
  Exact functional-group distance distributions of multivariate pores (two or
  more groups, any ratio) and defective pores (missing linkers or nodes),
  averaged over every placement of the groups.
---

# Multivariate pores & defects

When the identity of each building block is itself uncertain (a 50:50
NH2/CH3 MOF, a 25 % functionalised pore, a pore with a missing linker), the
pore is described by its *distribution within a distribution*: how many
NH2-NH2, NH2-CH3 and CH3-CH3 pairs (endo-endo, endo-exo, ...) there are at
each distance, averaged over every way of placing the groups.

```python
from cage_isomer_builder import example_data
from cage_isomer_builder.cage import Tri4Di6CageBuilder

cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
                          linker=example_data.path("bdc"), scale_multiplier=0.225)
cage.build()
cage.optimise(include_charges=False, logfile=None)
cage.functionalise()
```

## Exact ensemble averages

`get_ensemble_descriptor` counts every placement of the groups once (which
equals averaging the symmetry-unique isomers weighted by how many
placements each stands for). The result has a closed form, so nothing is
enumerated; weights are the expected number of pairs per pore.

```python
ens = cage.get_ensemble_descriptor(linker_patterns={("NH2",): 3, ("CH3",): 3}, by="label")
for key, (n, weight) in ens.summary().items():
    print(key, round(weight, 3))
# ('fg:CH3', 'fg:CH3') 3.0
# ('fg:CH3', 'fg:NH2') 9.0
# ('fg:NH2', 'fg:NH2') 3.0
```

With `by="orientation"` (the default) each pair type is also split by the
sites' endo/exo/tangential class. Named groups are tuples or lists; a string
means one capital letter per group (`"AB"` is an A and a B on one linker).

| What | Arguments |
| --- | --- |
| one group per linker | `groups="A"` |
| two groups on every linker | `groups="AB"` |
| fixed numbers of linkers per pattern | `linker_patterns={"A": 3, "B": 3}` |
| partial functionalisation | `linker_patterns={"A": 3, "": 3}` (`""` = no group) |
| missing linkers | `n_missing_linkers=1` (or a vacancy pattern) |
| any composition of a large crystal | `mode="independent", fractions={"A": 0.16, "B": 0.84}` |
| pairs on one linker too | `include_same_linker=True` |

## Ratios

Every whole-linker ratio, and linear interpolation between them:

```python
from cage_isomer_builder.utils.ensemble import interpolate_descriptor, ratio_series

series = ratio_series(cage.slot_features(), cage.slot_layout(), labels=("A", "B"), by="label")
print([round(f, 3) for f in series])                    # 0, 1/6, ..., 1 linkers with A
mid = interpolate_descriptor(series, 0.4)
print(round(mid.total([("fg:A", "fg:A")]), 3))          # 1.8 A-A pairs at 40 % A
```

For a fixed number of linkers the exact weights are quadratic in the ratio,
so interpolation is approximate between grid points. For a large crystal,
`mode="independent"` gives the exact value at any ratio:

```python
crystal = cage.get_ensemble_descriptor(mode="independent",
                                       fractions={"A": 0.4, "B": 0.6}, by="label")
print(round(crystal.total([("fg:A", "fg:A")]), 3))      # 2.4 = 15 pairs x 0.4^2
```

## Defects

A missing linker is handled as a linker pattern whose sites all carry a
vacancy label, so the symmetry counting and the ensemble averages treat it
like any other pattern:

```python
print(cage.count_defect_isomers(1, groups="A"))         # 256 structures, one linker missing
defect = cage.get_ensemble_descriptor(groups="A", n_missing_linkers=1, by="label")
print(round(defect.total(), 3))                         # 10.0 pairs of the 5 remaining groups
```

The defect structures themselves have the missing building blocks deleted
and the exposed sites left open (the same for cages, MOFs and COFs):

```python
structures = cage.enumerate_defect_isomers(1, groups="", output_path="defects")
iso, atoms = structures[0]
print(len(structures), len(cage.to_ase()) - len(atoms))   # 1 structure, 10 atoms removed

print(cage.count_node_defect_isomers(1))                  # 1: the four nodes are equivalent
removed, no_node = cage.enumerate_node_defect_isomers(1, output_path="node_defects")[0]
print(len(cage.to_ase()) - len(no_node))                  # 86: one Zr6 node removed
```

Open metal sites created by a defect are found by `get_features()` on the
defect structure (see [Descriptors](descriptors.md)).

## Other weightings

Each symmetry-unique isomer once, or Boltzmann weights from an energy per
isomer, instead of every placement once:

```python
from cage_isomer_builder.utils import rgroup
from cage_isomer_builder.utils.ensemble import (
    boltzmann_weights, isomer_ensemble_descriptor, orbit_sizes)

group, layout, slots = cage.slot_group(), cage.slot_layout(), cage.slot_features()
isomers = rgroup.enumerate_rgroup_isomers(group, layout.n_slots, layout,
                                          linker_patterns={"A": 3, "B": 3})

per_isomer = isomer_ensemble_descriptor(slots, isomers, layout, by="label")   # each isomer once
by_placement = isomer_ensemble_descriptor(slots, isomers, layout,
                                          weights=orbit_sizes(group, isomers), by="label")
print(len(isomers), round(by_placement.total(), 3))     # 3424 15.0
# energies = [...]  one per isomer, e.g. from optimise(); then
# weights = boltzmann_weights(energies, temperature=298.15)
```

## Several kinds of linker

In a MOF with several linker types every argument can be given per type,
e.g. `linker_patterns={"C18H18N4": {"A": 1}, "C6H4": {"B": 1}}` or
`n_missing_linkers={"C6H4": 1}` (see
[Functionalise MOFs](functionalise-mofs.md)). Linkers of different types are
independent; linkers of one type are correlated through their fixed counts.
