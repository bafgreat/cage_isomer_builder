---
description: >-
  Attach real functional groups to the R sites of enumerated isomers, mix
  groups by ratio, and count and enumerate isomers with several different
  groups (R1, R2, ...).
---

# Functionalise isomers

This page uses the UiO-66 tetrahedral pore from
[Build & enumerate](build-and-enumerate.md):

```python
from cage_isomer_builder import example_data
from cage_isomer_builder.cage import Tri4Di6CageBuilder

cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
                          linker=example_data.path("bdc"), scale_multiplier=0.225)
cage.build()
cage.optimise(include_charges=False, logfile=None)
cage.functionalise()
```

## Fragments

A fragment is a small molecule with one `X` dummy atom marking where it
bonds to the cage. NH2 ships with the package; others are easy to make:

```python
from ase import Atoms

nh2 = example_data.read("NH2")          # N, H, H and an X
oh = Atoms("XOH", positions=[[-1.43, 0.0, 0.0], [0.0, 0.0, 0.0], [0.32, 0.91, 0.0]])
```

## Attaching one kind of group

Every R site of an isomer gets the fragment:

```python
from cage_isomer_builder.utils.functionalise import functionalise_isomer_sites
from cage_isomer_builder.utils.sites import read_isomer

isomers = cage.enumerate_isomers(output_path="isomers")
name = "-".join(str(i) for i in isomers[0])
isomer = read_isomer(f"isomers/{name}.xyz")

amino = functionalise_isomer_sites(isomer, nh2)
print(amino.get_chemical_formula())      # C120H138N6O128Zr24: six NH2
```

Several fragment types can be spread over the sites by ratio (which site
gets which is random; `seed` makes it reproducible):

```python
mixed = functionalise_isomer_sites(isomer, [nh2, oh], ratios=[2, 1], seed=0)
print(mixed.get_chemical_formula())      # C120H136N4O130Zr24: four NH2, two OH
```

An isomer keeps the cage's atom order, so passing the cage's bond matrix
also gives the bonds of the decorated structure (for a later force-field run
on the new groups only):

```python
result = functionalise_isomer_sites(isomer, nh2, host_bond_matrix=cage.get_bond_matrix())
print(len(result.free_atom_indices), len(result.anchor_pairs))   # 18 6: new atoms, new bonds
print(result.bond_matrix.shape)                                  # (416, 416)
```

## Several different groups (R1, R2, ...)

`count_rgroup_isomers` and `enumerate_rgroup_isomers` place two or more
different groups at the same time. Each group has a name.

### The same groups on every linker

| `groups` | Meaning |
| --- | --- |
| `"A"` | one group per linker (same as `enumerate_isomers`) |
| `"AB"` | one A and one B on every linker |
| `"AAB"` | two A and one B on every linker |
| `["NH2", "OH"]` | one NH2 and one OH on every linker |

A string is read one capital letter per group; give longer names as a list
or tuple.

```python
print(cage.count_rgroup_isomers("AB"))                      # 124464
first = cage.enumerate_rgroup_isomers(["NH2", "OH"], output_path="rgroup", limit=5)
print(first[0].to_dict())       # {'NH2': [2, 6, 10, 14, 18, 22], 'OH': [3, 7, 11, 15, 19, 23]}
```

### Different groups on different linkers

`linker_patterns` says how many linkers carry each set of groups; the counts
must add up to the number of linkers, and `""` is a linker with no group:

```python
print(cage.count_rgroup_isomers(linker_patterns={"A": 3, "B": 3}))   # 3424: 50:50
print(cage.count_rgroup_isomers(linker_patterns={"AB": 2, "": 4}))   # two linkers with A and B
print(cage.count_rgroup_isomers(linker_patterns={("NH2",): 2, ("CH3",): 4}))   # named groups
```

Counts use Burnside's lemma, so they are fast even when there are far too
many isomers to write out; a full enumeration is always checked against
them. Groups with different names are never swapped by symmetry, and groups
are taken as achiral (mirror images are the same isomer).

### Attaching the groups

Each written file records which group goes on each site (see
[R sites & file formats](r-sites.md)). Pass the fragments by group name:

```python
isomer = read_isomer(f"rgroup/{first[0].name}.xyz")
print(isomer.info["rgroup_labels"])      # NH2 OH  (R1 = NH2, R2 = OH)
decorated = functionalise_isomer_sites(isomer, {"NH2": nh2, "OH": oh})
print(decorated.get_chemical_formula())  # C120H138N6O134Zr24
```
