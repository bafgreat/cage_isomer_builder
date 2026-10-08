---
description: >-
  Build a metal-organic cage from a node and a linker, optimise it, find its
  functional-group sites and point group, and count and write its
  symmetry-unique isomers.
---

# Build & enumerate isomers

## Building a cage

Pick the builder class whose node and linker topicity match your building
blocks (see [Topologies](../topologies.md)). Node and linker can be given as

* a **structure file** or `ase.Atoms` with `X` dummy atoms marking the
  connection points (as in the shipped `uio66_tri_node` and `bdc`),
* a **SMILES** string with dummy atoms (`*` or `[*]`) marking the connection
  points, e.g. `"[*]c1ccc([*])cc1"`,
* a **SMILES** string whose reactive groups STK recognises (amines,
  aldehydes, ...; e.g. `"Nc1ccc(N)cc1"`), or a single metal such as
  `"[Zr]"`.

```python
from cage_isomer_builder import example_data
from cage_isomer_builder.cage import Tet2Di4CageBuilder, Tri4Di6CageBuilder

lantern = Tet2Di4CageBuilder(linker="[*]c1ccc([*])cc1")   # default node: one [Zr]
lantern.build()
print(lantern.to_ase().get_chemical_formula())           # C24H16Zr2

cage = Tri4Di6CageBuilder(
    node=example_data.path("uio66_tri_node"),
    linker=example_data.path("bdc"),
    scale_multiplier=0.225,
)
cage.build()
```

### Bulky nodes: `scale_multiplier` and `optimise()`

STK sizes the whole cage from the largest building block, so a bulky node
such as a Zr6 cluster next to a short linker is first placed far from its
linkers. Either lower `scale_multiplier` (as above) or optimise before going
further. `functionalise()` warns if the building blocks are not bonded,
because symmetry, isomer counts and distances of such a geometry describe a
structure that does not exist.

```python
cage.optimise(include_charges=False, logfile=None)   # see the Optimisation page
cage.save("uio66_pore.xyz")
```

## Functional-group sites

`functionalise()` turns every aromatic C-H of every linker into an R site
(an `X` dummy atom, see [R sites & file formats](r-sites.md)):

```python
anchors, sites = cage.functionalise()
print(len(sites))                          # 24
print(cage.slot_layout().sizes)            # (4, 4, 4, 4, 4, 4): 6 linkers x 4 sites
```

## Symmetry

For a finite cage the point group is found from the atoms of the structure.
This works for any topology and orientation, including 5-fold axes (Ih, D5h
cages) that spglib cannot find. Periodic MOFs use their space group
(spglib).

```python
pg = cage.point_group()
print(pg.name, pg.order)        # Td 24
print(pg.max_deviation < 0.5)   # True: how far (A) the structure is from perfect symmetry
```

| Setting (on the builder) | Default | Meaning |
| --- | --- | --- |
| `symmetry_method` | `"detect"` | `"detect"` finds the group from the atoms. `"topology"` uses a hand-written table for the topology, which assumes the standard orientation. |
| `symmetry_reference` | `"auto"` | Atoms used: `"auto"` (metals, or heavy atoms if fewer than three metals off a line), `"metals"`, `"heavy"` or `"all"`. |
| `symmetry_tol` | `0.5` | Matching tolerance (A) for those atoms. |
| `symmetry_anchor_tol` | `1.0` | Matching tolerance (A) for the functional-group sites. |

If a tolerance is too tight or too loose for a distorted structure, the
operations found do not form a group and you get a `ValueError` instead of
a wrong count. Optimise the cage or adjust `symmetry_tol`.

## Counting and writing isomers

An isomer here places one functional group on every linker (several groups:
see [Functionalise isomers](functionalising-isomers.md)). The exact count
comes from Burnside's lemma, without generating anything:

```python
print(cage.count_unique_isomers_burnside())                      # 176
print(cage.count_unique_isomers_burnside(active_per_linker=2))   # 2013: two groups per linker
```

`enumerate_isomers` writes one structure per symmetry-unique isomer; a full
enumeration always matches the Burnside count. Each isomer is the list of
active site indices, and the file is named after it:

```python
isomers = cage.enumerate_isomers(output_path="isomers")
print(len(isomers), isomers[0])            # 176 [0, 4, 8, 12, 16, 20]
first_three = cage.enumerate_isomers(generate_files=False, limit=3)   # stop early
```

For large structures use `limit`, or count only (see
[Functionalise MOFs](functionalise-mofs.md)).

The complete list of builder classes is on the [Topologies](../topologies.md)
page; every method is in the [API reference](../reference/cage.md).
