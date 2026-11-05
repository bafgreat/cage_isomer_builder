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

### Which linkers meet at each vertex

The sites are numbered linker by linker: with 4 sites per linker, sites 0-3
are on linker 0, sites 4-7 on linker 1, and so on. Every isomer file records
which sites each linker owns and which linkers meet at each vertex (node).
The numbering is the same in every isomer of one cage:

```python
from cage_isomer_builder.utils.sites import read_isomer

info = read_isomer("isomers/0-4-8-12-16-20.xyz").info
print(info["linker_sites"])     # 0:0-3 1:4-7 2:8-11 3:12-15 4:16-19 5:20-23
print(info["vertex_linkers"])   # 0:0,1,2 1:0,3,4 2:1,3,5 3:2,4,5
```

So vertex 2 joins linkers 1, 3 and 5, which own sites 4-7, 12-15 and 20-23.
The vertex numbers are those of `missing_nodes` in node-defect isomers.

### Vertex names

An isomer can also be named by what each vertex sees of the groups on its
linkers. Look at a linker from the pore centre with the vertex at the
bottom: it reads **1** if its group is next to this vertex on the left,
**4** if on the right, and **0** if it has no group or the group is next to
the other vertex. A vertex's code lists its three linkers clockwise as seen
from the pore centre, starting at its lowest-numbered linker, and the
vertex name joins the codes of vertices 0, 1, 2, 3:

```python
scheme = cage.vertex_scheme()
iso = [0, 4, 9, 13, 16, 22]
print(scheme.name(iso))      # 000-100-141-401
print(scheme.labels(iso))    # ['000', '100', '114', '140']
print(scheme.vertex_linkers) # ((0, 1, 2), (0, 4, 3), (1, 3, 5), (2, 5, 4))
```

`vertex_linkers` is the order in which each vertex reads its linkers: vertex 1
reads linker 0, then 4, then 3.

The vertex name is unique for every isomer and converts back exactly, so
an isomer can be picked by either name:

```python
print(scheme.resolve("000-100-141-401"))   # [0, 4, 9, 13, 16, 22]
print(scheme.resolve("0-4-9-13-16-22"))    # [0, 4, 9, 13, 16, 22]
```

`labels` gives each vertex's *environment*: its code turned to the rotation
that sorts first with 1 < 4 < 0, so `141` and `411` are both `114`. The 11
environments of a 3-connected vertex are `000 100 400 110 440 140 104 111
114 144 444`. A label says which environment a vertex has but not which of
its linkers carries which group, so different isomers can share all four
labels; only the vertex name identifies one isomer.

Isomer files can be named either way. Every file records both names and the
labels, so a file found by one name gives the other:

```python
isomers = cage.enumerate_isomers(output_path="by_vertex", naming="vertex")
info = read_isomer("by_vertex/000-100-110-111.xyz").info
print(info["isomer"])          # 0-4-8-12-16-20
print(info["vertex_labels"])   # 000-100-110-111
```

#### From the command line

Every isomer file also stores the cage's vertex scheme, so the
`cage-vertex-name` command converts names without rebuilding the cage. Give
it isomer files to write a copy under the other name (vertex name for a
site-named file, site name for a vertex-named one):

```bash
cage-vertex-name isomers/0-4-9-13-16-22.xyz          # -> isomers/000-100-141-401.xyz
cage-vertex-name by_vertex/000-100-141-401.xyz       # -> by_vertex/0-4-9-13-16-22.xyz
cage-vertex-name isomers/*.xyz --to vertex -o renamed   # a whole folder, by vertex name
```

`--to vertex` (or `--to sites`) fixes the name to give, which is safer for a
folder that mixes both; `-o` writes the copies to another folder, `--move`
renames instead of copying, and `--force` overwrites. With `--ref`, any
isomer file of the cage converts names either way and prints them, writing
nothing:

```bash
cage-vertex-name --ref isomers/0-4-8-12-16-20.xyz 0-4-9-13-16-22 000-100-141-401
# 0-4-9-13-16-22 -> 000-100-141-401   labels 000-100-114-140
# 000-100-141-401 -> 0-4-9-13-16-22   labels 000-100-114-140
```

The same command from Python (it returns the exit status, 0 on success):

```python
from cage_isomer_builder.cli import vertex_name_main

vertex_name_main(["isomers/0-4-9-13-16-22.xyz"])   # writes isomers/000-100-141-401.xyz
vertex_name_main(["--ref", "isomers/0-4-8-12-16-20.xyz", "000-100-141-401"])
```

The enumeration counts mirror images as one isomer, so an isomer stands for
its mirror image too. That one has 1 and 4 swapped and every code read the
other way round (`114` becomes `144`, `110` becomes `440`; `140` and `104` stay).

For large structures use `limit`, or count only (see
[Functionalise MOFs](functionalise-mofs.md)).

The complete list of builder classes is on the [Topologies](../topologies.md)
page; every method is in the [API reference](../reference/cage.md).
