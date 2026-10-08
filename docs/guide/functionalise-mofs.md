---
description: >-
  Load a MOF from a CIF file, reduce it to its primitive cell, count and
  enumerate its functionalised isomers with the full space group, and handle
  MOFs with more than one kind of linker.
---

# Functionalise MOFs

Any structure file can enter the same workflow as a built cage through
`CageBuilder.load_from_file`. For a periodic structure the symmetry is its
full space group (spglib), translations included.

```python
from cage_isomer_builder import example_data
from cage_isomer_builder.cage import CageBuilder

mof = CageBuilder.load_from_file(example_data.path("RUBTAK01"))   # UiO-66, conventional cell
anchors, sites = mof.functionalise()
print(len(mof.to_ase()), len(sites))             # 440 atoms, 96 sites on 24 linkers
print(mof.count_unique_isomers_burnside())       # 1466084546944
first = mof.enumerate_isomers(output_path="mof_isomers", limit=10)   # only the first ten
```

Counting uses Burnside's lemma and takes about a second even for counts far
too large to enumerate; `limit` stops an enumeration early.

## Use the primitive cell

A CIF often holds several copies of the smallest repeating cell; a
face-centred cell holds four. Each copy multiplies the number of isomers, so
a full enumeration that is impossible for the CIF cell can be easy for the
primitive cell:

```python
mof = CageBuilder.load_from_file(example_data.path("RUBTAK01"), primitive=True)
mof.functionalise()
print(len(mof.to_ase()))                         # 110
print(mof.count_rgroup_isomers("AB"))            # 63528 (4.1e23 for the CIF cell)
isomers = mof.enumerate_rgroup_isomers("AB", generate_files=False)   # all 63528
```

Atom positions are unchanged; only the cell is reduced. Every isomer then
repeats in every primitive cell, so only patterns with the period of that
cell are covered; patterns that change from cell to cell need a larger
cell. `symprec` (default `1e-2`) sets spglib's tolerance.

## More groups per linker

```python
print(mof.count_unique_isomers_burnside(active_per_linker=2))   # 1109: two groups on every linker
```

`count_rgroup_isomers` and `enumerate_rgroup_isomers` work the same way as
for cages (see [Functionalise isomers](functionalising-isomers.md)), and the
written files decorate the same way with `functionalise_isomer_sites`.

## MOFs with more than one kind of linker

A linker is an organic unit bonded to the framework at two or more points,
so terminal ligands (bound to one metal) and guests are not linkers. Every
linker type is kept, and only aromatic C-H become sites
(`functionalise(aromatic_only=False)` also marks alkyl C-H):

```python
mof = CageBuilder.load_from_file(example_data.path("ABIXEQ"))   # a Co MOF, two linkers
mof.functionalise()
layout = mof.slot_layout()
print(layout.type_names, layout.sizes)           # ('C18H18N4', 'C6H4') (8, 4)
```

Groups can be given per linker type; a plain string applies to every type:

```python
print(mof.count_rgroup_isomers(groups="A"))                               # 16
print(mof.count_rgroup_isomers(groups={"C18H18N4": "AB", "C6H4": "A"}))   # 112
print(mof.count_rgroup_isomers(groups={"C18H18N4": "A", "C6H4": ""}))     # 4: only one type
print(mof.count_rgroup_isomers(linker_patterns={"C18H18N4": {"A": 1}, "C6H4": {"B": 1}}))   # 16
print(mof.count_defect_isomers({"C6H4": 1}, groups="A"))                  # 4: phenylene missing
```

The same per-type arguments work for `enumerate_rgroup_isomers`,
`get_ensemble_descriptor` and the defect methods. A symmetry operation may
only map a linker onto one of the same type, and this is checked.

Terminal ligands are recognised and left out:

```python
mof = CageBuilder.load_from_file(example_data.path("ABINOQ"))   # Zn, bipyridine + thiolates
mof.functionalise()
print(mof.slot_layout().type_names)              # ('C10H8N2',): the thiolates are not linkers
```
