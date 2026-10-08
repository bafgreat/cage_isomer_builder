---
description: >-
  The whole cage_isomer_builder workflow on one page: build a UiO-66 pore,
  optimise it, count and write its isomers, attach functional groups,
  compute descriptors and dock a guest.
---

# Quick start

This page runs the complete workflow on the tetrahedral pore of UiO-66,
built as a Tri4Di6 cage from the UiO-66 Zr6 node and a 1,4-phenylene
linker. Every structure used ships with the package
(`cage_isomer_builder.example_data`), so the code runs as it is. Each step
links to the page that explains it in full.

## 1. Build and optimise the pore

```python
from cage_isomer_builder import example_data
from cage_isomer_builder.cage import Tri4Di6CageBuilder

cage = Tri4Di6CageBuilder(
    node=example_data.path("uio66_tri_node"),   # Zr6 node, three X connection points
    linker=example_data.path("bdc"),            # 1,4-phenylene, two X connection points
    scale_multiplier=0.225,                     # place the bulky node close to the linkers
)
cage.build()
cage.optimise(include_charges=False, logfile=None)   # UFF4MOF, nodes and linkers kept rigid
cage.save("uio66_pore.xyz")
```

See [Build & enumerate](build-and-enumerate.md) and
[Optimisation](optimisation.md).

## 2. Find the functional-group sites and count isomers

```python
anchors, sites = cage.functionalise()      # every aromatic C-H becomes an R site
print(len(sites))                          # 24: four on each of the six linkers
print(cage.point_group().name)             # Td
print(cage.count_unique_isomers_burnside())   # 176 symmetry-unique isomers, one group per linker
```

## 3. Write the isomers and attach a real group

```python
from cage_isomer_builder.utils.functionalise import functionalise_isomer_sites
from cage_isomer_builder.utils.sites import read_isomer

isomers = cage.enumerate_isomers(output_path="isomers")      # 176 files
isomer = read_isomer(f"isomers/{'-'.join(map(str, isomers[0]))}.xyz")
print(isomer.info["r_sites"])            # {atom index: 'R1', ...}

decorated = functionalise_isomer_sites(isomer, example_data.read("NH2"))
print(decorated.get_chemical_formula())  # six NH2 groups attached
```

See [Functionalise isomers](functionalising-isomers.md) and
[R sites & file formats](r-sites.md).

## 4. Several groups, ratios and defects

```python
print(cage.count_rgroup_isomers(linker_patterns={"A": 3, "B": 3}))   # 3424 (50:50)
print(cage.count_defect_isomers(1, groups="A"))                      # 256 (one linker missing)
```

See [Multivariate pores & defects](multivariate.md).

## 5. Descriptors

```python
desc = cage.get_descriptor()                 # pairs of all feature sites
print(desc.summary()[("fg", "pi")])          # (120, 120.0): FG site - ring pairs

# 50:50 NH2/CH3 linkers; named groups are tuples (a string means one letter per group)
ensemble = cage.get_ensemble_descriptor(linker_patterns={("NH2",): 3, ("CH3",): 3}, by="label")
print(round(ensemble.total(), 3))            # 15.0 group-group pairs per pore (6 groups)
```

See [Descriptors](descriptors.md).

## 6. Dock a guest

```python
from cage_isomer_builder.utils.functionalise import max_guests_in_host, place_guest_in_host

host = cage.to_ase(hydrogens=True)          # the pore with H back on every site
methane = example_data.read("methane")
print(max_guests_in_host(host, methane, seed=0))     # 4 methane fit in the 7.5 A cavity
best = place_guest_in_host(host, methane, method="energy", n_samples=200, seed=0)[0]
print(best.atoms.info["interaction_energy"])         # eV, negative = bound
```

See [Host-guest docking](host-guest-docking.md) and
[Guests & matching](matching.md).
