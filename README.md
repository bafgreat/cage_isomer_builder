# Cage Isomer Builder

[![Docs](https://img.shields.io/badge/docs-mkdocs--material-1b263b)](https://bafgreat.github.io/cage_isomer_builder/)
[![License: MIT](https://img.shields.io/badge/license-MIT-415a77)](LICENSE)

A Python library for building, enumerating, and functionalising metal-organic cages and framework (MOF) isomers. It can also be used as a lightweight tool for generating host-guest docked complexes for finite cages.

Full documentation, including a complete API reference: **[docs](https://bafgreat.github.io/cage_isomer_builder/)**

## Main functionalities

- Build any of the 29 most common metal-organic cage topologies from a simple node and linker input
- Optimise cage structures with UFF4MOF using real bond orders, or with GFN-xTB, while preserving the shape of nodes and linkers through rigid-body relaxation
- Generate symmetry-unique isomers for finite cages using point-group methods, and for periodic MOFs using space-group methods based on spglib
- Find the point group of a finite cage from its atoms, so it works for any topology, including Ih and D5h cages
- Count and enumerate isomers with two or more different functional groups (R1, R2, ...)
- Functionalise cages and MOFs by attaching real fragments to one, several, or all active sites, with control over the distribution by ratio
- Dock guest molecules into finite cages, including mixed-guest systems and capacity checks
- Place guests at their lowest-energy sites, scored with DFT-D4 dispersion
- Calculate cage-level statistics such as isomer counts, functional-group-to-functional-group distance distributions, and endo/exo classifications
- Describe pores and guests by feature-site pair distributions: functional groups, aromatic rings (with normals), H-bond donors/acceptors on linkers and nodes, and open metal sites
- Average descriptors exactly over multivariate, partially functionalised and defective (missing linker/node) isomer ensembles, over any composition ratio, and over linker ring rotations; scale pores isotropically
- Describe a flexible guest over its conformers and score how well it matches a pore

## Installation

`cage_isomer_builder` requires Python `^3.11` (Python 3.11 or later,
but below Python 4.0). It can be installed directly from PyPI or by
cloning the GitHub repository.

### PyPI

```bash
pip install cage_isomer_builder
```

### Optional extras

tblite (GFN-xTB) and dftd4 (D4 dispersion) are optional, because they have
no Windows wheels. Without them the package still works: UFF4MOF optimises
without its Coulomb term, and guest sites are scored with UFF instead of D4.
Both print a warning.

```bash
pip install "cage_isomer_builder[all]"
```

See the [installation guide](https://bafgreat.github.io/cage_isomer_builder/installation/)
for Windows notes.

### GitHub

```bash
git clone https://github.com/bafgreat/cage_isomer_builder.git

cd cage_isomer_builder

pip install .
```

## Quickstart

Every structure used below ships with the package
(`cage_isomer_builder.example_data`), so the code runs as it is. The
[documentation](https://bafgreat.github.io/cage_isomer_builder/) has a guide
for each step and an example for every function.

### Build a pore and count its isomers

The tetrahedral pore of UiO-66, built as a Tri4Di6 cage from the UiO-66 Zr6
node and a 1,4-phenylene linker:

```python
from cage_isomer_builder import example_data
from cage_isomer_builder.cage import Tri4Di6CageBuilder

cage = Tri4Di6CageBuilder(
    node=example_data.path("uio66_tri_node"),   # X atoms mark the connection points
    linker=example_data.path("bdc"),
    scale_multiplier=0.225,                     # place the bulky node next to the linkers
)
cage.build()
cage.optimise(include_charges=False, logfile=None)   # UFF4MOF, rigid nodes and linkers

anchors, sites = cage.functionalise()    # every aromatic C-H becomes an R site
print(cage.point_group().name)           # Td
print(cage.count_unique_isomers_burnside())   # 176

isomers = cage.enumerate_isomers(output_path="isomers")
```

Optimise (or lower `scale_multiplier`) before `functionalise()`: STK places
bulky nodes away from their linkers, and `functionalise()` warns when the
building blocks are not bonded.

### Attach functional groups

Each written isomer marks its sites as R sites: `X` dummy atoms with an
`rgroup` number, listed in `atoms.info["r_sites"]` (e.g. `"352:R1 362:R1"`).

```python
from cage_isomer_builder.utils.functionalise import functionalise_isomer_sites
from cage_isomer_builder.utils.sites import read_isomer

isomer = read_isomer("isomers/" + "-".join(map(str, isomers[0])) + ".xyz")
decorated = functionalise_isomer_sites(isomer, example_data.read("NH2"))
print(decorated.get_chemical_formula())  # C120H138N6O128Zr24
```

### Several groups, ratios and defects

```python
print(cage.count_rgroup_isomers("AB"))                               # 124464
print(cage.count_rgroup_isomers(linker_patterns={"A": 3, "B": 3}))   # 3424: 50:50
print(cage.count_defect_isomers(1, groups="A"))                      # 256: one linker missing
```

### MOFs

```python
from cage_isomer_builder.cage import CageBuilder

mof = CageBuilder.load_from_file(example_data.path("RUBTAK01"), primitive=True)  # UiO-66
mof.functionalise()
print(mof.count_rgroup_isomers("AB"))    # 63528 (space group Fm-3m)

mixed = CageBuilder.load_from_file(example_data.path("ABIXEQ"))   # two kinds of linker
mixed.functionalise()
print(mixed.slot_layout().type_names)    # ('C18H18N4', 'C6H4')
print(mixed.count_rgroup_isomers(groups={"C18H18N4": "A", "C6H4": ""}))   # 4
```

### Descriptors

```python
print(cage.get_features().counts())      # {'acceptor': 32, 'fg': 24, 'pi': 6}
descriptor = cage.get_descriptor()       # distances and angles of all feature pairs

ensemble = cage.get_ensemble_descriptor(linker_patterns={("NH2",): 3, ("CH3",): 3},
                                        by="label", rotation_angles=range(0, 360, 30))
print(round(ensemble.total(), 3))        # 15.0 NH2/CH3 pairs, averaged over ring rotations
```

### Host-guest docking

```python
from cage_isomer_builder.utils.functionalise import max_guests_in_host, place_guest_in_host

host = cage.to_ase(hydrogens=True)       # the pore without R-site markers
methane = example_data.read("methane")
print(max_guests_in_host(host, methane, seed=0))     # 4 in the 7.5 A cavity
best = place_guest_in_host(host, methane, method="energy", n_samples=200, seed=0)[0]
print(best.atoms.info["interaction_energy"])         # eV (DFT-D4 + UFF repulsion)
```

## Topologies

The `cage_isomer_builder.cage` classes build the 29 most common
topologies without any manual construction of a topology graph. Each class
also builds with default inputs (a single `[Zr]` node and an amine linker of
the right topicity). The table lists the topicity of their nodes and
linkers.

| Topology | Node topicity | Linker topicity | Nodes × linkers |
| --- | --- | --- | --- |
| Tet2Di4 | Tetratopic (4) | Ditopic (2) | 2 × 4 |
| Tet3-3Di3 | Tetratopic (4) | Ditopic (2) | 3 × 6 |
| Tet4-4Di8 | Tetratopic (4) | Ditopic (2) | 4 × 8 |
| Tet5Di10 | Tetratopic (4) | Ditopic (2) | 5 × 10 |
| Tet6Di12 | Tetratopic (4) | Ditopic (2) | 6 × 12 |
| Tet8Di16 | Tetratopic (4) | Ditopic (2) | 8 × 16 |
| Tet16Di32 (M12L24) | Tetratopic (4) | Ditopic (2) | 12 × 24 |
| Tet24Di48 (M24L48) | Tetratopic (4) | Ditopic (2) | 24 × 48 |
| Tri2Di3 | Tritopic (3) | Ditopic (2) | 2 × 3 |
| Tri4Di6 | Tritopic (3) | Ditopic (2) | 4 × 6 |
| Tri4-2Di6 | Tritopic (3) | Ditopic (2) | 4 × 6 |
| Tri6Di9 | Tritopic (3) | Ditopic (2) | 6 × 9 |
| Tri8Di12 | Tritopic (3) | Ditopic (2) | 8 × 12 |
| Tri20Di30 | Tritopic (3) | Ditopic (2) | 20 × 30 |
| Tet6Tri8 | Tetratopic (4) | Tritopic (3) | 6 × 8 |
| Tri4Tri4 | Tritopic (3) | Tritopic (3) | 4 × 4 |
| Tri1Tri1 | Tritopic (3) | Tritopic (3) | 1 × 1 |
| Tri2²Tri2 | Tritopic (3) | Tritopic (3) | 2 × 2 |
| M2L4Lantern | Tetratopic (4) | Ditopic (2) | 2 × 4 |
| M3L6 | Tetratopic (4) | Ditopic (2) | 3 × 6 |
| M4L6TetrahedronSpacer | Tritopic (3) | Ditopic (2) | 4 × 6 |
| M4L8 | Tetratopic (4) | Ditopic (2) | 4 × 8 |
| M4L82 | Tetratopic (4) | Ditopic (2) | 4 × 8 |
| M4L8x | Tetratopic (4) | Ditopic (2) | 4 × 8 |
| M6L12Cube | Tetratopic (4) | Ditopic (2) | 6 × 12 |
| M8L6Cube | Tritopic (3) | Tetratopic (4) | 8 × 6 |
| M9L18 | Tetratopic (4) | Ditopic (2) | 9 × 18 |
| TenPlusTwenty | Tetratopic (4) | Ditopic (2) | 10 × 20 |
| TwelvePlusThirty | Pentatopic (5) | Ditopic (2) | 12 × 30 |

Choose the builder whose node and linker topicities match the input building
units. Input structures can be supplied as file paths or as ASE `Atoms`
objects; generated structures can be saved in any format supported by ASE.

## License

This project is released under the MIT License.
