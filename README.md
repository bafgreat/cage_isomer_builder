# Framework Isomer Builder

A comprehensive Python library for building, enumerating, and functionalising metal-organic cages and framework (MOF) isomers. It can also be used as a lightweight tool for generating host–guest docked complexes for finite cages.

## Main functionalities

- Build any of the 29 most common metal-organic cage topologies from a simple node and linker input
- Optimise cage structures with UFF4MOF using real bond orders, or with GFN-xTB, while preserving the shape of nodes and linkers through rigid-body relaxation
- Generate symmetry-unique isomers for finite cages using point-group methods, and for periodic MOFs using space-group methods based on spglib
- Functionalise cages and MOFs by attaching real fragments to one, several, or all active sites, with control over the distribution by ratio
- Dock guest molecules into finite cages, including mixed-guest systems and capacity checks
- Calculate cage-level statistics such as isomer counts, functional-group-to-functional-group distance distributions, and endo/exo classifications

## Installation

`cage_isomer_builder` requires Python `^3.11` (Python 3.11 or later,
but below Python 4.0). It can be installed directly from PyPI or by
cloning the GitHub repository.

### PyPI

```bash
pip install cage_isomer_builder
```

### GitHub

```bash
git clone https://github.com/bafgreat/cage_isomer_builder.git

cd cage_isomer_builder

pip install .
```

## Quickstart

### Build a cage from building units and enumerate isomers

You can import any of the cage builder classes
from `cage_isomer_builder.cage` to build a cage.
The example below builds a `Tri4Di6` cage.

The isomer enumeration matches all C–H bonds
in the linker (courtesy of mofstructure) and
enumerates all unique isomers that can be formed.
All unique isomerisation placeholders are labelled
with `At`.

```python
from cage_isomer_builder.cage import Tri4Di6CageBuilder

cage = Tri4Di6CageBuilder(
    node="tests/data/uio66_tri_node.xyz",
    linker="tests/data/bdc.xyz",
)

cage.build()
cage.optimise(rigid_sbus=True)          # UFF4MOF, node/linker shapes preserved
cage.save("cage.xyz")

isomers = cage.enumerate_isomers(output_path="isomers")
print(f"{len(isomers)} symmetry-unique isomers")
```

### Functionalising the enumerated isomers

All `At`-labelled sites in the enumerated isomers
can be functionalised using any functional-group moiety.
The only requirement is that the group must contain an
`X` dummy atom at the position to be functionalised.
See the example below.

```python
from ase.io import read
from cage_isomer_builder.utils.functionalise import functionalise_isomer_sites

isomer = read("isomers/0-4-10-12-16-20.xyz")
nh2 = read("tests/data/NH2.xyz")   # a small molecule with one "X" dummy atom

decorated = functionalise_isomer_sites(isomer, fragments=nh2)
```

### Host-guest docking

Host-guest complexes can be generated to study how isomerisation affects
adsorption.

```python
from ase.io import read
from cage_isomer_builder.utils.functionalise import (
    max_guests_in_host, place_guest_in_host,
)

host = read('Cage_molecule.xyz')
guest = read('guest_molecule.xyz')

n_max = max_guests_in_host(host, guest, seed=0)
complexes = place_guest_in_host(host, guest, n_guests=n_max, seed=0)
print(f"{len(complexes[0].guest_labels)} guests placed inside the cage")
```

### Statistics

Summary statistics can be calculated after isomer enumeration. These include
the number of unique isomers, functional-group pair counts, distance
distributions, and endo/exo classifications.

```python
stats = cage.get_statistics(n_kde_bins=200)
print(f"{stats.n_unique_isomers} unique isomers, "
      f"{stats.total_fg_pairs} FG-FG pairs "
      f"({stats.endo_endo_pairs} endo-endo)")
```

### Functionalise MOFs

Existing MOF structures can be loaded directly from a CIF file. The
`functionalise()` method finds the available C–H sites on the linkers and
replaces them with `At` anchor atoms for isomer enumeration. For large unit
cells, use Burnside's lemma to count the symmetry-unique isomers without
generating every structure.

```python
from cage_isomer_builder.cage import CageBuilder

mof = CageBuilder.load_from_file("tests/data/RUBTAK01.cif")

fg_anchors, fg_anchor_indices = mof.functionalise()
print(f"{len(fg_anchor_indices)} functional-group anchor sites detected")

n_unique = mof.count_unique_isomers_burnside()
print(f"{n_unique:.3g} symmetry-unique isomers")

first_few = mof.enumerate_isomers(output_path="mof_isomers", limit=10)
```

The generated files contain the same `At` placeholders used for cage
isomers, so they can be decorated with real fragments using
`functionalise_isomer_sites()` as shown above.

## Topologies

The `cage_isomer_builder.cage` classes can be used to build
the 29 most common topologies without any manual construction
of a topology graph. The table below lists the 29 cage topologies
and the topicity of their nodes and linkers, to make cage building
easy.

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
