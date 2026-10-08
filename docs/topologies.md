---
description: >-
  The 29 metal-organic cage topologies supported: node and
  linker topicity, and which builder class to use for each.
---

# Topologies

The `cage_isomer_builder.cage` classes can be used to build the 29 most
common topologies without any manual construction of a topology graph. The
table below lists the 29 cage topologies and the topicity of their nodes
and linkers, to make cage building easy.

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

Every builder class shares the same interface as
[`CageBuilder`](reference/cage.md); only the topology graph and point-group
transformation library differ. See `cage_isomer_builder.cage.library` in the
[API reference](reference/cage.md) for the full class list.
