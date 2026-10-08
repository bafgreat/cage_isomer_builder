---
description: >-
  Describe a pore by the distances and angles between its chemical feature
  sites (functional groups, aromatic rings, H-bond donors and acceptors,
  open metal sites), including additivity, flexibility and pore scaling.
---

# Descriptors

A descriptor is the distribution of distances (and angles) between the
chemical *feature sites* of a pore or a molecule. The same machinery
describes a static pore, an ensemble of isomers
([Multivariate pores & defects](multivariate.md)), a flexible framework, and
a guest over its conformers ([Guests & matching](matching.md)), so host and
guest can be compared directly.

```python
from cage_isomer_builder import example_data
from cage_isomer_builder.cage import Tri4Di6CageBuilder

cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
                          linker=example_data.path("bdc"), scale_multiplier=0.225)
cage.build()
cage.optimise(include_charges=False, logfile=None)
cage.functionalise()
```

## Feature sites

```python
feats = cage.get_features()
print(feats.counts())        # {'acceptor': 32, 'fg': 24, 'pi': 6}
```

| kind | position | vector |
| --- | --- | --- |
| `fg` | functional-group site (R site) | ring C to site |
| `pi` | aromatic ring centroid | ring normal (no sign) |
| `donor` | H of an O-H or N-H | heavy atom to H |
| `acceptor` | O or N with a free lone pair | lone-pair direction |
| `open_metal` | metal with an incomplete coordination sphere | into the empty site |

Labels refine the kinds, e.g. `mu3-O` (an acceptor bonded to three metals)
or `mu3-OH` (a donor) on a Zr6 node. This node file has no H on its mu3-O.
Open metal sites come from mofstructure's geometric detector (the CoRE MOF
method): a defect-free UiO-66 has none, a missing linker or a bare Cu
paddlewheel has them. `get_features(open_metal_method="coordination")` uses
a simpler rule instead (fewer ligands than `expected_cn={"Cu": 5}` or than
the most highly coordinated metal of that element). The detector works on
any `ase.Atoms`, guests included:

```python
from cage_isomer_builder.utils.features import detect_features

print(detect_features(example_data.read("ibuprofen")).counts())
# {'acceptor': 2, 'donor': 1, 'pi': 1}
```

The rules (bond perception, aromaticity, which N count as acceptors, ...)
are listed in the [Features reference](../reference/features.md). A nearly
planar aniline-type N may or may not count as an acceptor, depending on how
pyramidal its geometry is.

## Pair descriptor

```python
import numpy as np

desc = cage.get_descriptor()               # every pair of feature sites
print(desc.summary()[("fg", "pi")])        # (120, 120.0): (pairs, weight)

r = np.linspace(0, 30, 1501)
rho = desc.kde(r, keys=[("acceptor", "pi")], bandwidth=0.1)   # pairs per Angstrom
print(round(float(np.trapezoid(rho, r))))  # 192 = number of acceptor-ring pairs
```

Pairs inside one building block (two sites on one ring) are left out. By
default pair types are feature kinds; `by="label"` uses the finer labels.
For periodic structures, pairs are counted per unit cell over every image up
to `max_distance` (default 20 A), so the result does not depend on the
choice of cell.

### Distance and angle

The angle between the two features' vectors adds direction, which matters
for pi stacking and open metal sites. Angles involving a ring normal are
folded into 0-90 degrees.

```python
d_grid, a_grid = np.linspace(2, 25, 231), np.linspace(0, 90, 91)
surface = desc.kde2d(d_grid, a_grid, keys=[("pi", "pi")], bandwidth=0.2, angle_bandwidth=5)
print(surface.shape)                       # (231, 91): pairs per Angstrom per degree
```

## Partial distributions are additive with their cross terms

The full descriptor of feature sets A and B together is exactly
`D(A) + D(B) + D(A, B)`, where `D(A, B)` holds the pairs with one site from
each. Summing partial distributions alone misses every cross pair:

```python
from cage_isomer_builder.utils.distributions import additivity_residual, assemble

parts = [feats.select(kinds="fg"), feats.select(kinds="pi"), feats.select(kinds="acceptor")]
full = assemble(parts)                     # partials + cross terms
residual, missing = additivity_residual(parts, r)
print(residual < 1e-9, missing)            # True 1080.0 pairs missing without cross terms
```

## Flexibility: ring rotation

Each linker ring can turn about the axis through its two backbone atoms;
every bond is preserved and the sites turn with the ring:

```python
print(len(cage.rotors()))                  # 6: one phenylene per linker

flexible = cage.get_ensemble_descriptor(groups="A", rotation_angles=range(0, 360, 30))
print(round(flexible.total(), 6))          # 15.0 pairs of the 6 groups, averaged over angles
```

Rings turn independently. `angle_weights` (one per angle, e.g. Boltzmann
weights from a torsion scan, see `ensemble.boltzmann_weights`) replace the
default uniform average. Endo/exo classes are re-evaluated at every angle.

## Pore expansion and contraction

```python
bigger = cage.scaled(1.05)                 # every building block moved out by 5 %
pairs, distances, slopes = cage.distance_scaling((0.95, 1.0, 1.05))
print(distances.shape)                     # (3, 276): each site pair at each factor
print(bool((slopes >= -1e-9).all()))       # True: no distance shrinks as the pore grows
```

`slopes` is d(distance)/d(factor) for every site pair: zero for two sites on
one linker, close to the distance between the two linkers' centres
otherwise.
