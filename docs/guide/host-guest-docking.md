---
description: >-
  Find a cage's cavity and windows, check which guests fit, and place guests
  at random or at their lowest-energy sites (DFT-D4 dispersion), including
  mixed guests.
---

# Host-guest docking

```python
from cage_isomer_builder import example_data
from cage_isomer_builder.cage import Tri4Di6CageBuilder

cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
                          linker=example_data.path("bdc"), scale_multiplier=0.225)
cage.build()
cage.optimise(include_charges=False, logfile=None)
host = cage.to_ase()                    # the UiO-66 tetrahedral pore
methane = example_data.read("methane")
```

Docking needs real atoms: if the cage has been functionalised, attach
groups first or use `cage.to_ase(hydrogens=True)`. R-site markers (`X`)
are refused.

## Cavity and windows

The cavity is the largest sphere that fits inside the host (clearance to the
nearest atom surface, van der Waals radii); a window is a straight opening
out of it. Both come from `cage_isomer_builder.utils.cavity`:

```python
from cage_isomer_builder.utils.cavity import largest_window, pore_sphere

centre, diameter = pore_sphere(host)
print(round(diameter, 1))                      # 7.5 A
print(round(largest_window(host, centre), 1))  # 4.1 A
```

A guest whose narrowest cross-section is wider than the largest window
cannot get in, and is refused before any placement is tried:

```python
from cage_isomer_builder.utils.functionalise import max_guests_in_host

try:
    max_guests_in_host(host, example_data.read("ibuprofen"))
except ValueError as error:
    print(error)    # ... narrowest cross-section (5.83 A) is larger than ... window (4.10 A) ...
```

## Random placement

Guests are packed into the cavity by Random Sequential Adsorption: random
positions and orientations are accepted when no two atoms (host-guest or
guest-guest) come closer than `overlap_tolerance` (default 0.75) times the
sum of their van der Waals radii, until the packing jams.

```python
from cage_isomer_builder.utils.functionalise import place_guest_in_host

print(max_guests_in_host(host, methane, seed=0))    # 4
complexes = place_guest_in_host(host, methane, n_guests=2, n_complexes=3, seed=0,
                                host_bond_matrix=cage.get_bond_matrix())
first = complexes[0]
print(len(complexes), first.guest_labels)            # 3 [0, 0]
```

Each complex is a `HostGuestComplex` named tuple `(atoms, bond_matrix,
guest_atom_indices, guest_labels)`. Its bond matrix is the host's bonds plus
each guest's own bonds; no bond is added between host and guest. Pass
`host_bond_matrix` (as above) to use the cage's exact bonds instead of
perceiving them from the geometry. Asking for more guests than fit caps the
number, with a warning.

## Lowest-energy sites

`method="energy"` places each guest at its lowest-energy site, scored with
the DFT-D4 dispersion energy (needs the optional dftd4 package, else UFF
Lennard-Jones with a warning) plus the repulsive part of UFF's van der
Waals term:

```python
best = place_guest_in_host(host, methane, n_guests=1, n_complexes=2, method="energy",
                           n_samples=300, seed=0)
for c in best:
    print(c.atoms.info["dispersion_model"], round(c.atoms.info["interaction_energy"], 3))
# D4(pbe)+UFF repulsion -0.111   (eV; the complexes come lowest energy first)
```

1. For each guest, `n_samples` random positions in the cavity are tried in
   24 orientations each.
2. The best pose at each of 30 different sites is minimised as a rigid body.
3. Guests are added one at a time, each at the lowest-energy site left, so
   guest-guest dispersion counts too.
4. Each complex is rescored with a full dftd4 calculation (three-body term
   included) plus the repulsion.

| Option | Default | Meaning |
| --- | --- | --- |
| `method` | `"rsa"` | `"rsa"` random placement, `"energy"` lowest-energy sites |
| `n_samples` | `1000` | random positions tried per guest (energy method) |
| `contact_scale` | `0.75` | smallest allowed distance, as a fraction of the sum of van der Waals radii (energy method) |
| `dispersion` | `"auto"` | D4 (PBE) if dftd4 is installed, else UFF; `"uff"`; a D4 method name; or a dict of D4 damping parameters |
| `overlap_tolerance` | `0.75` | hard-core contact for random placement, fraction of van der Waals radii sums |

## Mixed guests

```python
from ase.build import molecule

mixed = place_guest_in_host(host, [methane, molecule("CO2")], ratios=[1, 1],
                            n_guests=2, seed=0)
print(sorted(mixed[0].guest_labels))     # [0, 1]: one methane, one CO2
```

## Writing a complex

```python
from cage_isomer_builder.utils.read_write import write_host_guest_complex

write_host_guest_complex(first, "complex.gin")   # GULP, with the complex's bonds
write_host_guest_complex(first, "complex.xyz")   # any ASE format
```
