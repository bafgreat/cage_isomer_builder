---
description: >-
  Dock guest molecules into a metal-organic cage cavity in Python via
  Random Sequential Adsorption, including mixed-guest systems and
  maximum-capacity checks.
---

# Host-guest docking

Host-guest complexes can be generated to study how isomerisation affects
adsorption. Guests are packed into the cage cavity via Random Sequential
Adsorption (RSA): random position/orientation trials are accepted whenever
they don't overlap the host or an already-placed guest, until the packing
jams. See [`_build_guest_packing`](../reference/functionalise.md) for the
algorithm and its tradeoff against a slower, denser optimisation-based
packer such as PACKMOL.

```python
from ase.io import read
from cage_isomer_builder.utils.functionalise import (
    max_guests_in_host, place_guest_in_host,
)

host = read("cage.xyz")
guest = read("guest_molecule.xyz")

n_max = max_guests_in_host(host, guest)
print(f"Maximum number of guests that can fit inside the cage: {n_max}")

complexes = place_guest_in_host(host, guest, n_guests=n_max, seed=0)
print(f"{len(complexes[0].guest_labels)} guests placed inside the cage")
```

Every element of `complexes` is a `HostGuestComplex` - a named tuple of
`(atoms, bond_matrix, guest_atom_indices, guest_labels)`. `bond_matrix`
carries the host's own bonds (auto-derived from geometry via
`gulp_setup.mmanalysis.analyze_mm` unless you pass `host_bond_matrix`
yourself) merged with each placed guest's internal bonds - no bond is ever
added between host and guest, or between two guest copies, since
encapsulation is non-covalent.

## Mixed guests

Pass a list of guest types (and optionally `ratios`) to dock a mix:

```python
complexes = place_guest_in_host(
    host, [guest_a, guest_b], ratios=[3, 1], n_guests=10, n_complexes=5, seed=0,
)
```

Each of the `n_complexes` configurations draws a distinct random subset of
the underlying packing, so no two complexes repeat the same combination of
guest sites.

## Writing a complex out

`write_host_guest_complex` picks the writer from the output path's
extension - `.gin` (GULP) and `.run` (AMS) use the complex's bond matrix
directly; anything else falls back to plain `ase.io.write`:

```python
from cage_isomer_builder.utils.read_write import write_host_guest_complex

write_host_guest_complex(complexes[0], "complex_0.gin")
```

See the [API reference](../reference/functionalise.md) for the full
parameter set of `place_guest_in_host` and `max_guests_in_host`, including
`overlap_tolerance`, `max_attempts`, and `inner_radius_fraction`.
