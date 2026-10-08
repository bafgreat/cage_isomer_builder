---
description: >-
  Relax a built cage with UFF4MOF (explicit bond orders, rigid nodes and
  linkers, optional xTB charges) or GFN-xTB, and write GULP or AMS inputs.
---

# Optimisation

`CageBuilder.optimise()` relaxes a built cage. Two methods are available:

| `method` | Energy | Needs |
| --- | --- | --- |
| `"UFF4MOF"` (default) | UFF4MOF force field, with the STK bond orders | a cage made with `build()` |
| `"GFN1-xTB"`, `"GFN2-xTB"` | tblite tight binding | the optional `tblite` package |

```python
from cage_isomer_builder import example_data
from cage_isomer_builder.cage import Tri4Di6CageBuilder

cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
                          linker=example_data.path("bdc"), scale_multiplier=0.225)
cage.build()
relaxed = cage.optimise(fmax=0.05, steps=500, include_charges=False, logfile=None)
print(len(relaxed))        # 404 atoms; the builder now holds the relaxed geometry
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `rigid_sbus` | `True` | Keep every node and linker rigid and relax only how they sit relative to each other. The linker atom bonded to a node stays free, so the metal-ligand bond can adjust. |
| `include_charges` | `True` | Compute tblite GFN2-xTB charges once (before optimising) and add UFF4MOF's Coulomb term. Without tblite this is skipped with a warning. |
| `logfile` | `"-"` | Optimiser log: `"-"` prints each step, a path writes a file, `None` is silent. |
| `trajectory` | `None` | Path of an ASE `.traj` file of the run. |
| `optimizer_cls` | `LBFGS` | Any ASE optimiser class. |

Optimise before `functionalise()`: STK places bulky nodes away from their
linkers, and symmetry and isomer counts of that geometry are meaningless
(`functionalise()` warns).

## Atom types

UFF4MOF atom types are assigned by
[gulp_setup](https://github.com/bafgreat/gulp_setup) from the bond orders
and geometry. A few types it can assign have no published parameter set;
those are derived by the UFF4MOF convention (new bond radius and angle,
everything else from the same element and oxidation state) and
`optimise()` warns when one is used (see `DERIVED_UFF4MOF_DATA` in
`cage_isomer_builder/__data__/uff4mof_data.py`).

## Lower-level functions

The force field and the xTB backend can be used on any structure with a
bond-order matrix:

```python
import numpy as np
from ase.build import molecule
from cage_isomer_builder.utils.optimise import run_uff4mof_optimisation

methane = molecule("CH4")
bonds = np.zeros((5, 5))
bonds[0, 1:] = bonds[1:, 0] = 1.0
relaxed = run_uff4mof_optimisation(methane, bonds, logfile=None)
print(round(relaxed.get_distance(0, 1), 2))     # 1.11 (UFF C-H)
```

```python
from cage_isomer_builder.utils.tblite_charges import predict_tblite_charges, tblite_available
from cage_isomer_builder.utils.tblite_optimise import run_tblite_optimisation

if tblite_available():
    water, charges = run_tblite_optimisation(molecule("H2O"), method="GFN2-xTB", logfile=None)
    print(charges.round(2))                     # O negative, H positive
    print(predict_tblite_charges(molecule("CH3OH")).round(2))
```

## Writing inputs for other programs

```python
cage.write_gulp_gin("uio66_pore.gin")   # GULP: UFF4MOF types and explicit bonds
cage.write_ams_run("uio66_pore.run")    # AMS (DFTB) run script with bond orders
cage.save("uio66_pore.cif")             # any ASE format
```
