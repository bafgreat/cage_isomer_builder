---
description: >-
  How functional-group sites (R1, R2, ...) are stored on structures, written
  to and read from extended XYZ files, and exported to SDF and CXSMILES with
  real R-group atoms.
---

# R sites & file formats

## How a site is stored

ASE has no element "R", so an active site is an ASE dummy atom `X`
(atomic number 0) whose R-group number is kept in the per-atom integer array
`"rgroup"` (1 = R1, 2 = R2, ...; 0 for every other atom). The helpers in
`cage_isomer_builder.utils.sites` read and change them:

```python
from ase.build import molecule
from cage_isomer_builder.utils.sites import mark_site, site_indices, site_map

benzene = molecule("C6H6")          # C 0-5, H 6-11
mark_site(benzene, 6, 1)            # H 6 -> R1
mark_site(benzene, 9, 2)            # H 9 -> R2
print(site_indices(benzene))        # [6, 9]
print(site_map(benzene))            # {6: 'R1', 9: 'R2'}
```

`CageBuilder.functionalise()` marks every aromatic C-H of the linkers this
way, and the isomer writers turn the inactive sites back into H.

## Files

Isomers are written as extended XYZ. Besides the `rgroup` column, the
comment line carries a readable map `r_sites="12:R1 40:R2"` (atom index:
R-group) and the group names in `rgroup_labels` (R1 first). The map is
rebuilt from the array whenever a file is written, so it always matches the
atom order even after atoms were added or removed.

```python
from ase.io import read
from cage_isomer_builder.utils.sites import read_isomer, write_isomer

benzene.info["rgroup_labels"] = "NH2 OH"
write_isomer("benzene_sites.xyz", benzene)
back = read_isomer("benzene_sites.xyz")
print(back.info["r_sites"])                     # {6: 'R1', 9: 'R2'}
print(read("benzene_sites.xyz").info["r_sites"])   # 6:R1 9:R2  (plain ASE reads it too)
```

## SDF and SMILES with real R-group atoms

RDKit exports show the sites as R-group atoms: `R#` with an `M  RGP`
record in SDF/MOL, and atom labels `_R1`, `_R2` in CXSMILES. Bonds come from
a bond-order matrix (for a built cage, `CageBuilder.get_bond_matrix()`); an
order of 1.5 is written as aromatic.

```python
import numpy as np
from rdkit import Chem
from cage_isomer_builder.utils.features import bond_graph
from cage_isomer_builder.utils.rgroup import rgroup_atoms_to_rdkit, rgroup_smiles

bonds = np.zeros((12, 12))
for i, neighbours in bond_graph(back).items():
    for j in neighbours:
        bonds[i, j] = 1.5 if max(i, j) < 6 else 1.0     # ring C-C aromatic, C-site single

print(rgroup_smiles(back, bonds).split(" ")[0])   # [H]c1c([H])c([2*:2])c([H])c([H])c1[1*:1]
Chem.MolToMolFile(rgroup_atoms_to_rdkit(back, bonds), "benzene_sites.sdf")
```

## Other writers

| Function | Output |
| --- | --- |
| `CageBuilder.save(path)` | any ASE format; PDB through STK for built cages |
| `CageBuilder.write_gulp_gin(path)` | GULP input with UFF4MOF types and explicit bonds |
| `CageBuilder.write_ams_run(path)` | AMS run script with bond orders |
| `read_write.write_host_guest_complex(c, path)` | a docked complex: `.gin`, `.run` or any ASE format |
