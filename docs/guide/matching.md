---
description: >-
  Describe a flexible guest by its feature-site pair distributions averaged
  over its conformers, and score how well it matches a pore's complementary
  sites.
---

# Guests & matching

A guest is described like a pore: by the distances between its
feature sites (H-bond donors and acceptors, aromatic rings), averaged over
its conformers. Comparing the guest's distances with the pore's distances
between *complementary* sites ranks pores, isomers or guests before any
host-guest energy is computed.

## Conformers

Conformers can come from anywhere: a GA/PES scan, MD snapshots, or a
conformer generator. Here RDKit (installed with STK) makes ten conformers of
ascorbic acid with MMFF energies:

```python
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.AddHs(Chem.MolFromSmiles("OC[C@H](O)[C@H]1OC(=O)C(O)=C1O"))   # ascorbic acid
ids = AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=42)
results = AllChem.MMFFOptimizeMoleculeConfs(mol)                      # [(converged, energy)]
symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
conformers = [Atoms(symbols, positions=mol.GetConformer(i).GetPositions()) for i in ids]
energies = [energy for _, energy in results]                          # kcal/mol
```

## Guest descriptor

```python
from cage_isomer_builder.utils.matching import conformer_descriptor

guest = conformer_descriptor(conformers, energies=energies, unit="kcal/mol", temperature=298.15)
print(guest.keys())    # [('acceptor', 'acceptor'), ('acceptor', 'donor'), ('donor', 'donor')]
print(round(sum(guest.info["weights"]), 6))   # 1.0: Boltzmann weights of the conformers
```

Without energies every conformer counts equally (e.g. MD snapshots, already
Boltzmann-distributed by the simulation); `weights=` sets them directly.

## Matching a pore

A guest donor-acceptor pair at distance d fits a pore acceptor-donor pair at
about d. For every guest pair type, the pore pairs of the complementary
types are pooled and the two distance distributions compared after
normalising each (overlap from 0 to 1); the score is the average over the
guest's pair types, weighted by the guest's pair counts.

| guest site | matched by pore site |
| --- | --- |
| donor | acceptor |
| acceptor | donor, open metal site |
| pi | pi |

```python
from cage_isomer_builder import example_data
from cage_isomer_builder.cage import Tri4Di6CageBuilder
from cage_isomer_builder.utils.matching import match_descriptors

cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
                          linker=example_data.path("bdc"), scale_multiplier=0.225)
cage.build()
cage.optimise(include_charges=False, logfile=None)
cage.functionalise()

pore = cage.get_descriptor(kinds=("pi", "donor", "acceptor", "open_metal"))
result = match_descriptors(pore, guest)
print(result.unmatched)    # guest pair types with no complementary pore pairs
```

This pore has acceptors (the node's mu3-O) but no donors, so the guest's
acceptor pairs find no partner. Functionalising the pore with NH2 adds donors:

```python
from cage_isomer_builder.utils.features import detect_features
from cage_isomer_builder.utils.distributions import pair_descriptor
from cage_isomer_builder.utils.functionalise import functionalise_isomer_sites
from cage_isomer_builder.utils.sites import read_isomer

isomers = cage.enumerate_isomers(output_path="isomers", limit=2)
hosts = {}
for isomer in isomers:
    name = "-".join(map(str, isomer))
    decorated = functionalise_isomer_sites(read_isomer(f"isomers/{name}.xyz"),
                                           example_data.read("NH2"))
    hosts[name] = pair_descriptor(detect_features(decorated))

for name, desc in hosts.items():
    print(name, round(match_descriptors(desc, guest).score, 3))
```

`rank_hosts(hosts, guest)` returns the same scores sorted best first. The
score is a screening measure, not an energy: dock the best candidates and
compute their energies (see [Host-guest docking](host-guest-docking.md)). A
guest with fewer than two features has no pairs at all; its result then has
`defined=False`.

Pass `complementarity=` to change which kinds match, `bandwidth=` (default
0.25 A) for how much mismatch a pocket tolerates, and `metric="bhattacharyya"`
for the Bhattacharyya coefficient instead of the overlap.
