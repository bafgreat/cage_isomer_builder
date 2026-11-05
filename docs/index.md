---
description: >-
  Build, enumerate, and functionalise metal-organic cage and MOF isomers in
  Python: 29 cage topologies, UFF4MOF/GFN-xTB optimisation, symmetry-unique
  isomer enumeration, and host-guest docking.
---

# Cage Isomer Builder

A Python library for building, enumerating, and functionalising
metal-organic cages and framework (MOF) isomers. It can also be used as a
lightweight tool for generating host-guest docked complexes for finite cages.

## Main functionalities

- Build any of the 29 most common metal-organic cage topologies from a simple
  node and linker input
- Optimise cage structures with UFF4MOF using real bond orders, or with
  GFN-xTB, while preserving the shape of nodes and linkers through rigid-body
  relaxation
- Generate symmetry-unique isomers for finite cages using point-group
  methods, and for periodic MOFs using space-group methods based on
  [spglib](https://spglib.readthedocs.io/)
- Find the point group of a finite cage from its atoms, so it works for any
  topology, including Ih and D5h cages
- Count and enumerate isomers with two or more different functional groups
  (R1, R2, ...)
- Name isomers by vertex codes (what each node sees of the groups on its
  linkers) and convert between vertex names and site names, in Python or
  with the `cage-vertex-name` command
- Functionalise cages and MOFs by attaching real fragments to one, several,
  or all active sites, with control over the distribution by ratio
- Dock guest molecules into finite cages, including mixed-guest systems and
  capacity checks
- Place guests at their lowest-energy sites, scored with DFT-D4 dispersion
- Calculate cage-level statistics such as isomer counts,
  functional-group-to-functional-group distance distributions, and
  endo/exo classifications
- Describe pores and guests by feature-site pair distributions (functional
  groups, aromatic rings, H-bond sites, open metal sites), averaged exactly
  over multivariate, defective and flexible ensembles, and match guests to
  pores (see [Descriptors](guide/descriptors.md))

## Where to start

<div class="grid cards" markdown>

- **[Installation](installation.md)**

    Install from PyPI or from source.

- **[Quick start](guide/quickstart.md)**

    The whole workflow on one page: build a pore, count and write its
    isomers, attach groups, compute descriptors and dock a guest.

- **[Guide](guide/build-and-enumerate.md)**

    One page per workflow, every example runnable as it is.

- **[Topologies](topologies.md)**

    Look up which builder class matches your node/linker topicity.

- **[API Reference](reference/cage.md)**

    Every public class and function, each with a runnable example.

</div>

## License

This project is released under the [MIT License](license.md).
