# Cage Isomer Builder

A comprehensive Python library for building, enumerating, and functionalising
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
- Functionalise cages and MOFs by attaching real fragments to one, several,
  or all active sites, with control over the distribution by ratio
- Dock guest molecules into finite cages, including mixed-guest systems and
  capacity checks
- Calculate cage-level statistics such as isomer counts,
  functional-group-to-functional-group distance distributions, and
  endo/exo classifications

## Where to start

<div class="grid cards" markdown>

- **[Installation](installation.md)**

    Install from PyPI or from source.

- **[Guide](guide/build-and-enumerate.md)**

    Walk through building a cage, functionalising it, docking guests, and
    computing statistics.

- **[Topologies](topologies.md)**

    Look up which builder class matches your node/linker topicity.

- **[API Reference](reference/cage.md)**

    Full reference for every public class and function.

</div>

## License

This project is released under the [MIT License](license.md).
