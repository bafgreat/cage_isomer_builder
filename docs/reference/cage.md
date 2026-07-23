# Cage builders

`CageBuilder` is the base class every topology-specific builder inherits
from; it holds the full build, optimise, enumerate, functionalise, save, and
statistics API. Topology-specific classes (`Tri4Di6CageBuilder`,
`M6L12CubeCageBuilder`, ...) only override the topology graph and the
point-group transformation library - see [Topologies](../topologies.md) for
the full list.

::: cage_isomer_builder.cage.base.CageBuilder

::: cage_isomer_builder.cage.base.CageStatistics

## Topology-specific builders

::: cage_isomer_builder.cage.library
    options:
      show_root_heading: false
      members_order: source
