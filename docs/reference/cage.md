# Cage builders

`CageBuilder` is the base class every topology-specific builder inherits
from; it holds the full build, optimise, enumerate, functionalise, save, and
statistics API. Topology-specific classes (`Tri4Di6CageBuilder`,
`M6L12CubeCageBuilder`, ...) only override the topology graph and the
point-group transformation library. See [Topologies](../topologies.md) for
the full list.

::: cage_isomer_builder.cage.base.CageBuilder

::: cage_isomer_builder.cage.base.CageStatistics

## Topology-specific builders

::: cage_isomer_builder.cage.library
    options:
      show_root_heading: false
      members_order: source

## Building-block helpers

Functions in `cage_isomer_builder.cage.base` that turn node and linker
inputs (SMILES, files, ASE atoms) into STK building blocks, and find the
linkers and nodes of a structure.

::: cage_isomer_builder.cage.base
    options:
      show_root_heading: false
      members_order: source
      members:
        - dummies_to_placeholder
        - make_dummy_factory
        - find_unique_building_units
        - find_linkers
        - get_sbus_indices
        - get_node_centers
        - is_file_like
        - is_ase_atoms
        - is_smiles_like
        - atoms_to_smiles
        - file_to_atoms
        - file_to_smiles
        - mol_input_to_smiles
        - mol_input_to_atoms_or_none
        - normalize_molecule_input
        - get_smiles_and_position_matrix
        - has_dummy_marker_atoms
        - remove_dummy_marker_atoms
        - make_building_block_from_dummy_markers
        - has_dummy_atom
        - is_single_atom_smiles
        - is_single_atom_input
        - get_single_atom_symbol
        - make_single_atom_building_block
        - resolve_factory
        - make_building_block_with_factory
        - try_building_block_with_factory
        - autodetect_building_block
