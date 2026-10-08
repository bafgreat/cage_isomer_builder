#!/usr/bin/env python3
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

import os
import re
import warnings
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import stk
from ase import Atoms
from ase.io import read, write
from mofstructure.mofdeconstructor import (
    ase_2_pybel,
    secondary_building_units,
    transition_metals,
)
from stk._internal.topology_graphs.cage.cage import Cage
from stk._internal.topology_graphs.cage.vertices import LinearVertex, NonLinearVertex
from stk._internal.topology_graphs.edge import Edge

from cage_isomer_builder.utils import (
    distributions, ensemble, features, flexibility, pointgroup, rgroup, symmetry,
)
from cage_isomer_builder.utils.functionalise import full_functionalisation
from cage_isomer_builder.utils.geometry import bonded_pairs
from cage_isomer_builder.utils.read_write import (
    generate_isomer_structure_file,
    stk_2_ase_atoms,
    write_run,
)
from cage_isomer_builder.utils.util import (
    iter_unique_isomers,
    kernel_density_estimation,
)


FG_FACTORY_MAP: dict = {
    "aldehyde":          stk.AldehydeFactory,
    "primary_amine":     stk.PrimaryAminoFactory,
    "secondary_amine":   stk.SecondaryAminoFactory,
    "ring_amine":        stk.RingAmineFactory,

    "boronic_acid":      stk.BoronicAcidFactory,
    "diol":              stk.DiolFactory,

    "alcohol":           stk.AlcoholFactory,
    "amide":             stk.AmideFactory,
    "carboxylic_acid":   stk.CarboxylicAcidFactory,
    "bromo":             stk.BromoFactory,
    "dibromo":           stk.DibromoFactory,
    "fluoro":            stk.FluoroFactory,
    "difluoro":          stk.DifluoroFactory,
    "iodo":              stk.IodoFactory,
    "terminal_alkene":   stk.TerminalAlkeneFactory,
    "terminal_alkyne":   stk.TerminalAlkyneFactory,

    "single_atom":       None,

    "smarts":            stk.SmartsFunctionalGroupFactory,
}

MoleculeInput = Union[str, Path, Atoms]

#: SMARTS for SMILES dummy atoms (``*``, ``[*]``, ``[1*]``, ``[#0]``). STK
#: cannot build an atom of atomic number 0, so dummies are rewritten as
#: francium, which does not occur in MOF or cage chemistry. Atom 0 is the
#: placeholder (deleted), atom 1 its neighbour (the bonder).
DUMMY_SMARTS = "[Fr][*]"

_DUMMY_RE = re.compile(r"\[\d*(?:\*|#0)(?::\d+)?\]|\*")


def dummies_to_placeholder(smiles: str) -> str:
    """
    Rewrite every SMILES dummy atom as the ``[Fr]`` placeholder that
    :func:`make_dummy_factory` matches.

    Examples
    --------
    >>> from cage_isomer_builder.cage.base import dummies_to_placeholder
    >>> dummies_to_placeholder("[*]c1ccc(*)cc1"), dummies_to_placeholder("[1*:1]CC[#0]")
    ('[Fr]c1ccc([Fr])cc1', '[Fr]CC[Fr]')
    """
    return _DUMMY_RE.sub("[Fr]", smiles)

# Noble-gas symbols used as geometry/orientation markers in MOF node files.
# They are stripped from building blocks before STK construction so they
# don't appear in cage output or bias the centroid calculation.
_NOBLE_GAS_SYMBOLS = frozenset({'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'})

def _metal_symbols() -> frozenset:
    """mofstructure's metal symbols without ``X``: its list includes the
    dummy symbol, which here marks R sites, not metals."""
    return frozenset(transition_metals()) - {"X"}


_SINGLE_ATOM_RE = re.compile(r"^\[([A-Z][a-z]?)\]$")

_STRUCTURE_EXTENSIONS = {
    ".xyz", ".pdb", ".cif", ".mol", ".sdf", ".mol2", ".gjf", ".com"
}


def make_dummy_factory() -> stk.SmartsFunctionalGroupFactory:
    """
    Factory for dummy atoms attached to any atom, written as the ``[Fr]``
    placeholder (see :func:`dummies_to_placeholder`).

    Atom 0: dummy atom, deleted.
    Atom 1: real attachment atom, used as bonder.

    Examples
    --------
    >>> import stk
    >>> from cage_isomer_builder.cage.base import dummies_to_placeholder, make_dummy_factory
    >>> smiles = dummies_to_placeholder("[*]c1ccc([*])cc1")
    >>> stk.BuildingBlock(smiles, [make_dummy_factory()]).get_num_functional_groups()
    2
    """
    return stk.SmartsFunctionalGroupFactory(
        smarts=DUMMY_SMARTS,
        bonders=(1,),
        deleters=(0,),
    )

def find_unique_building_units(list_of_connected_components,
                               bonds_to_break,
                               ase_atom,
                               porphyrin_checker,
                               all_regions):
    """
    A function adapted from mofstructure that deconstructs a MOF/cage into
    its constituent building units (metal SBUs vs. organic linkers).

    Parameters
    ----------
    list_of_connected_components : list
        Connected components of the structure's bond graph.
    bonds_to_break : list of list
        Atom indices at each breaking point.
    ase_atom : ase.Atoms
    porphyrin_checker : list of int
        Indices of porphyrin atoms.
    all_regions : dict
        Region index -> connected-component mapping.

    Returns
    -------
    mof_metal : list of ase.Atoms
        Metal building-unit fragments.
    mof_linker : list of ase.Atoms
        Organic linker building-unit fragments.
    building_unit_regions : dict
        Region index -> atom-index mapping.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> uio66 = example_data.read("RUBTAK01")
    >>> from mofstructure.mofdeconstructor import secondary_building_units
    >>> from cage_isomer_builder.cage.base import find_unique_building_units
    >>> components, breaks, porphyrin, regions, _ = secondary_building_units(uio66)
    >>> metals, organics, _ = find_unique_building_units(components, breaks, uio66, porphyrin, regions)
    >>> [m.get_chemical_formula() for m in metals], [o.get_chemical_formula() for o in organics]
    (['C12O32Zr6'], ['C6H4'])
    """
    mof_metal = []
    mof_linker = []
    building_unit_regions = {}
    for idx, key in enumerate(all_regions.keys()):
        frag = list(all_regions.keys())[idx]
        components = list_of_connected_components[all_regions[frag][0]]
        all_breaking_point = sum(bonds_to_break, [])
        point_of_extension = [i for i in all_breaking_point if i in components]
        mapped_indices = dict(
            [(i, j) for i, j in zip(components, range(len(components)))])
        molecule_to_write = ase_atom[components]
        molecule_to_write.info['point_of_extension'] = [
            mapped_indices[i] for i in point_of_extension]

        atom_indices_mapping = [
            list_of_connected_components[i] for i in all_regions[key]]
        molecule_to_write.info['atom_indices_mapping'] = atom_indices_mapping

        metal = [i.index for i in molecule_to_write if i.symbol in _metal_symbols()
                 and i.index not in porphyrin_checker]
        if len(metal) > 0:
            mof_metal.append(molecule_to_write)
        else:
            mof_linker.append(molecule_to_write)
        building_unit_regions[idx] = molecule_to_write.info['atom_indices_mapping']
    return mof_metal, mof_linker, building_unit_regions

def find_linkers(ase_atom):
    """
    Every linker of a MOF or cage, grouped by chemical type.

    A linker is an organic building unit (from mofstructure's deconstruction)
    bonded to the framework at two or more points, so terminal ligands (e.g. a
    thiolate bound to one metal) and unbound guests are not linkers. All linker
    types are kept. If no unit has two connection points, the first organic unit
    with at least six atoms is used.

    Returns
    -------
    linker_atom_indices : list of list of int
        Atom indices of every linker instance.
    linker_types : list of int
        Type index of each instance.
    type_names : list of str
        Chemical formula of each type (made unique if two types share one).
    metal_atom_indices : list of int
        Every atom not in a linker.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage.base import find_linkers
    >>> linkers, types, names, rest = find_linkers(example_data.read("ABIXEQ"))
    >>> names, len(linkers)                        # two linker types, one of each per cell
    (['C18H18N4', 'C6H4'], 2)
    >>> find_linkers(example_data.read("ABINOQ"))[2]   # bipyridine; terminal thiolates excluded
    ['C10H8N2']
    """
    c_components, breaking_point, porpyrin, regions, _ = secondary_building_units(ase_atom)
    _, organic_sbu, _ = find_unique_building_units(
        c_components, breaking_point, ase_atom, porpyrin, regions)
    chosen = [sbu for sbu in organic_sbu if len(sbu.info['point_of_extension']) >= 2]
    if not chosen:
        chosen = [sbu for sbu in organic_sbu if len(sbu) >= 6][:1]
    if not chosen:
        raise ValueError("no linker found: no organic building unit is bonded to the "
                         "framework at two or more points.")
    linker_atom_indices, linker_types, type_names = [], [], []
    for t, sbu in enumerate(chosen):
        name = sbu.get_chemical_formula()
        while name in type_names:
            name += "'"
        type_names.append(name)
        for instance in sbu.info['atom_indices_mapping']:
            linker_atom_indices.append(list(instance))
            linker_types.append(t)
    in_linker = {i for lk in linker_atom_indices for i in lk}
    rest = [i for i in range(len(ase_atom)) if i not in in_linker]
    return linker_atom_indices, linker_types, type_names, rest


def get_sbus_indices(ase_atom):
    """
    Return linker and metal atom indices from a MOF or metal-organic cage.

    Parameters
    ----------
    ase_atom : ase.Atoms
        Structure to analyse.

    Returns
    -------
    linker_atom_indices : list of list of int
        Atom indices per organic linker.
    metal_atom_indicces : list of int
        Flat list of atom indices belonging to metal nodes.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> uio66 = example_data.read("RUBTAK01")
    >>> from cage_isomer_builder.cage.base import get_sbus_indices
    >>> linkers, rest = get_sbus_indices(uio66)
    >>> len(linkers), len(linkers[0]), len(rest)    # 24 phenylene rings in the cell
    (24, 10, 200)
    """

    linker_atom_indices, _, _, rest = find_linkers(ase_atom)
    return linker_atom_indices, rest


def get_node_centers(ase_atom, metal_atom_indices):
    """
    Return the centre-of-mass position of each metal node instance.

    Builds a bond graph restricted to the supplied metal atom indices,
    finds connected components (one per node cluster), and returns one
    CoM per component. This works for both single-atom nodes (each atom
    is its own component) and multi-atom clusters (e.g. Zr6).

    The flat metal_atom_indices list from get_sbus_indices is used directly
    so CH3 groups on the node are correctly assigned to the node cluster.

    Parameters
    ----------
    ase_atom : ase.Atoms
    metal_atom_indices : list of int
        Flat list of atom indices belonging to nodes (not linkers).

    Returns
    -------
    node_centers : np.ndarray, shape (n_nodes, 3), or None if empty

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> uio66 = example_data.read("RUBTAK01")
    >>> from cage_isomer_builder.cage.base import get_node_centers, get_sbus_indices
    >>> _, rest = get_sbus_indices(uio66)
    >>> get_node_centers(uio66, rest).shape          # the four Zr6 nodes of the fcu cell
    (4, 3)
    """
    if not metal_atom_indices:
        return None

    metal_sub = ase_atom[metal_atom_indices]
    adjacency: dict[int, list[int]] = {i: [] for i in range(len(metal_sub))}
    for i, j in bonded_pairs(metal_sub):
        adjacency[i].append(j)
        adjacency[j].append(i)

    # BFS to find connected components within the node subgraph
    n = len(metal_sub)
    visited = [False] * n
    components = []
    for start in range(n):
        if visited[start]:
            continue
        component = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop()
            component.append(node)
            for nb in adjacency[node]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(component)

    centers = [
        ase_atom[[metal_atom_indices[i] for i in comp]].get_center_of_mass()
        for comp in components
    ]
    return np.array(centers)


def is_file_like(value: Any) -> bool:
    """Return True if value looks like a molecular structure file path.

    Examples
    --------
    >>> from cage_isomer_builder.cage.base import is_file_like
    >>> is_file_like("linker.xyz"), is_file_like("Nc1ccc(N)cc1")
    (True, False)
    """
    if not isinstance(value, (str, Path)):
        return False

    path = Path(value)
    return path.suffix.lower() in _STRUCTURE_EXTENSIONS


def is_ase_atoms(value: Any) -> bool:
    """Return True if value is an ASE Atoms object.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.cage.base import is_ase_atoms
    >>> is_ase_atoms(molecule("CH4")), is_ase_atoms("C")
    (True, False)
    """
    return isinstance(value, Atoms)


def is_smiles_like(value: Any) -> bool:
    """Return True if value should be interpreted as a SMILES string.

    Examples
    --------
    >>> from cage_isomer_builder.cage.base import is_smiles_like
    >>> is_smiles_like("Nc1ccc(N)cc1"), is_smiles_like("linker.mol2")
    (True, False)
    """
    return isinstance(value, str) and not is_file_like(value)


def atoms_to_smiles(atoms: Atoms) -> str:
    """
    Convert ASE Atoms to SMILES using mofstructure/OpenBabel.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.cage.base import atoms_to_smiles
    >>> atoms_to_smiles(molecule("CH3OH")).split()[0]
    'CO'
    """
    smiles = ase_2_pybel(atoms).write("smi")
    if not smiles:
        raise ValueError("Could not convert ASE Atoms object to SMILES.")
    return smiles


def file_to_atoms(path: Union[str, Path]) -> Atoms:
    """Read a molecular structure file with ASE.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage.base import file_to_atoms
    >>> file_to_atoms(example_data.path("methane")).get_chemical_formula()
    'CH4'
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")
    return read(str(path))


def file_to_smiles(path: Union[str, Path]) -> str:
    """
    Convert a molecular structure file to SMILES using ASE + mofstructure.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage.base import file_to_smiles
    >>> file_to_smiles(example_data.path("methane")).split()[0]
    'C'
    """
    atoms = file_to_atoms(path)
    return atoms_to_smiles(atoms)


def mol_input_to_smiles(mol: MoleculeInput) -> str:
    """
    Convert SMILES, file path, or ASE Atoms into a SMILES string.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.cage.base import mol_input_to_smiles
    >>> mol_input_to_smiles("  NCCN "), mol_input_to_smiles(molecule("CH4")).split()[0]
    ('NCCN', 'C')
    """
    if is_ase_atoms(mol):
        return atoms_to_smiles(mol)

    if is_file_like(mol):
        return file_to_smiles(mol)

    if is_smiles_like(mol):
        return mol.strip()

    raise TypeError(
        "Molecule input must be a SMILES string, a structure file path, "
        "or an ASE Atoms object."
    )


def mol_input_to_atoms_or_none(mol: MoleculeInput) -> Optional[Atoms]:
    """
    Return ASE Atoms if input is a file or ASE Atoms, otherwise None.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage.base import mol_input_to_atoms_or_none
    >>> print(mol_input_to_atoms_or_none("NCCN"))    # a SMILES string has no coordinates
    None
    >>> len(mol_input_to_atoms_or_none(example_data.path("methane")))
    5
    """
    if is_ase_atoms(mol):
        return mol

    if is_file_like(mol):
        return file_to_atoms(mol)

    return None


def normalize_molecule_input(
    mol: Optional[MoleculeInput],
    default: MoleculeInput,
) -> MoleculeInput:
    """
    Normalize user molecule input for later building-block creation.

    Defaults and SMILES strings are kept as SMILES. Structure files are read
    once into ASE Atoms, and ASE Atoms are passed through.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage.base import normalize_molecule_input
    >>> normalize_molecule_input(None, default="[Zr]"), normalize_molecule_input(" NCCN", "[Zr]")
    ('[Zr]', 'NCCN')
    >>> type(normalize_molecule_input(example_data.path("bdc"), "[Zr]")).__name__   # files are read
    'Atoms'
    """
    if mol is None:
        return default

    if is_smiles_like(mol):
        return mol.strip()

    atoms = mol_input_to_atoms_or_none(mol)
    if atoms is not None:
        return atoms

    raise TypeError(
        "Molecule input must be a SMILES string, a structure file path, "
        "or an ASE Atoms object."
    )


def get_smiles_and_position_matrix(
    mol: MoleculeInput,
) -> tuple[str, Optional[np.ndarray]]:
    """
    Return SMILES and coordinates for STK BuildingBlock construction.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.cage.base import get_smiles_and_position_matrix
    >>> get_smiles_and_position_matrix("NCCN")
    ('NCCN', None)
    >>> smiles, positions = get_smiles_and_position_matrix(molecule("CH4"))
    >>> positions.shape
    (5, 3)
    """
    atoms = mol_input_to_atoms_or_none(mol)
    if atoms is None:
        return mol_input_to_smiles(mol), None

    return atoms_to_smiles(atoms), atoms.get_positions()


def has_dummy_marker_atoms(atoms: Atoms) -> bool:
    """Return True if ASE Atoms contains dummy attachment markers.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage.base import has_dummy_marker_atoms
    >>> has_dummy_marker_atoms(example_data.read("bdc")), has_dummy_marker_atoms(molecule("CH4"))
    (True, False)
    """
    return any(atom.symbol == "X" for atom in atoms)


def remove_dummy_marker_atoms(atoms: Atoms) -> tuple[Atoms, list[int]]:
    """
    Remove dummy marker and noble-gas atoms; return bonder indices.

    ``X`` atoms mark attachment directions. Noble-gas atoms (He, Ne, ...) are
    orientation markers added by MOF deconstruction tools. Both are removed
    before STK construction. Each ``X`` marker is paired with the nearest
    non-hydrogen atom of the stripped molecule, which becomes the bonder of the
    corresponding functional group.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage.base import remove_dummy_marker_atoms
    >>> stripped, bonders = remove_dummy_marker_atoms(example_data.read("bdc"))
    >>> stripped.get_chemical_formula(), [stripped[i].symbol for i in bonders]
    ('C6H4', ['C', 'C'])
    """
    dummy_positions = [
        atom.position
        for atom in atoms
        if atom.symbol == "X"
    ]
    keep_indices = [
        atom.index
        for atom in atoms
        if atom.symbol != "X" and atom.symbol not in _NOBLE_GAS_SYMBOLS
    ]
    stripped = atoms[keep_indices]
    stripped_positions = stripped.get_positions()
    candidate_indices = [
        atom.index
        for atom in stripped
        if atom.symbol != "H"
    ]

    if not candidate_indices:
        candidate_indices = list(range(len(stripped)))

    bonder_ids = []
    candidate_positions = stripped_positions[candidate_indices]
    for position in dummy_positions:
        distances = np.linalg.norm(candidate_positions - position, axis=1)
        bonder_ids.append(candidate_indices[int(np.argmin(distances))])

    return stripped, bonder_ids


def _bonds_from_ase_atoms(atoms: Atoms, stk_atoms: list) -> list:
    """Infer bonds from interatomic distances (see geometry.bonded_pairs)."""
    return [
        stk.Bond(atom1=stk_atoms[i], atom2=stk_atoms[j], order=1)
        for i, j in bonded_pairs(atoms)
    ]


def make_building_block_from_dummy_markers(
    atoms: Atoms,
) -> stk.BuildingBlock:
    """
    Build an STK BuildingBlock from ASE Atoms with ``X`` marker atoms.

    Uses ``BuildingBlock.init()`` directly, without a SMILES round trip, so the
    atom order and 3D coordinates are kept. A SMILES round trip through
    OpenBabel and RDKit reorders atoms, which would put the bonders of large
    inorganic clusters (e.g. UiO-66 Zr nodes) at the wrong positions.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage.base import make_building_block_from_dummy_markers
    >>> bb = make_building_block_from_dummy_markers(example_data.read("bdc"))
    >>> bb.get_num_atoms(), bb.get_num_functional_groups()
    (10, 2)
    """
    stripped, bonder_ids = remove_dummy_marker_atoms(atoms)

    # Build STK atoms in the same order as the stripped ASE Atoms object.
    stk_atoms = []
    for i, atom in enumerate(stripped):
        element_cls = getattr(stk, atom.symbol, None)
        if element_cls is None:
            raise ValueError(
                f"STK does not expose element stk.{atom.symbol!r}. "
                "Remove unsupported elements from the building block file."
            )
        stk_atoms.append(element_cls(i))

    bonds = _bonds_from_ase_atoms(stripped, stk_atoms)

    functional_groups = [
        stk.GenericFunctionalGroup(
            atoms=(stk_atoms[i],),
            bonders=(stk_atoms[i],),
            deleters=(),
        )
        for i in bonder_ids
    ]

    return stk.BuildingBlock.init(
        atoms=stk_atoms,
        bonds=bonds,
        position_matrix=stripped.get_positions(),
        functional_groups=functional_groups,
    )


def has_dummy_atom(smiles: str) -> bool:
    """
    Detect dummy atoms in SMILES.

    Handles common dummy representations:
    - [*]
    - *
    - [#0]

    Examples
    --------
    >>> from cage_isomer_builder.cage.base import has_dummy_atom
    >>> has_dummy_atom("[*]c1ccc([*])cc1"), has_dummy_atom("c1ccccc1")
    (True, False)
    """
    return "[*]" in smiles or "[#0]" in smiles or "*" in smiles


def is_single_atom_smiles(smiles: str) -> bool:
    """Return True for bracketed single-atom SMILES such as [Zn], [Pd], [Zr].

    Examples
    --------
    >>> from cage_isomer_builder.cage.base import is_single_atom_smiles
    >>> is_single_atom_smiles("[Zn]"), is_single_atom_smiles("[Zn]O")
    (True, False)
    """
    return bool(_SINGLE_ATOM_RE.match(smiles.strip()))


def is_single_atom_input(mol: MoleculeInput) -> bool:
    """
    Return True if input represents exactly one atom.

    Examples
    --------
    >>> from ase import Atoms
    >>> from cage_isomer_builder.cage.base import is_single_atom_input
    >>> is_single_atom_input("[Pd]"), is_single_atom_input(Atoms("Cu")), is_single_atom_input("NCCN")
    (True, True, False)
    """
    if is_ase_atoms(mol):
        return len(mol) == 1

    if is_file_like(mol):
        return len(file_to_atoms(mol)) == 1

    if is_smiles_like(mol):
        return is_single_atom_smiles(mol)

    return False


def get_single_atom_symbol(mol: MoleculeInput) -> str:
    """
    Extract element symbol from a single-atom input.

    Examples
    --------
    >>> from ase import Atoms
    >>> from cage_isomer_builder.cage.base import get_single_atom_symbol
    >>> get_single_atom_symbol("[Pd]"), get_single_atom_symbol(Atoms("Cu"))
    ('Pd', 'Cu')
    """
    if is_ase_atoms(mol):
        if len(mol) != 1:
            raise ValueError("ASE Atoms object is not a single atom.")
        return mol[0].symbol

    if is_file_like(mol):
        atoms = file_to_atoms(mol)
        if len(atoms) != 1:
            raise ValueError(f"File {mol!r} does not contain a single atom.")
        return atoms[0].symbol

    smiles = str(mol).strip()
    match = _SINGLE_ATOM_RE.match(smiles)
    if not match:
        raise ValueError(f"Expected single-atom SMILES, got {smiles!r}.")

    return match.group(1)


def make_single_atom_building_block(
    mol: MoleculeInput,
    fg_count: int,
) -> stk.BuildingBlock:
    """
    Build an STK single-atom BuildingBlock.

    This is intended for metal nodes such as [Zn], [Pd], [Zr], etc.

    Examples
    --------
    >>> from cage_isomer_builder.cage.base import make_single_atom_building_block
    >>> make_single_atom_building_block("[Zr]", fg_count=4).get_num_functional_groups()
    4
    """
    symbol = get_single_atom_symbol(mol)
    element_cls = getattr(stk, symbol, None)

    if element_cls is None:
        raise ValueError(
            f"STK does not expose element stk.{symbol}. "
            "Check whether this element is supported by your STK version."
        )

    atoms = mol_input_to_atoms_or_none(mol)

    if atoms is not None:
        position_matrix = atoms.get_positions()
        smiles = f"[{symbol}]"
    else:
        position_matrix = [[0.0, 0.0, 0.0]]
        smiles = str(mol).strip()

    return stk.BuildingBlock(
        smiles=smiles,
        functional_groups=(
            stk.SingleAtom(element_cls(0))
            for _ in range(fg_count)
        ),
        position_matrix=position_matrix,
    )


def resolve_factory(
    fg: Any,
    smarts_pattern: Optional[str] = None,
):
    """
    Resolve a functional-group name or factory instance.

    Returns None for 'single_atom'.

    Examples
    --------
    >>> from cage_isomer_builder.cage.base import resolve_factory
    >>> type(resolve_factory("primary_amine")).__name__, resolve_factory("single_atom")
    ('PrimaryAminoFactory', None)
    """
    if fg is None:
        return None

    if not isinstance(fg, str):
        return fg

    if fg not in FG_FACTORY_MAP:
        raise ValueError(
            f"Unknown functional group {fg!r}. "
            f"Choose from {list(FG_FACTORY_MAP)} or pass a factory instance."
        )

    factory_cls = FG_FACTORY_MAP[fg]

    if fg == "single_atom":
        return None

    if fg == "smarts":
        if smarts_pattern is None:
            raise ValueError("smarts_pattern is required when fg='smarts'.")

        return stk.SmartsFunctionalGroupFactory(
            smarts=smarts_pattern,
            bonders=(1,),
            deleters=(0,),
        )

    return factory_cls()


def make_building_block_with_factory(
    smiles: str,
    factory,
    position_matrix: Optional[np.ndarray] = None,
) -> stk.BuildingBlock:
    """
    Build an STK BuildingBlock from SMILES and one factory.

    Examples
    --------
    >>> import stk
    >>> from cage_isomer_builder.cage.base import make_building_block_with_factory
    >>> bb = make_building_block_with_factory("NCCN", stk.PrimaryAminoFactory())
    >>> bb.get_num_functional_groups()
    2
    """
    return stk.BuildingBlock(
        smiles=smiles,
        functional_groups=[factory],
        position_matrix=position_matrix,
    )


def try_building_block_with_factory(
    smiles: str,
    factory,
    position_matrix: Optional[np.ndarray] = None,
) -> Optional[stk.BuildingBlock]:
    """
    Try one factory and return the BuildingBlock only if it finds FGs.

    Examples
    --------
    >>> import stk
    >>> from cage_isomer_builder.cage.base import try_building_block_with_factory
    >>> print(try_building_block_with_factory("CCO", stk.PrimaryAminoFactory()))   # no amine
    None
    """
    try:
        bb = make_building_block_with_factory(
            smiles=smiles,
            factory=factory,
            position_matrix=position_matrix,
        )
    except Exception:
        return None

    if bb.get_num_functional_groups() == 0:
        return None

    return bb


def autodetect_building_block(
    mol: MoleculeInput,
    fg: Any = None,
    fg_count: int = 1,
    smarts_pattern: Optional[str] = None,
    allow_single_atom: bool = True,
    role: str = "building block",
) -> stk.BuildingBlock:
    """
    Build an STK BuildingBlock from SMILES, file, or ASE Atoms.

    Detection priority
    ------------------
    1. If true single atom and allow_single_atom=True:
       use stk.SingleAtom.
    2. If ASE/file input contains ``X`` marker atoms:
       strip markers and create GenericFunctionalGroup bonders.
    3. Convert file/ASE input to SMILES using mofstructure/OpenBabel.
    4. If SMILES dummy atoms (``*``, ``[*]``, ``[#0]``) are present:
       rewrite them as ``[Fr]`` and use the ``[Fr][*]`` SMARTS factory.
    5. If user provided fg:
       use that factory.
    6. Otherwise search FG_FACTORY_MAP.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage.base import autodetect_building_block
    >>> autodetect_building_block("Nc1ccc(N)cc1").get_num_functional_groups()    # two amines
    2
    >>> autodetect_building_block(example_data.path("bdc")).get_num_functional_groups()  # X markers
    2
    >>> autodetect_building_block("[Zr]", fg_count=4).get_num_functional_groups()
    4
    """

    if allow_single_atom and is_single_atom_input(mol):
        return make_single_atom_building_block(
            mol=mol,
            fg_count=fg_count,
        )

    atoms = mol_input_to_atoms_or_none(mol)
    if atoms is not None and has_dummy_marker_atoms(atoms):
        return make_building_block_from_dummy_markers(atoms)

    smiles, position_matrix = get_smiles_and_position_matrix(mol)

    if has_dummy_atom(smiles):
        bb = try_building_block_with_factory(
            smiles=dummies_to_placeholder(smiles),
            factory=make_dummy_factory(),
            position_matrix=position_matrix,
        )
        if bb is not None:
            return bb

    if fg is not None:
        factory = resolve_factory(
            fg=fg,
            smarts_pattern=smarts_pattern,
        )

        if factory is None:
            return make_single_atom_building_block(
                mol=mol,
                fg_count=fg_count,
            )

        bb = try_building_block_with_factory(
            smiles=smiles,
            factory=factory,
            position_matrix=position_matrix,
        )
        if bb is not None:
            return bb

        raise ValueError(
            f"Factory {fg!r} was provided for {role}, but it found no "
            f"functional groups in SMILES: {smiles}"
        )

    for name, factory_cls in FG_FACTORY_MAP.items():
        if name in {"single_atom", "smarts"}:
            continue

        try:
            factory = factory_cls()
            bb = try_building_block_with_factory(
                smiles=smiles,
                factory=factory,
                position_matrix=position_matrix,
            )
        except Exception:
            continue

        if bb is not None:
            return bb

    raise ValueError(
        f"Could not detect functional groups for {role}.\n"
        f"SMILES used: {smiles}\n"
        "Solutions:\n"
        "  - add dummy atoms such as [*]...[*],\n"
        "  - pass linker_fg= or node_fg= explicitly,\n"
        "  - or extend FG_FACTORY_MAP."
    )


@dataclass
class CageStatistics:
    """
    FG-FG distance statistics of a structure (see :meth:`CageBuilder.get_statistics`).

    Attributes
    ----------
    n_linkers : int
        Number of linkers carrying sites.
    n_fg_slots : int
        Total number of functional-group sites.
    n_unique_isomers : int
        Symmetry-unique isomers with one group per linker (Burnside's lemma).
    distance_keys : list of float
        Sorted distinct site-site distances between linkers (Angstrom).
    total_fg_pairs : list of int
        Number of site pairs at each distance.
    endo_endo_pairs : list of int
        Pairs where both sites point into the pore.
    endo_exo_pairs : list of int
        Pairs with one site pointing in and one out.
    exo_exo_pairs : list of int
        Pairs where both sites point out.
    kde_x : numpy.ndarray, shape (n_kde_bins,)
        Distance grid of the smoothed curve.
    kde_y : numpy.ndarray, shape (n_kde_bins,)
        Smoothed pair counts on ``kde_x``, scaled to the total pair count.
    other_pairs : list of int
        Pairs involving a tangential site (its C-site bond within 60 degrees of
        perpendicular to the pore direction, e.g. on a ring lying face-on to the
        pore) or, in periodic structures, an unclassified site. The four classes
        add up to ``total_fg_pairs``.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
    >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
    ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
    >>> _ = cage.build()
    >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
    >>> _ = cage.functionalise()
    >>> stats = cage.get_statistics()
    >>> stats.n_linkers, stats.n_fg_slots, stats.n_unique_isomers, sum(stats.total_fg_pairs)
    (6, 24, 176, 240)
    """

    n_linkers: int
    n_fg_slots: int
    n_unique_isomers: int
    distance_keys: list
    total_fg_pairs: list
    endo_endo_pairs: list
    endo_exo_pairs: list
    exo_exo_pairs: list
    kde_x: np.ndarray
    kde_y: np.ndarray
    other_pairs: list = None


def _bipartition(topology_cls):
    """
    The two colour classes of a topology graph's vertices (every edge joins one
    vertex of each), as two tuples of vertex ids. The first contains vertex 0.

    Raises
    ------
    ValueError
        If the vertex graph is not bipartite.
    """
    n = len(topology_cls._vertex_prototypes)
    neighbours = {v: [] for v in range(n)}
    for edge in topology_cls._edge_prototypes:
        a, b = edge.get_vertex1_id(), edge.get_vertex2_id()
        neighbours[a].append(b)
        neighbours[b].append(a)
    colour = {}
    for start in range(n):
        if start in colour:
            continue
        colour[start] = 0
        stack = [start]
        while stack:
            v = stack.pop()
            for w in neighbours[v]:
                if w not in colour:
                    colour[w] = 1 - colour[v]
                    stack.append(w)
                elif colour[w] == colour[v]:
                    raise ValueError(f"{topology_cls.__name__}'s vertex graph is not bipartite.")
    return (tuple(v for v in range(n) if colour[v] == 0),
            tuple(v for v in range(n) if colour[v] == 1))


# ============================================================================
#  Custom STK topology graphs
# ============================================================================

class _TwoPlusThree(Cage):
    """Custom STK topology for Tri2Di3 (2 triamine nodes + 3 ditopic linkers)."""
    _vertex_prototypes = (
        NonLinearVertex(0, [0, 0, 1]),
        NonLinearVertex(1, [0, 0, -1]),
        LinearVertex(2, [np.sin(np.pi * 2 / 3), np.cos(np.pi * 2 / 3), 0], False),
        LinearVertex(3, [np.sin(np.pi * 4 / 3), np.cos(np.pi * 4 / 3), 0], False),
        LinearVertex(4, [0, 1, 0], False),
    )
    _edge_prototypes = (
        Edge(0, _vertex_prototypes[0], _vertex_prototypes[2]),
        Edge(1, _vertex_prototypes[0], _vertex_prototypes[3]),
        Edge(2, _vertex_prototypes[0], _vertex_prototypes[4]),
        Edge(3, _vertex_prototypes[1], _vertex_prototypes[2]),
        Edge(4, _vertex_prototypes[1], _vertex_prototypes[3]),
        Edge(5, _vertex_prototypes[1], _vertex_prototypes[4]),
    )
    _num_windows = 3
    _num_window_types = 1


_x = 1.0


class _ThreePlusSix(Cage):
    """Custom STK topology for Tet3-3Di3 (3 triamine nodes + 6 ditopic linkers)."""
    _vertex_prototypes = (
        NonLinearVertex(0, [_x * np.sin(np.pi * 4 / 3), _x * np.cos(np.pi * 4 / 3), 0], False),
        NonLinearVertex(1, [_x * np.sin(np.pi * 2 / 3), _x * np.cos(np.pi * 2 / 3), 0], False),
        NonLinearVertex(2, [0, _x, 0], False),
        LinearVertex(3, [_x * np.sin(np.pi),      _x * np.cos(np.pi),      _x * np.sqrt(3) / 4], False),
        LinearVertex(4, [_x * np.sin(np.pi),      _x * np.cos(np.pi),     -_x * np.sqrt(3) / 4], False),
        LinearVertex(5, [_x * np.sin(np.pi / 3),  _x * np.cos(np.pi / 3),  _x * np.sqrt(3) / 4], False),
        LinearVertex(6, [_x * np.sin(np.pi / 3),  _x * np.cos(np.pi / 3), -_x * np.sqrt(3) / 4], False),
        LinearVertex(7, [_x * np.sin(-np.pi / 3), _x * np.cos(-np.pi / 3), _x * np.sqrt(3) / 4], False),
        LinearVertex(8, [_x * np.sin(-np.pi / 3), _x * np.cos(-np.pi / 3),-_x * np.sqrt(3) / 4], False),
    )
    _edge_prototypes = (
        Edge(0,  _vertex_prototypes[3], _vertex_prototypes[0]),
        Edge(1,  _vertex_prototypes[3], _vertex_prototypes[1]),
        Edge(2,  _vertex_prototypes[4], _vertex_prototypes[0]),
        Edge(3,  _vertex_prototypes[4], _vertex_prototypes[1]),
        Edge(4,  _vertex_prototypes[5], _vertex_prototypes[1]),
        Edge(5,  _vertex_prototypes[5], _vertex_prototypes[2]),
        Edge(6,  _vertex_prototypes[6], _vertex_prototypes[1]),
        Edge(7,  _vertex_prototypes[6], _vertex_prototypes[2]),
        Edge(8,  _vertex_prototypes[7], _vertex_prototypes[0]),
        Edge(9,  _vertex_prototypes[7], _vertex_prototypes[2]),
        Edge(10, _vertex_prototypes[8], _vertex_prototypes[0]),
        Edge(11, _vertex_prototypes[8], _vertex_prototypes[2]),
    )
    _num_windows = 5
    _num_window_types = 2


# ============================================================================
#  Abstract base class
# ============================================================================

class CageBuilder(ABC):
    """
    Base class for building cages and enumerating their isomers.

    Each subclass sets one STK topology graph and, optionally, a symmetry
    library. The shared workflow is: build, optimise, functionalise, count or
    enumerate isomers, then compute statistics and descriptors. Structures from
    files (a MOF CIF, a cage from elsewhere) enter the same workflow through
    :meth:`load_from_file`.

    Optimise a built cage before functionalising it. STK can place bulky nodes
    away from their linkers, and ``functionalise()`` warns when the building
    blocks are not bonded.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
    >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
    ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
    >>> _ = cage.build()                                  # STK assembly
    >>> _ = cage.optimise(include_charges=False, logfile=None)   # UFF4MOF, rigid building blocks
    >>> anchors, slots = cage.functionalise()             # every aromatic C-H becomes an R site
    >>> cage.point_group().name, cage.count_unique_isomers_burnside()
    ('Td', 176)
    """

    topology_name: str = ""
    default_node: MoleculeInput = "[Zr]"
    default_node_fg: str = "single_atom"
    default_node_fg_count: int = 4
    default_linker: MoleculeInput = "Nc1ccc(N)cc1"
    default_linker_fg = None

    # Subclasses set these to avoid overriding _build_stk_topology entirely.
    # _stk_topology_cls : the STK topology class to instantiate.
    # _use_collapser    : whether to pass a Collapser optimizer (default True).
    _stk_topology_cls = None
    _use_collapser: bool = True

    #: How a finite cage's symmetry is found. ``"detect"``: from the atoms
    #: (utils.pointgroup), for any topology, orientation and node count.
    #: ``"topology"``: the per-topology library in utils.symmetry, which
    #: assumes the standard orientation. Periodic structures always use
    #: spglib.
    symmetry_method: str = "detect"
    #: Atoms the operations are detected from: "auto" (metals, else heavy
    #: atoms), "metals", "heavy" or "all" (see pointgroup.reference_indices).
    symmetry_reference: str = "auto"
    #: Matching tolerance (Angstrom) for the reference atoms.
    symmetry_tol: float = 0.5
    #: Matching tolerance (Angstrom) for the FG anchor slots.
    symmetry_anchor_tol: float = 1.0

    def __init__(
        self,
        linker: MoleculeInput = None,
        node: MoleculeInput = None,
        linker_fg=None,
        node_fg=None,
        node_fg_count: int = None,
        smarts_pattern: str = None,
        optimizer_step: float = 0.01,
        distance_threshold: float = 5.0,
        expansion_factor: float = 1.0,
        scale_multiplier: float = 1.0,
        linker_smiles: str = None,
        node_smiles: str = None,
    ):
        """
        Initialise a cage builder.

        The linker and node may be a SMILES string, an ASE ``Atoms`` object, or a
        structure file readable by ASE (``.xyz``, ``.mol``, ``.mol2``, ``.sdf``,
        ``.pdb``, ``.cif``, ...).

        Functional groups (the connection points) are found in this order:

        1. **Single atom** (nodes only): a node of one atom (e.g. ``"[Zn]"``)
           becomes an ``stk.SingleAtom`` building block.
        2. **Dummy atoms**: ``X`` atoms in a structure, or ``*``, ``[*]`` or
           ``[#0]`` in a SMILES string, mark the connection points. SMILES dummies
           are rewritten as ``[Fr]`` placeholders (STK cannot build atomic number
           0) and matched by the SMARTS ``[Fr][*]``.
        3. **Given factory**: ``linker_fg`` or ``node_fg``.
        4. **Automatic**: the factories of ``FG_FACTORY_MAP`` are tried in turn.

        Parameters
        ----------
        linker : MoleculeInput, optional
            Linker building block. Defaults to ``default_linker``.
        node : MoleculeInput, optional
            Node building block. Defaults to ``default_node``.
        linker_fg : str or stk.FunctionalGroupFactory, optional
            Functional-group factory of the linker: a key of ``FG_FACTORY_MAP``, a
            factory instance, or ``None`` for automatic detection.
        node_fg : str or stk.FunctionalGroupFactory, optional
            Functional-group factory of the node. ``"single_atom"`` for an
            isolated metal, ``None`` for automatic detection. The default node uses
            ``default_node_fg``; a given node defaults to automatic detection.
        node_fg_count : int, optional
            Number of functional groups of a single-atom node (ignored otherwise).
            Defaults to ``default_node_fg_count``.
        smarts_pattern : str, optional
            SMARTS pattern used when ``linker_fg`` or ``node_fg`` is ``"smarts"``.
        optimizer_step : float, default 0.01
            Step size of the STK ``Collapser`` optimiser.
        distance_threshold : float, default 5.0
            Distance threshold (Angstrom) of the STK ``Collapser`` optimiser.
        expansion_factor : float, default 1.0
            After assembly, every building block is translated rigidly so that its
            distance from the cage centroid is multiplied by this factor. Use a value
            slightly above 1 when bulky nodes overlap their neighbours in the
            initial placement; the smallest factor is ``2 * r_max / d`` for a node of
            outer radius ``r_max`` and vertex-vertex distance ``d`` (about 1.15 for
            ``uio66_tri_node``).
        scale_multiplier : float, default 1.0
            Passed to the STK topology graph, which scales its vertex positions by
            ``scale_multiplier`` times the largest building-block diameter. A bulky
            node next to a short linker (a Zr6 node with a phenylene) is then placed
            far from its linkers. To correct it, build once, note the node-node
            distance ``d0`` and the distance you want ``d`` (linker length plus two
            metal-ligand bonds), and set ``scale_multiplier = d / d0``.
        linker_smiles, node_smiles : str, optional
            Deprecated aliases of ``linker`` and ``node``.
        """

        # ------------------------------------------------------------------
        # Input normalization
        # ------------------------------------------------------------------
        linker_input = linker if linker is not None else linker_smiles
        node_input = node if node is not None else node_smiles

        self.linker = normalize_molecule_input(
            mol=linker_input,
            default=self.default_linker,
        )

        self.node = normalize_molecule_input(
            mol=node_input,
            default=self.default_node,
        )

        self.linker_fg = (
            linker_fg
            if linker_fg is not None
            else self.default_linker_fg
        )

        self.node_fg = (
            node_fg
            if node_fg is not None
            else self.default_node_fg
            if node_input is None
            else None
        )

        self.node_fg_count = (
            node_fg_count
            if node_fg_count is not None
            else self.default_node_fg_count
        )

        self.smarts_pattern = smarts_pattern
        self.optimizer_step = optimizer_step
        self.distance_threshold = distance_threshold
        self.expansion_factor = expansion_factor
        self.scale_multiplier = scale_multiplier

        # Internal state
        self._stk_cage = None
        self._ase_cage = None
        self._fg_anchors = None
        self._fg_anchor_indices = None
        self._fg_anchor_h_positions = None
        self._fg_per_linker = None
        self._linker_atom_indices = None
        self._node_centers = None

    # ------------------------------------------------------------------
    #  Subclass interface
    # ------------------------------------------------------------------

    def _build_stk_topology(
        self,
        node_bb: stk.BuildingBlock,
        linker_bb: stk.BuildingBlock,
    ):
        """
        The STK topology graph of this cage.

        Instantiates ``self._stk_topology_cls`` with the two building blocks, the
        builder's ``scale_multiplier`` and, if ``self._use_collapser``, an
        ``stk.Collapser`` optimiser with ``optimizer_step`` and
        ``distance_threshold``. Subclasses with non-standard construction override
        this method; the others only set ``_stk_topology_cls``.
        """
        if self._stk_topology_cls is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set _stk_topology_cls "
                "or override _build_stk_topology."
            )
        building_blocks = (node_bb, linker_bb)
        if node_bb.get_num_functional_groups() == linker_bb.get_num_functional_groups():
            # Two building blocks of the same topicity (e.g. Tri4Tri4): STK
            # cannot tell them apart by functional-group count. Their vertex
            # graph is bipartite, so each block gets one colour class. A
            # complete graph such as TwoPlusTwo has no such split; the first
            # half of its vertices go to the node, the second half to the
            # linker.
            try:
                node_vertices, linker_vertices = _bipartition(self._stk_topology_cls)
            except ValueError:
                n_vertices = len(self._stk_topology_cls._vertex_prototypes)
                node_vertices = tuple(range(n_vertices // 2))
                linker_vertices = tuple(range(n_vertices // 2, n_vertices))
            building_blocks = {node_bb: node_vertices, linker_bb: linker_vertices}
        kwargs: dict = {
            'building_blocks': building_blocks,
            'scale_multiplier': self.scale_multiplier,
        }
        if self._use_collapser:
            kwargs['optimizer'] = stk.Collapser(
                step_size=self.optimizer_step,
                distance_threshold=self.distance_threshold,
            )
        return self._stk_topology_cls(**kwargs)


    def _get_transformation_library(self, fg_anchors: Atoms):
        """
        The symmetry operations of this structure as permutations of its FG sites.

        A periodic structure uses its space group (spglib), translations and
        non-symmorphic operations included. A finite one uses the point group,
        detected from the atoms (``symmetry_method="detect"``) or from the
        topology's library (``"topology"``).

        Parameters
        ----------
        fg_anchors : ase.Atoms
            The site atoms, as returned by ``full_functionalisation``.

        Returns
        -------
        list of tuple
            One entry per site: its images under the non-identity operations.
        """
        ase_cage = self.to_ase()
        if ase_cage.pbc.any():
            return symmetry.spacegroup_transformation_library(
                ase_cage, self._fg_anchor_indices
            )
        if self.symmetry_method == "detect":
            return pointgroup.pointgroup_transformation_library(
                ase_cage, self._fg_anchor_indices,
                anchor_tol=self.symmetry_anchor_tol, point_group=self.point_group(),
            )
        if self.symmetry_method == "topology":
            return self._get_point_group_transformation_library(fg_anchors)
        raise ValueError(
            f"symmetry_method={self.symmetry_method!r}; use 'detect' or 'topology'."
        )

    def point_group(self) -> "pointgroup.PointGroup":
        """
        Point group of the finite structure's framework, detected from the atoms
        chosen by ``symmetry_reference`` with tolerance ``symmetry_tol``.

        ``result.name`` is the Schoenflies symbol (e.g. ``"Td"``), ``result.order``
        the number of operations and ``result.max_deviation`` how far (Angstrom) the
        structure is from perfect symmetry.

        Raises
        ------
        ValueError
            For a periodic structure (which uses spglib), or if no consistent group
            exists at this tolerance (see utils.pointgroup).

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> pg = cage.point_group()
        >>> pg.name, pg.order
        ('Td', 24)
        """
        if self._fg_anchors is None:
            self.functionalise()
        ase_cage = self.to_ase()
        if ase_cage.pbc.any():
            raise ValueError("point_group() is for finite structures; periodic "
                             "structures use their space group via spglib.")
        ref = pointgroup.reference_indices(ase_cage, self.symmetry_reference)
        return pointgroup.detect_point_group(ase_cage[ref], tol=self.symmetry_tol)

    def _get_point_group_transformation_library(self, fg_anchors: Atoms):
        """
        The point-group library of this topology (``symmetry_method="topology"``).

        The default derives the operations from the node centres with
        :func:`~cage_isomer_builder.utils.symmetry.universal_transformation_library`.
        This works when no non-identity operation fixes every node centre: four
        nodes on a tetrahedron, six on an octahedron, eight on a cube. Prism and
        bipyramid topologies (Tet2Di4, Tri2Di3, ...) and cages with more than eight
        nodes override this method.

        Parameters
        ----------
        fg_anchors : ase.Atoms
            The site atoms, as returned by ``full_functionalisation``.

        Returns
        -------
        list of tuple
            One entry per site: its images under the non-identity operations.

        Raises
        ------
        NotImplementedError
            If ``_node_centers`` is not set or holds more than eight centres.
        """
        if self._node_centers is not None and 2 <= len(self._node_centers) <= 8:
            return symmetry.universal_transformation_library(
                fg_anchors, self._node_centers
            )
        raise NotImplementedError(
            f"{type(self).__name__} must override _get_point_group_transformation_library "
            "or ensure _node_centers is populated with 2 to 8 node centres before "
            "calling enumerate_isomers()."
        )


    # ------------------------------------------------------------------
    #  Build
    # ------------------------------------------------------------------

    def _make_node_bb(self) -> stk.BuildingBlock:
        """
        Construct the node building block.

        The node may be supplied as

        * a SMILES string,
        * an ASE ``Atoms`` object, or
        * a molecular structure file readable by ASE.

        Functional groups are detected automatically using
        :func:`autodetect_building_block`.

        Returns
        -------
        stk.BuildingBlock
            STK building block representing the cage node.
        """
        return autodetect_building_block(
            mol=self.node,
            fg=self.node_fg,
            fg_count=self.node_fg_count,
            smarts_pattern=self.smarts_pattern,
            allow_single_atom=True,
            role="node",
        )


    def _make_linker_bb(self) -> stk.BuildingBlock:
        """
        Construct the linker building block.

        The linker may be supplied as

        * a SMILES string,
        * an ASE ``Atoms`` object, or
        * a molecular structure file readable by ASE.

        Functional groups are detected automatically using
        :func:`autodetect_building_block`.

        Returns
        -------
        stk.BuildingBlock
            STK building block representing the linker.
        """
        return autodetect_building_block(
            mol=self.linker,
            fg=self.linker_fg,
            fg_count=1,
            smarts_pattern=self.smarts_pattern,
            allow_single_atom=False,
            role="linker",
        )


    def _expand_cage_rigid(self) -> None:
        """
        Translate every building block rigidly outward by ``expansion_factor``.

        Each block's centroid moves to
        ``cage_centroid + expansion_factor * (block_centroid - cage_centroid)``; its
        atoms move by the same vector, so bond lengths and angles are unchanged.
        Blocks are identified from STK's atom records, not from distances, so this
        works where large nodes overlap in the initial placement.
        """
        subunits: dict[int, list[int]] = {}
        for atom_info in self._stk_cage.get_atom_infos():
            if atom_info.get_building_block() is None:
                continue
            # get_building_block_id() is unique per placement, so the four
            # placements of one node building block get four keys.
            key = atom_info.get_building_block_id()
            if key not in subunits:
                subunits[key] = []
            subunits[key].append(atom_info.get_atom().get_id())

        ase_cage, _ = stk_2_ase_atoms(self._stk_cage)
        cage_centroid = ase_cage.get_center_of_mass()
        positions = ase_cage.get_positions().copy()

        for atom_ids in subunits.values():
            subunit_centroid = positions[atom_ids].mean(axis=0)
            displacement = (self.expansion_factor - 1.0) * (subunit_centroid - cage_centroid)
            for idx in atom_ids:
                positions[idx] += displacement

        ase_cage.set_positions(positions)
        self._ase_cage = ase_cage

    def build(self) -> stk.ConstructedMolecule:
        """
        Build the cage using the configured node and linker.

        Returns
        -------
        stk.ConstructedMolecule
            Constructed STK cage.

        Notes
        -----
        Building blocks are generated automatically from the supplied node
        and linker inputs before constructing the topology graph.
        If ``expansion_factor > 1``, each subunit is translated rigidly
        outward after assembly (see :meth:`_expand_cage_rigid`).

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> cage.to_ase().get_chemical_formula()
        'C120H132O128Zr24'
        """
        node_bb = self._make_node_bb()
        linker_bb = self._make_linker_bb()

        try:
            self._stk_cage = stk.ConstructedMolecule(
                topology_graph=self._build_stk_topology(
                    node_bb=node_bb,
                    linker_bb=linker_bb,
                )
            )
        except UnboundLocalError as error:
            # stk.Collapser shrinks the cage until building blocks come
            # within distance_threshold of each other. When they already are
            # (common with bulky Zr6 nodes) its loop never runs and it fails
            # on an unassigned 'step_result'; the uncollapsed construction is
            # then the right result.
            if not self._use_collapser or 'step_result' not in str(error):
                raise
            self._use_collapser = False
            self._stk_cage = stk.ConstructedMolecule(
                topology_graph=self._build_stk_topology(
                    node_bb=node_bb,
                    linker_bb=linker_bb,
                )
            )

        if self.expansion_factor > 1.0:
            self._expand_cage_rigid()

        return self._stk_cage


    def optimise(
        self,
        fmax: float = 0.05,
        steps: int = 500,
        optimizer_cls=None,
        trajectory: Optional[str] = None,
        rigid_sbus: bool = True,
        include_charges: bool = True,
        charge: float = 0.0,
        method: str = "UFF4MOF",
        logfile: Optional[str] = "-",
    ) -> Atoms:
        """
        Optimise the cage geometry.

        * ``method="UFF4MOF"`` (default): the UFF4MOF force field with the STK bond
          orders and UFF atom types (see :mod:`cage_isomer_builder.utils.optimise`).
          The bonds are those of the construction, so atypical distances in the
          starting geometry do not change the connectivity.
        * ``method="GFN1-xTB"`` or ``"GFN2-xTB"``: tblite tight binding. It needs no
          bond orders or atom types, and so cannot be held to a bond matrix.

        Call this after :meth:`build` and before :meth:`functionalise`. After
        :meth:`functionalise`, the site markers are removed before optimising and
        :meth:`functionalise` must be run again.

        Parameters
        ----------
        fmax : float, default 0.05
            Force convergence threshold (eV/Angstrom).
        steps : int, default 500
            Maximum number of steps.
        optimizer_cls : ASE optimizer class, optional
            Any ``ase.optimize`` class (``LBFGS``, ``BFGS``, ``FIRE``, ...). Default
            ``LBFGS``.
        trajectory : str, optional
            Path of an ASE ``.traj`` file.
        rigid_sbus : bool, default True
            Keep every node and linker rigid (see
            :class:`~cage_isomer_builder.utils.rigid_body.FixRigidBodies`) and relax
            only their placement. The linker atoms bonded to a node stay free, so
            each metal-ligand bond can adjust without changing the rest of the
            linker. ``False`` relaxes every atom.
        include_charges : bool, default True
            Store partial charges on the result (``set_initial_charges``). With xTB
            they come from the optimisation. With UFF4MOF they come from one tblite
            GFN2-xTB calculation on the starting geometry and also drive UFF4MOF's
            Coulomb term. ``False`` computes no charges and runs UFF4MOF without the
            Coulomb term, as does a missing tblite (with a warning).
        charge : float, default 0.0
            Total charge passed to tblite.
        method : str, default "UFF4MOF"
            ``"UFF4MOF"``, ``"GFN1-xTB"`` or ``"GFN2-xTB"``.
        logfile : str or None, default "-"
            Optimiser log: ``"-"`` for the terminal, a file path, or ``None`` for no
            output.

        Returns
        -------
        ase.Atoms
            The optimised structure, which also replaces the builder's structure.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> relaxed = cage.optimise(include_charges=False, logfile=None)
        >>> len(relaxed), relaxed is cage.to_ase()
        (404, True)
        """
        from cage_isomer_builder.utils.rigid_body import FixRigidBodies

        if self._stk_cage is not None:
            # Start from the STK cage so that site markers are never included.
            ase_atoms, bond_matrix = stk_2_ase_atoms(self._stk_cage)
        else:
            ase_atoms = self.to_ase().copy()
            # Remove the site markers, which have no force-field parameters.
            from cage_isomer_builder.utils.sites import site_indices
            markers = set(site_indices(ase_atoms))
            ase_atoms = ase_atoms[[i for i in range(len(ase_atoms)) if i not in markers]]
            bond_matrix = None

        if rigid_sbus:
            if self._stk_cage is None:
                raise RuntimeError(
                    "rigid_sbus=True requires the STK cage to identify SBU "
                    "groupings (unavailable after load_from_file()). Pass "
                    "rigid_sbus=False to relax every atom independently instead."
                )
            bb_atoms, bb_is_node = self._classify_atoms_by_bb(ase_atoms)

            # Nodes stay fully rigid. Each linker atom bonded to a node is
            # left out of its linker's rigid group, so the metal-ligand bond
            # can adjust.
            node_bb_ids = {bb_id for bb_id, is_node in bb_is_node.items() if is_node}
            node_atom_ids = {idx for bb_id in node_bb_ids for idx in bb_atoms[bb_id]}
            donor_atom_ids = {
                i
                for i in range(len(ase_atoms))
                if i not in node_atom_ids
                for j in np.nonzero(bond_matrix[i])[0]
                if j in node_atom_ids
            }

            groups = [bb_atoms[bb_id] for bb_id in node_bb_ids] + [
                [idx for idx in atoms if idx not in donor_atom_ids]
                for bb_id, atoms in bb_atoms.items()
                if bb_id not in node_bb_ids
            ]
            groups = [g for g in groups if len(g) > 1]
            ase_atoms.set_constraint(FixRigidBodies.from_atoms(ase_atoms, groups))

        if method.upper() == "UFF4MOF":
            if bond_matrix is None:
                raise RuntimeError(
                    "method='UFF4MOF' requires the STK bond matrix (unavailable "
                    "after load_from_file()). Pass method='GFN2-xTB' instead."
                )
            from cage_isomer_builder.utils.optimise import run_uff4mof_optimisation

            charges = None
            if include_charges:
                from cage_isomer_builder.utils.tblite_charges import (
                    predict_tblite_charges,
                    tblite_available,
                )

                if tblite_available():
                    # Computed once from the starting geometry and held fixed,
                    # as is usual for UFF with QEq charges; they drive the
                    # Coulomb term.
                    charges = predict_tblite_charges(ase_atoms, charge=charge)
                else:
                    warnings.warn(
                        "tblite is not installed, so no partial charges are "
                        "computed and UFF4MOF runs without its Coulomb term. "
                        "Install it with 'pip install cage_isomer_builder[xtb]', "
                        "or pass include_charges=False to silence this warning."
                    )

            optimised = run_uff4mof_optimisation(
                atoms=ase_atoms,
                bond_matrix=bond_matrix,
                fmax=fmax,
                steps=steps,
                optimizer_cls=optimizer_cls,
                trajectory=trajectory,
                charges=charges,
                logfile=logfile,
            )
        else:
            from cage_isomer_builder.utils.tblite_optimise import run_tblite_optimisation

            optimised, charges = run_tblite_optimisation(
                atoms=ase_atoms,
                fmax=fmax,
                steps=steps,
                optimizer_cls=optimizer_cls,
                trajectory=trajectory,
                charge=charge,
                method=method,
                logfile=logfile,
            )

        if include_charges and charges is not None:
            optimised.set_initial_charges(charges)

        self._ase_cage = optimised
        return optimised


    # ------------------------------------------------------------------
    #  Load existing structure
    # ------------------------------------------------------------------

    @classmethod
    def load_from_file(cls, path: str, primitive: bool = False,
                       symprec: float = 1e-2, **init_kwargs):
        """
        Create a builder from an existing structure instead of building one with
        STK. The structure can then be functionalised, analysed and enumerated.

        Parameters
        ----------
        path : str or pathlib.Path
            Structure file readable by ASE (PDB, CIF, XYZ, MOL, SDF, ...).
        primitive : bool, default False
            Replace a periodic cell by its primitive cell (see
            :func:`~cage_isomer_builder.utils.symmetry.primitive_cell`). A CIF often
            holds several primitive cells (four for a face-centred cell) and each
            multiplies the number of isomers. Every isomer then repeats in every
            primitive cell.
        symprec : float, default 1e-2
            spglib tolerance used when ``primitive=True``.
        **init_kwargs
            Passed to the constructor.

        Returns
        -------
        CageBuilder

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import CageBuilder
        >>> len(CageBuilder.load_from_file(example_data.path("RUBTAK01")).to_ase())
        440
        >>> len(CageBuilder.load_from_file(example_data.path("RUBTAK01"), primitive=True).to_ase())
        110
        """
        instance = cls.__new__(cls)
        CageBuilder.__init__(instance, **init_kwargs)
        atoms = read(path)
        if primitive:
            atoms = symmetry.primitive_cell(atoms, symprec=symprec)
        instance._ase_cage = atoms
        return instance

    # ------------------------------------------------------------------
    #  Conversion
    # ------------------------------------------------------------------

    def to_ase(self, hydrogens: bool = False) -> Atoms:
        """
        The structure as ASE Atoms (converted from the STK cage on first use).

        After :meth:`functionalise` it holds the R-site markers (``X`` atoms).
        ``hydrogens=True`` returns a copy with every site turned back into its
        original H, i.e. the parent cage, for docking or optimisation.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> atoms = cage.to_ase()
        >>> len(atoms), bool(atoms.pbc.any())
        (404, False)
        >>> _ = cage.functionalise()
        >>> "X" in cage.to_ase().get_chemical_symbols(), "X" in cage.to_ase(hydrogens=True).symbols
        (True, False)
        """
        if self._ase_cage is None:
            if self._stk_cage is None:
                raise RuntimeError(
                    "No structure available. Call build() or load_from_file() first."
                )
            self._ase_cage, _ = stk_2_ase_atoms(self._stk_cage)
        if not hydrogens or self._fg_anchor_indices is None:
            return self._ase_cage
        from cage_isomer_builder.utils.sites import clear_site

        parent = self._ase_cage.copy()
        for slot, atom in enumerate(self._fg_anchor_indices):
            clear_site(parent, atom, "H", self._fg_anchor_h_positions[slot])
        return parent

    def get_bond_matrix(self) -> np.ndarray:
        """
        Bond-order matrix of the built cage, taken from the STK construction rather
        than perceived from the geometry. It is correct even where bonded atoms are
        far apart, and it adds no bonds between atoms that are merely close. Used by
        the GULP and AMS writers and accepted by
        ``functionalise.place_guest_in_host(host_bond_matrix=...)`` and
        ``functionalise_isomer_sites(host_bond_matrix=...)``.

        Returns
        -------
        np.ndarray, shape (n_atoms, n_atoms)

        Raises
        ------
        RuntimeError
            If the structure was not built with :meth:`build` (a structure from
            :meth:`load_from_file` has no bond orders).

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> bonds = cage.get_bond_matrix()
        >>> bonds.shape, bool((bonds == bonds.T).all())
        ((404, 404), True)
        """
        if self._stk_cage is None:
            raise RuntimeError(
                "get_bond_matrix() requires the STK cage. Call build() "
                "first; a structure loaded via load_from_file() has no "
                "STK-derived bond order information."
            )
        _, bond_matrix = stk_2_ase_atoms(self._stk_cage)
        return bond_matrix

    def to_smiles(self) -> str:
        """
        Convert the cage to a SMILES string using mofstructure (OpenBabel).

        Returns
        -------
        smiles : str

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> "Zr" in cage.to_smiles()
        True
        """
        return ase_2_pybel(self.to_ase()).write('smi').strip()

    # ------------------------------------------------------------------
    #  Functionalisation
    # ------------------------------------------------------------------

    def _classify_atoms_by_bb(self, ase_cage):
        """
        Group atom indices by building-block placement, from STK's atom records.

        ``get_building_block_id()`` is unique per placement, so the four
        placements of one node building block get four keys. A placement is a node
        if it contains a metal, otherwise a linker.

        Parameters
        ----------
        ase_cage : ase.Atoms

        Returns
        -------
        bb_atoms : dict[int, list[int]]
            Placement id -> atom indices.
        bb_is_node : dict[int, bool]
            Placement id -> True if the placement contains a metal.
        """
        tm_set = _metal_symbols()
        bb_atoms: dict[int, list[int]] = {}
        bb_is_node: dict[int, bool] = {}
        for atom_info in self._stk_cage.get_atom_infos():
            if atom_info.get_building_block() is None:
                continue
            bb_id = atom_info.get_building_block_id()
            atom_id = atom_info.get_atom().get_id()
            if bb_id not in bb_atoms:
                bb_atoms[bb_id] = []
                bb_is_node[bb_id] = False
            bb_atoms[bb_id].append(atom_id)
            if ase_cage[atom_id].symbol in tm_set:
                bb_is_node[bb_id] = True
        return bb_atoms, bb_is_node

    def _get_sbus_from_stk(self, ase_cage):
        """
        Linker and node atom indices from STK's atom records. Works where building
        blocks overlap, because it does not rely on distances.

        Returns
        -------
        linker_atom_indices : list[list[int]]
            One list per linker placement.
        metal_atom_indices : list[int]
            All atoms of node placements.
        """
        bb_atoms, bb_is_node = self._classify_atoms_by_bb(ase_cage)
        linker_atom_indices: list[list[int]] = []
        metal_atom_indices: list[int] = []
        for bb_id, atom_ids in bb_atoms.items():
            if bb_is_node[bb_id]:
                metal_atom_indices.extend(atom_ids)
            else:
                linker_atom_indices.append(atom_ids)
        return linker_atom_indices, metal_atom_indices

    def _get_node_centers_from_stk(self, ase_cage):
        """
        One centre per node placement: the mean of the node atoms bonded to
        linkers. For a single-atom node this is the metal itself; for a Zr6 node
        with capping groups it is the centre of its connection sites, independent
        of the caps. Falls back to the metal centroid if a node has no bond to a
        linker.

        Returns
        -------
        node_centers : np.ndarray, shape (n_nodes, 3), or None
        """
        bb_atoms, bb_is_node = self._classify_atoms_by_bb(ase_cage)

        # Reverse map: cage atom id -> building-block id
        atom_to_bb = {
            atom_id: bb_id
            for bb_id, atom_ids in bb_atoms.items()
            for atom_id in atom_ids
        }

        # For each node placement, collect the node-side atoms at inter-BB bonds
        node_ext: dict[int, list[int]] = {
            bb_id: []
            for bb_id, is_node in bb_is_node.items()
            if is_node
        }
        for bond in self._stk_cage.get_bonds():
            id1 = bond.get_atom1().get_id()
            id2 = bond.get_atom2().get_id()
            bb1 = atom_to_bb.get(id1)
            bb2 = atom_to_bb.get(id2)
            if bb1 is None or bb2 is None or bb1 == bb2:
                continue
            # Cross-BB bond: collect the node-side atom
            if bb_is_node.get(bb1, False) and not bb_is_node.get(bb2, False):
                node_ext[bb1].append(id1)
            elif bb_is_node.get(bb2, False) and not bb_is_node.get(bb1, False):
                node_ext[bb2].append(id2)

        tm_set = _metal_symbols()
        centers = []
        for bb_id, ext_ids in node_ext.items():
            if ext_ids:
                centers.append(ase_cage.positions[ext_ids].mean(axis=0))
            else:
                # Fallback: centroid of metal atoms only
                metal_ids = [i for i in bb_atoms[bb_id] if ase_cage[i].symbol in tm_set]
                ref_ids = metal_ids if metal_ids else bb_atoms[bb_id]
                centers.append(ase_cage.positions[ref_ids].mean(axis=0))

        return np.array(centers) if centers else None

    def functionalise(self, aromatic_only: bool = True):
        """
        Find the functional-group sites: every aromatic C-H of every linker becomes
        an R site (all linker types of a MOF with several, see :meth:`slot_layout`).
        ``aromatic_only=False`` also marks H on alkyl carbons.

        Returns
        -------
        fg_anchors : ase.Atoms
            The site atoms.
        fg_anchor_indices : list of int
            Index of each site atom in the structure.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)
        >>> anchors, slots = cage.functionalise()
        >>> len(slots), set(cage.to_ase()[slots].get_chemical_symbols())   # 4 slots on each of 6 rings
        (24, {'X'})
        """
        # Mark the sites on a copy and keep that as the builder's structure.
        functionalised = self.to_ase().copy()

        # For a built cage, building blocks come from STK's atom records,
        # which work even where blocks overlap in the initial placement.
        if self._stk_cage is not None:
            linker_atom_indices, metal_atom_indices = self._get_sbus_from_stk(functionalised)
            self._node_centers = self._get_node_centers_from_stk(functionalised)
            linker_types = [0] * len(linker_atom_indices)
            type_names = [functionalised[linker_atom_indices[0]].get_chemical_formula()]
        else:
            linker_atom_indices, linker_types, type_names, metal_atom_indices = \
                find_linkers(functionalised)
            self._node_centers = get_node_centers(functionalised, metal_atom_indices)

        self._linker_atom_indices = linker_atom_indices
        anchors, anchor_indices, h_positions = full_functionalisation(
            functionalised, sum(linker_atom_indices, []), aromatic_only=aromatic_only,
        )
        if not anchor_indices:
            raise ValueError("no aromatic C-H on any linker: nothing to functionalise.")
        self._set_slot_layout(functionalised, anchor_indices, h_positions,
                              linker_types, type_names)
        self._check_assembly()
        self._ase_cage = functionalised

        return self._fg_anchors, self._fg_anchor_indices

    def _set_slot_layout(self, atoms, anchor_indices, h_positions, linker_types, type_names):
        """
        Group the FG slots linker by linker and record the slot layout.

        Slots are put in consecutive blocks, one per linker, linkers in
        order of first appearance (so an already grouped list keeps its
        numbering). Linkers without any slot (e.g. a bridging cyanide or a
        saturated pillar) carry no groups and are left out of the layout.
        """
        owner = {a: k for k, lk in enumerate(self._linker_atom_indices) for a in lk}
        first_seen = {}
        for slot, atom in enumerate(anchor_indices):
            first_seen.setdefault(owner[atom], slot)
        order = sorted(range(len(anchor_indices)),
                       key=lambda s: (first_seen[owner[anchor_indices[s]]], s))
        self._fg_anchor_indices = [int(anchor_indices[s]) for s in order]
        self._fg_anchor_h_positions = np.asarray(h_positions)[order]
        self._fg_anchors = atoms[self._fg_anchor_indices]

        slot_linkers = sorted(first_seen, key=first_seen.get)
        sizes = [sum(1 for a in self._fg_anchor_indices if owner[a] == L) for L in slot_linkers]
        used_types = sorted({linker_types[L] for L in slot_linkers})
        compact = {t: k for k, t in enumerate(used_types)}
        layout = rgroup.SlotLayout(
            sizes=tuple(sizes),
            types=tuple(compact[linker_types[L]] for L in slot_linkers),
            type_names=tuple(type_names[t] for t in used_types),
        )
        for t in range(layout.n_types):
            layout.type_size(t)          # raises if one type has varying slot counts
        self._slot_linkers = slot_linkers
        self._layout = layout
        self._fg_per_linker = layout.uniform_size

    def slot_layout(self) -> "rgroup.SlotLayout":
        """
        How the FG slots are grouped: slots per linker and linker type
        (``layout.type_names`` are the linkers' chemical formulas). A MOF with
        several kinds of linker has several types, possibly with different
        numbers of slots; groups can then be given per type, e.g.
        ``groups={"C6H4": "A", "C10H8N2": ""}``.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import CageBuilder
        >>> mof = CageBuilder.load_from_file(example_data.path("ABIXEQ"))   # two kinds of linker
        >>> _ = mof.functionalise()
        >>> layout = mof.slot_layout()
        >>> layout.type_names, layout.sizes
        (('C18H18N4', 'C6H4'), (8, 4))
        """
        if self._fg_anchors is None:
            self.functionalise()
        return self._layout

    # ------------------------------------------------------------------
    #  Isomer enumeration
    # ------------------------------------------------------------------

    def enumerate_isomers(
        self,
        output_path: str = None,
        generate_files: bool = True,
        limit: int = None,
    ) -> list:
        """
        Enumerate the symmetry-unique isomers with one group on every linker.

        Each candidate is checked against its own symmetry orbit, so memory use does
        not grow with the number of linkers; the run time grows with the number of
        raw placements unless ``limit`` stops early.

        Parameters
        ----------
        output_path : str, optional
            Directory for one extended XYZ file per isomer. Nothing is written if
            None or if ``generate_files`` is False.
        generate_files : bool, default True
        limit : int, optional
            Stop after this many isomers. Canonical representatives occur about
            once per |symmetry group| raw placements, so a modest limit is fast even
            when the full enumeration is not.

        Returns
        -------
        list of list of int
            The active site indices of each isomer.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> import os, tempfile
        >>> folder = tempfile.mkdtemp()
        >>> isomers = cage.enumerate_isomers(output_path=folder, limit=3)
        >>> isomers[0], sorted(os.listdir(folder))[0]
        ([0, 4, 8, 12, 16, 20], '0-4-8-12-16-20.xyz')
        """
        if self._fg_anchors is None:
            self.functionalise()

        if generate_files and output_path:
            os.makedirs(output_path, exist_ok=True)

        if self._fg_per_linker is None:
            # Linkers with different numbers of slots: the same isomers come
            # from the R-group enumeration with one group per linker.
            descriptors = (
                [i for i, v in enumerate(iso.site_labels) if v]
                for iso in rgroup.iter_rgroup_isomers(
                    self._slot_group(), len(self._fg_anchors), self._layout,
                    groups="A", limit=limit)
            )
        else:
            fg_per_linker = self._fg_per_linker
            n_linkers = len(self._fg_anchors) // fg_per_linker
            tl = self._get_transformation_library(self._fg_anchors.copy())
            descriptors = (d for _, d in iter_unique_isomers(tl, n_linkers, fg_per_linker,
                                                              limit=limit))

        unique_isomers = []
        for descriptor in descriptors:
            unique_isomers.append(descriptor)

            if generate_files and output_path:
                generate_isomer_structure_file(
                    self._ase_cage,
                    self._fg_anchors,
                    descriptor,
                    self._fg_anchor_indices,
                    output_path,
                    fg_anchor_h_positions=self._fg_anchor_h_positions,
                )

        return unique_isomers

    def count_unique_isomers_burnside(self, active_per_linker: int = 1) -> int:
        """
        Number of symmetry-unique isomers, from Burnside's lemma, without
        enumerating them.

        Parameters
        ----------
        active_per_linker : int, default 1
            Number of groups on every linker: 1 gives the isomers of
            :meth:`enumerate_isomers`, 2 the ways of placing two identical groups on
            every linker, and so on. ``k`` and ``fg_per_linker - k`` give the same
            count.

        Returns
        -------
        int

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> cage.count_unique_isomers_burnside(), cage.count_unique_isomers_burnside(active_per_linker=2)
        (176, 2013)
        """
        # k identical groups per linker is groups="A" * k. The linker-cycle
        # formula of utils.rgroup stays fast for large periodic cells, where
        # the bitmask tables of utils.util do not.
        if self._fg_anchors is None:
            self.functionalise()
        fg_per_linker = min(self._layout.sizes)
        if not 0 <= active_per_linker <= fg_per_linker:
            raise ValueError(
                f"active_per_linker={active_per_linker} must be between 0 and "
                f"fg_per_linker={fg_per_linker}."
            )
        return self.count_rgroup_isomers(groups="A" * active_per_linker)

    # ------------------------------------------------------------------
    #  Several distinct groups at once (R1, R2, ...)
    # ------------------------------------------------------------------

    def _check_slot_layout(self) -> None:
        """
        Check that the FG sites sit linker by linker as the slot layout says.

        Raises
        ------
        ValueError
            If a block of sites of the layout does not belong to one linker.
        """
        owner = {
            atom: linker
            for linker, atoms in enumerate(self._linker_atom_indices)
            for atom in atoms
        }
        owners = [owner.get(i) for i in self._fg_anchor_indices]
        for L, (start, size) in enumerate(zip(self._layout.offsets, self._layout.sizes)):
            if set(owners[start:start + size]) != {self._slot_linkers[L]}:
                raise ValueError(
                    "FG slots are not grouped linker by linker as the slot layout "
                    "says, so per-linker isomer counts would be wrong."
                )

    def slot_group(self) -> "rgroup.SlotGroup":
        """
        The structure's symmetry as permutations of its functional-group sites,
        checked to be a group that maps linkers onto linkers of the same type.
        Every isomer count and enumeration uses it; pass it to the functions of
        :mod:`cage_isomer_builder.utils.rgroup` and
        :mod:`cage_isomer_builder.utils.ensemble`.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import CageBuilder
        >>> mof = CageBuilder.load_from_file(example_data.path("RUBTAK01"), primitive=True)
        >>> _ = mof.functionalise()
        >>> group = mof.slot_group()
        >>> group.order, group.perms.shape            # Fm-3m: 48 operations on 24 sites
        (48, (48, 24))
        """
        return self._slot_group()

    def _slot_group(self) -> "rgroup.SlotGroup":
        """Validated symmetry group on the FG slots (see rgroup.prepare_group)."""
        if self._fg_anchors is None:
            self.functionalise()
        n_fg_slots = len(self._fg_anchors)
        self._check_slot_layout()
        tl = self._get_transformation_library(self._fg_anchors.copy())
        return rgroup.prepare_group(list(zip(*tl)), n_fg_slots, self._layout)

    def count_rgroup_isomers(
        self,
        groups: Union[str, list] = "AB",
        linker_patterns: Optional[dict] = None,
    ) -> int:
        """
        Number of symmetry-unique isomers with several different groups placed at
        once, from Burnside's lemma.

        Parameters
        ----------
        groups : str, list or dict, default "AB"
            Groups on every linker: ``"AB"`` one A and one B, ``"AAB"`` two A and
            one B, ``"A"`` one group per linker. A string is read one capital letter
            per group; give longer names as a list, e.g. ``["NH2", "OH"]``. With
            several linker types, a dict ``{type name: groups}`` gives each type its
            own groups (e.g. ``{"C6H4": "A", "C10H8N2": ""}``).
        linker_patterns : dict, optional
            Fixed numbers of linkers per pattern instead: ``{"A": 3, "B": 3}``
            (three linkers with A, three with B) or ``{"AB": 2, "": 4}`` (four bare
            linkers). With several linker types, ``{type name: {pattern: number}}``.
            Overrides ``groups``.

        Returns
        -------
        int

        Raises
        ------
        ValueError
            If the specification does not fit the linkers, or the symmetry
            operations do not form a group on the sites.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> cage.count_rgroup_isomers("AB")                    # one A and one B on every linker
        124464
        >>> cage.count_rgroup_isomers(linker_patterns={"A": 3, "B": 3})   # 50:50 linkers
        3424
        """
        group = self._slot_group()
        return rgroup.count_rgroup_isomers(
            group, len(self._fg_anchors), self._layout,
            groups=groups, linker_patterns=linker_patterns,
        )

    def enumerate_rgroup_isomers(
        self,
        groups: Union[str, list] = "AB",
        linker_patterns: Optional[dict] = None,
        output_path: str = None,
        generate_files: bool = True,
        limit: int = None,
    ) -> list:
        """
        Enumerate the symmetry-unique isomers with several different groups.

        Each isomer written to ``output_path`` is an extended XYZ file in which every
        active site is an R site (``X`` atom) whose per-atom ``"rgroup"`` value says
        which group goes there (1 = first label, 2 = second, ...).
        :func:`~cage_isomer_builder.utils.functionalise.functionalise_isomer_sites`
        attaches the matching fragment at each site. A complete enumeration
        (``limit=None``) is checked against :meth:`count_rgroup_isomers`.

        Parameters
        ----------
        groups, linker_patterns
            As for :meth:`count_rgroup_isomers`.
        output_path : str, optional
            Directory for the structure files.
        generate_files : bool, default True
        limit : int, optional
            Stop after this many isomers.

        Returns
        -------
        list of RGroupIsomer
            ``isomer.to_dict()`` gives ``{"A": [sites], "B": [sites], ...}``.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> first = cage.enumerate_rgroup_isomers(["NH2", "OH"], generate_files=False, limit=1)[0]
        >>> first.to_dict()
        {'NH2': [2, 6, 10, 14, 18, 22], 'OH': [3, 7, 11, 15, 19, 23]}
        """
        group = self._slot_group()
        n_fg_slots = len(self._fg_anchors)
        isomers = rgroup.enumerate_rgroup_isomers(
            group, n_fg_slots, self._layout,
            groups=groups, linker_patterns=linker_patterns, limit=limit,
        )
        if generate_files and output_path:
            os.makedirs(output_path, exist_ok=True)
            for isomer in isomers:
                rgroup.write_rgroup_isomer_file(
                    self._ase_cage,
                    self._fg_anchors,
                    isomer,
                    self._fg_anchor_indices,
                    output_path,
                    fg_anchor_h_positions=self._fg_anchor_h_positions,
                )
        return isomers

    # ------------------------------------------------------------------
    #  Statistics
    # ------------------------------------------------------------------

    def get_statistics(self, n_kde_bins: int = 200, threshold: float = 0.5,
                       max_distance: Optional[float] = None) -> CageStatistics:
        """
        FG-FG distance statistics: every pair of sites on different linkers, once,
        split by where the sites point, with a smoothed curve.

        A site is endo or exo by the direction of its C-site bond relative to the
        pore centre (see :func:`cage_isomer_builder.utils.features.orientation_class`):
        the cosine with the inward direction is above ``threshold`` (endo) or below
        ``-threshold`` (exo); otherwise the site is tangential and its pairs are
        counted in ``other_pairs``. The counts are proportional to the average over
        all placements of one group per linker. For several groups, ratios, defects
        or flexibility, use :meth:`get_ensemble_descriptor`.

        Parameters
        ----------
        n_kde_bins : int, default 200
            Number of points of the smoothed curve.
        threshold : float, default 0.5
        max_distance : float, optional
            Periodic structures: count pairs over all periodic images up to this
            distance, per unit cell (default 20 Angstrom).

        Returns
        -------
        CageStatistics

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> stats = cage.get_statistics()
        >>> sum(stats.total_fg_pairs), sum(stats.other_pairs)    # face-on rings: tangential sites
        (240, 240)
        """
        if self._fg_anchors is None:
            self.functionalise()

        n_fg_slots = len(self._fg_anchors)
        n_linkers = self._layout.n_linkers
        n_unique = self.count_unique_isomers_burnside()

        slots = self.slot_features(threshold=threshold)
        periodic = bool(self.to_ase().pbc.any())
        if periodic and max_distance is None:
            max_distance = 20.0
        desc = distributions.pair_descriptor(
            slots, by="orientation", max_distance=max_distance if periodic else None,
        )
        distance_keys, total = desc.histogram()
        if len(distance_keys) == 0:
            raise ValueError("no FG pairs between different linkers.")

        def column(cls_a, cls_b):
            key = tuple(sorted((f"fg:R1:{cls_a}", f"fg:R1:{cls_b}")))
            values, weights = desc.histogram([key]) if key in desc.pairs else ([], [])
            table = dict(zip(np.round(values, 2), weights))
            return [table.get(d, 0.0) for d in distance_keys]

        endo_endo = column("endo", "endo")
        endo_exo = column("endo", "exo")
        exo_exo = column("exo", "exo")
        other = [t - a - b - c for t, a, b, c in zip(total, endo_endo, endo_exo, exo_exo)]

        def as_int(x):
            return [int(round(v)) if abs(v - round(v)) < 1e-9 else float(v) for v in x]

        total, endo_endo, endo_exo, exo_exo, other = map(
            as_int, (total, endo_endo, endo_exo, exo_exo, other))
        kde_x, kde_y = kernel_density_estimation(
            list(distance_keys), [int(round(t)) for t in total], n_kde_bins)
        kde_x = np.asarray(kde_x).ravel()

        return CageStatistics(
            n_linkers=n_linkers,
            n_fg_slots=n_fg_slots,
            n_unique_isomers=n_unique,
            distance_keys=[float(d) for d in distance_keys],
            total_fg_pairs=total,
            endo_endo_pairs=endo_endo,
            endo_exo_pairs=endo_exo,
            exo_exo_pairs=exo_exo,
            kde_x=kde_x,
            kde_y=kde_y,
            other_pairs=other,
        )

    # ------------------------------------------------------------------
    #  Structure checks and building blocks
    # ------------------------------------------------------------------

    def _check_assembly(self, factor: float = 1.6) -> None:
        """
        Warn if the building blocks of a built structure are not bonded to each
        other, i.e. an inter-block bond is longer than ``factor`` times the sum of
        the covalent radii. A bulky node next to a short linker often comes out of
        STK placement this way (see ``scale_multiplier``). Symmetry, isomer counts
        and distances of such a geometry do not describe a real structure.
        """
        if self._stk_cage is None:
            return
        from ase.data import covalent_radii

        atoms, bond_matrix = stk_2_ase_atoms(self._stk_cage)
        current = self.to_ase()
        if len(current) == len(atoms):
            # The STK molecule keeps its construction geometry; after
            # optimise() the current geometry is self._ase_cage.
            atoms = atoms.copy()
            atoms.positions = current.positions
        bb_atoms, _ = self._classify_atoms_by_bb(atoms)
        owner = {i: k for k, v in bb_atoms.items() for i in v}
        worst = 0.0
        for i, j in zip(*np.nonzero(np.triu(bond_matrix))):
            if owner.get(i) is None or owner.get(i) == owner.get(j):
                continue
            ratio = atoms.get_distance(i, j) / (covalent_radii[atoms.numbers[i]]
                                                + covalent_radii[atoms.numbers[j]])
            worst = max(worst, ratio)
        if worst > factor:
            warnings.warn(
                f"building blocks are not bonded to each other: the longest "
                f"inter-building-block bond is {worst:.1f}x its covalent length. "
                "Run optimise() before functionalise(), or lower scale_multiplier "
                "- symmetry, isomer counts and distances of this geometry are not "
                "meaningful."
            )

    def building_blocks(self) -> tuple:
        """
        Atom indices of every linker and every node.

        Returns
        -------
        linkers : list of list of int
            In the order used for FG slots (``self._linker_atom_indices``).
        nodes : list of list of int
            One list per node (metal cluster with whatever is bonded to it
            outside the linkers).

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> linkers, nodes = cage.building_blocks()
        >>> len(linkers), len(nodes), len(nodes[0])
        (6, 4, 86)
        """
        if self._fg_anchors is None:
            self.functionalise()
        atoms = self.to_ase()
        if self._stk_cage is not None:
            bb_atoms, bb_is_node = self._classify_atoms_by_bb(atoms)
            nodes = [list(v) for k, v in bb_atoms.items() if bb_is_node[k]]
            return [list(lk) for lk in self._linker_atom_indices], nodes
        in_linker = {i for lk in self._linker_atom_indices for i in lk}
        rest = [i for i in range(len(atoms)) if i not in in_linker]
        adjacency = self.adjacency()
        rest_set, seen, nodes = set(rest), set(), []
        for start in rest:
            if start in seen:
                continue
            comp, stack = [], [start]
            seen.add(start)
            while stack:
                a = stack.pop()
                comp.append(a)
                for b in adjacency[a]:
                    if b in rest_set and b not in seen:
                        seen.add(b)
                        stack.append(b)
            nodes.append(sorted(comp))
        # Metal-metal contacts are not in the bond graph, so merge components
        # whose metals are within 3.6 Angstrom of each other.
        return [list(lk) for lk in self._linker_atom_indices], self._merge_metal_clusters(atoms, nodes)

    @staticmethod
    def _merge_metal_clusters(atoms, groups, cutoff=3.6):
        metal_of = {}
        for k, g in enumerate(groups):
            for i in g:
                if features.is_metal(atoms.numbers[i]):
                    metal_of[i] = k
        parent = list(range(len(groups)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        metals = list(metal_of)
        for a in range(len(metals)):
            for b in range(a + 1, len(metals)):
                if atoms.get_distance(metals[a], metals[b], mic=bool(atoms.pbc.any())) < cutoff:
                    parent[find(metal_of[metals[a]])] = find(metal_of[metals[b]])
        merged = {}
        for k, g in enumerate(groups):
            merged.setdefault(find(k), []).extend(g)
        # A group without a metal (e.g. a solvent) stays on its own.
        return [sorted(g) for g in merged.values()]

    def adjacency(self) -> dict:
        """
        Bond graph of the current structure: from the STK bond matrix for a built
        structure (whatever its geometry), otherwise from distances
        (:func:`cage_isomer_builder.utils.features.bond_graph`).

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> graph = cage.adjacency()                           # from the STK bond matrix
        >>> zr = next(i for i, a in enumerate(cage.to_ase()) if a.symbol == "Zr")
        >>> len(graph[zr])                                     # 8 oxygens around each Zr
        8
        """
        atoms = self.to_ase()
        if self._stk_cage is not None:
            _, bond_matrix = stk_2_ase_atoms(self._stk_cage)
            if len(bond_matrix) == len(atoms):
                return features.adjacency_from_bond_matrix(atoms, bond_matrix)
        return features.bond_graph(atoms)

    def pore_centre(self) -> Optional[np.ndarray]:
        """
        Centre of the pore: the point-group centre of a finite structure
        (centroid of its non-H, non-site atoms if no group is found), None
        for a periodic one.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> import numpy as np
        >>> bool(np.allclose(cage.pore_centre(), cage.point_group().centre))
        True
        """
        atoms = self.to_ase()
        if atoms.pbc.any():
            return None
        try:
            return np.asarray(self.point_group().centre)
        except ValueError:
            heavy = [i for i in range(len(atoms)) if atoms.numbers[i] > 1]
            return atoms.positions[heavy].mean(axis=0)

    # ------------------------------------------------------------------
    #  Feature sites and descriptors
    # ------------------------------------------------------------------

    def slot_features(self, threshold: float = 0.5, centre=None) -> "features.FeatureSet":
        """
        One ``fg`` feature per FG slot, in slot order, with its C-site
        direction and endo/exo/tangential class relative to
        :meth:`pore_centre`. The owner of a slot is the index of the linker
        holding it in :meth:`building_blocks`, so it matches the owners of
        :meth:`get_features`' other sites.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> slots = cage.slot_features()
        >>> len(slots), sorted(set(slots.orientation.tolist()))
        (24, ['tangential'])
        """
        if self._fg_anchors is None:
            self.functionalise()
        atoms = self.to_ase()
        if centre is None:
            centre = self.pore_centre()
        fs = features.fg_features(atoms, self._fg_anchor_indices, centre=centre,
                                  threshold=threshold, adjacency=self.adjacency())
        linker_of = {a: k for k, lk in enumerate(self._linker_atom_indices) for a in lk}
        fs.owners = np.array([linker_of.get(i, -1) for i in self._fg_anchor_indices])
        if (fs.owners < 0).any():
            fs.owners = np.array(self._slot_linkers)[self._layout.slot_linker]
        return fs

    def get_features(self, kinds=("fg", "pi", "donor", "acceptor", "open_metal"),
                     expected_cn: Optional[dict] = None, include_metal_bound: bool = False,
                     threshold: float = 0.5,
                     open_metal_method: str = "auto") -> "features.FeatureSet":
        """
        Every feature site of the structure: FG slots, aromatic rings
        (with normals), H-bond donors/acceptors (with directions) and open
        metal sites (with the direction of the empty site). Owners are the
        building blocks (linkers first, then nodes), so pairs inside one
        building block can be excluded. Open metal sites come from
        mofstructure's geometric detector unless ``open_metal_method=
        "coordination"``. See :mod:`cage_isomer_builder.utils.features`.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> cage.get_features().counts()            # R sites, rings, mu3-O acceptors; no open metals
        {'acceptor': 32, 'fg': 24, 'pi': 6}
        """
        if self._fg_anchors is None:
            self.functionalise()
        atoms = self.to_ase()
        linkers, nodes = self.building_blocks()
        out = features.detect_features(
            atoms, kinds=[k for k in kinds if k != "fg"], owners=linkers + nodes,
            expected_cn=expected_cn, include_metal_bound=include_metal_bound,
            adjacency=self.adjacency(), open_metal_method=open_metal_method,
        )
        if "fg" in kinds:
            fs = self.slot_features(threshold=threshold)
            out = fs + out
        return out

    def get_descriptor(self, kinds=("fg", "pi", "donor", "acceptor", "open_metal"),
                       by: str = "kind", max_distance: Optional[float] = None,
                       exclude_same_owner: bool = True, **feature_kwargs
                       ) -> "distributions.Descriptor":
        """
        Pair descriptor of every feature site of this (static) structure,
        including all cross terms between kinds. For a periodic structure
        pairs are counted over all images up to ``max_distance`` (default
        20 Angstrom). See :mod:`cage_isomer_builder.utils.distributions`.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> desc = cage.get_descriptor()
        >>> desc.summary()[("fg", "pi")]            # 24 sites x 6 rings, minus each ring's own 4
        (120, 120.0)
        """
        fs = self.get_features(kinds=kinds, **feature_kwargs)
        if self.to_ase().pbc.any() and max_distance is None:
            max_distance = 20.0
        return distributions.pair_descriptor(fs, by=by, max_distance=max_distance,
                                             exclude_same_owner=exclude_same_owner)

    def get_ensemble_descriptor(self, groups="A", linker_patterns: Optional[dict] = None,
                                n_missing_linkers: int = 0, mode: str = "fixed",
                                fractions: Optional[dict] = None, by: str = "orientation",
                                include_same_linker: bool = False,
                                rotation_angles=None, angle_weights=None,
                                threshold: float = 0.5,
                                max_distance: Optional[float] = None,
                                ) -> "distributions.Descriptor":
        """
        Exact FG pair descriptor averaged over every placement of the groups
        (several groups, partial functionalisation, missing linkers), optionally
        also over ring rotations. See
        :func:`cage_isomer_builder.utils.ensemble.ensemble_descriptor`.

        Parameters
        ----------
        groups, linker_patterns
            As for :meth:`count_rgroup_isomers`.
        n_missing_linkers : int or dict, default 0
            Number of missing linkers (they carry no groups); ``{type name:
            number}`` when there are several linker types.
        mode, fractions
            ``mode="independent"`` with ``fractions={pattern: fraction}`` describes
            any composition of a large crystal exactly.
        by : str, default "orientation"
            Pair types by group and endo/exo class, or by group only (``"label"``).
        include_same_linker : bool, default False
            Also count pairs of two groups on one linker.
        rotation_angles : sequence of float, optional
            Ring rotation angles (degrees) to average over; every ring turns
            independently.
        angle_weights : sequence of float, optional
            One weight per angle (default uniform), e.g. Boltzmann weights from a
            torsion scan.
        threshold : float, default 0.5
            Endo/exo cosine limit.
        max_distance : float, optional
            Periodic structures: pairs over all images up to this distance, per
            unit cell (default 20 Angstrom). Finite structures: an optional cutoff.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> ens = cage.get_ensemble_descriptor(linker_patterns={"N": 3, "C": 3}, by="label")
        >>> {key: round(weight, 3) for key, (_, weight) in ens.summary().items()}
        {('fg:C', 'fg:C'): 3.0, ('fg:C', 'fg:N'): 9.0, ('fg:N', 'fg:N'): 3.0}
        >>> flexible = cage.get_ensemble_descriptor(groups="A", rotation_angles=range(0, 360, 90))
        >>> round(flexible.total(), 6)               # still 15 pairs of the 6 groups
        15.0
        """
        if self._fg_anchors is None:
            self.functionalise()
        F = self._layout
        slots = self.slot_features(threshold=threshold)
        if mode != "independent":
            linker_patterns = self._with_vacancies(groups, linker_patterns, n_missing_linkers)
            groups = "A" if linker_patterns else groups
        kwargs = {}
        if rotation_angles is not None:
            atoms = self.to_ase()
            adjacency = self.adjacency()
            rotors = flexibility.find_rotors(atoms, self._linker_atom_indices, adjacency)
            pos, vec, ori, rotor_of_slot = flexibility.rotation_samples(
                atoms, rotors, self._fg_anchor_indices, rotation_angles,
                centre=self.pore_centre(), threshold=threshold, adjacency=adjacency,
            )
            kwargs = dict(positions=pos, vectors=vec, orientations=ori,
                          sample_weights=angle_weights,
                          rotor_of_slot=np.where(rotor_of_slot >= 0, rotor_of_slot,
                                                 -1 - np.arange(len(rotor_of_slot))))
        if self.to_ase().pbc.any() and max_distance is None:
            max_distance = 20.0
        return ensemble.ensemble_descriptor(
            slots, F, groups=groups, linker_patterns=linker_patterns, mode=mode,
            fractions=fractions, by=by, include_same_linker=include_same_linker,
            max_distance=max_distance, **kwargs,
        )

    def rotors(self) -> list:
        """Rotatable linker rings (see :func:`cage_isomer_builder.utils.flexibility.find_rotors`).

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> len(cage.rotors())
        6
        """
        if self._fg_anchors is None:
            self.functionalise()
        return flexibility.find_rotors(self.to_ase(), self._linker_atom_indices,
                                       self.adjacency())

    def scaled(self, factor: float) -> Atoms:
        """The structure with every building block moved rigidly by an
        isotropic expansion (``factor`` > 1) or contraction (< 1) about the
        pore centre (periodic: the cell is scaled).

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> import numpy as np
        >>> bigger = cage.scaled(1.05)
        >>> _, nodes = cage.building_blocks()
        >>> r0 = np.linalg.norm(cage.to_ase().positions[nodes[0]].mean(0) - cage.pore_centre())
        >>> r1 = np.linalg.norm(bigger.positions[nodes[0]].mean(0) - cage.pore_centre())
        >>> round(float(r1 / r0), 6)
        1.05
        """
        linkers, nodes = self.building_blocks()
        return flexibility.scale_structure(self.to_ase(), linkers + nodes, factor,
                                           self.pore_centre())

    def distance_scaling(self, factors=(0.95, 1.0, 1.05)):
        """
        FG site-pair distances at each scale factor and their slopes (see
        :func:`cage_isomer_builder.utils.flexibility.distance_scaling`).

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> pairs, distances, slopes = cage.distance_scaling((0.95, 1.0, 1.05))
        >>> pairs.shape, distances.shape
        ((276, 2), (3, 276))
        """
        linkers, nodes = self.building_blocks()
        return flexibility.distance_scaling(self.to_ase(), linkers + nodes,
                                            self._fg_anchor_indices, factors,
                                            self.pore_centre())

    # ------------------------------------------------------------------
    #  Defects
    # ------------------------------------------------------------------

    def _with_vacancies(self, groups, linker_patterns, n_missing):
        """
        ``linker_patterns`` with missing linkers added as vacancy patterns,
        per linker type. ``n_missing`` is an int (one linker type) or
        ``{type name: number}``.
        """
        if not n_missing:
            return linker_patterns
        if self._fg_anchors is None:
            self.functionalise()
        layout = self._layout
        names = layout.type_names
        if isinstance(n_missing, dict):
            unknown = set(n_missing) - set(names)
            if unknown:
                raise ValueError(f"unknown linker type(s) {sorted(unknown)}; types: {list(names)}.")
            missing = [int(n_missing.get(n, 0)) for n in names]
        elif layout.n_types == 1:
            missing = [int(n_missing)]
        else:
            raise ValueError(f"with several linker types {list(names)}, give n_missing_linkers "
                             "as {type name: number}.")
        per_type = []
        if linker_patterns is not None:
            if layout.n_types > 1:
                if set(linker_patterns) != set(names):
                    raise ValueError("with several linker types, give linker_patterns as "
                                     "{type name: {pattern: number of linkers}}.")
                user = [dict(linker_patterns[n]) for n in names]
            else:
                lp = dict(linker_patterns)
                user = [dict(lp[names[0]]) if set(lp) == {names[0]} and isinstance(
                    lp[names[0]], dict) else lp]
        else:
            per_group = rgroup._per_type(groups, layout, "groups")
            user = [{rgroup._split_groups(g): None} for g in per_group]
        for t, name in enumerate(names):
            n_t = len(layout.linkers_of_type(t))
            k = missing[t]
            if not 0 <= k <= n_t:
                raise ValueError(f"cannot remove {k} of the {n_t} linkers of type {name!r}.")
            patterns = dict(user[t])
            if linker_patterns is None:
                patterns = {next(iter(patterns)): n_t - k}
            elif sum(patterns.values()) != n_t - k:
                raise ValueError(
                    f"linker_patterns cover {sum(patterns.values())} linkers of type "
                    f"{name!r}; with {k} missing they must cover {n_t - k}.")
            if k:
                patterns[ensemble.vacancy_pattern(layout.type_size(t))] = k
            per_type.append({p: n for p, n in patterns.items() if n})
        return per_type[0] if layout.n_types == 1 else dict(zip(names, per_type))

    def count_defect_isomers(self, n_missing_linkers=1, groups="A",
                             linker_patterns: Optional[dict] = None) -> int:
        """
        Number of symmetry-unique structures with ``n_missing_linkers`` linkers
        removed and the groups placed on the others (``groups=""`` for bare
        linkers), from Burnside's lemma. A missing linker is a linker pattern whose
        sites all carry a vacancy label. With several linker types,
        ``n_missing_linkers`` is ``{type name: number}`` and ``groups`` or
        ``linker_patterns`` can be given per type.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> cage.count_defect_isomers(1, groups="A")      # one linker missing, A on the other five
        256
        >>> cage.count_defect_isomers(1, groups="")       # bare: all six edges are equivalent
        1
        """
        patterns = self._with_vacancies(groups, linker_patterns, n_missing_linkers)
        return self.count_rgroup_isomers(groups=groups, linker_patterns=patterns)

    def enumerate_defect_isomers(self, n_missing_linkers=1, groups="A",
                                 linker_patterns: Optional[dict] = None,
                                 output_path: Optional[str] = None,
                                 generate_files: bool = True, limit: Optional[int] = None):
        """
        Enumerate the structures counted by :meth:`count_defect_isomers`.

        The missing linkers are deleted; the exposed node sites are left
        open (the same for cages, MOFs and COFs). Each file holds the R sites
        of the remaining linkers.

        Returns
        -------
        list of (RGroupIsomer, ase.Atoms)

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> iso, atoms = cage.enumerate_defect_isomers(1, groups="", generate_files=False)[0]
        >>> len(cage.to_ase()) - len(atoms)               # one 10-atom linker removed
        10
        """
        if self._fg_anchors is None:
            self.functionalise()
        patterns = self._with_vacancies(groups, linker_patterns, n_missing_linkers)
        isomers = self.enumerate_rgroup_isomers(groups=groups, linker_patterns=patterns,
                                                generate_files=False, limit=limit)
        offsets = self._layout.offsets
        out = []
        if generate_files and output_path:
            os.makedirs(output_path, exist_ok=True)
        for iso in isomers:
            atoms = rgroup.build_rgroup_isomer_atoms(
                self._ase_cage, self._fg_anchors, iso, self._fg_anchor_indices,
                self._fg_anchor_h_positions,
            )
            vac = (iso.labels.index(ensemble.VACANCY) + 1
                   if ensemble.VACANCY in iso.labels else None)
            remove = set()
            for L, start in enumerate(offsets):
                if vac is not None and iso.site_labels[start] == vac:
                    remove.update(self._linker_atom_indices[self._slot_linkers[L]])
            atoms = atoms[[i for i in range(len(atoms)) if i not in remove]]
            # Renumber the groups R1, R2, ... without the vacancy label.
            kept = [l for l in iso.labels if l != ensemble.VACANCY]
            renumber = np.zeros(len(iso.labels) + 1, dtype=int)
            for k, l in enumerate(iso.labels):
                if l != ensemble.VACANCY:
                    renumber[k + 1] = kept.index(l) + 1
            atoms.set_array("rgroup", renumber[atoms.arrays["rgroup"]])
            atoms.info["rgroup_labels"] = " ".join(kept)
            atoms.info["missing_linkers"] = n_missing_linkers
            name = iso.name.replace(ensemble.VACANCY, "vac")
            atoms.info["isomer"] = name
            if generate_files and output_path:
                from cage_isomer_builder.utils.sites import write_isomer
                write_isomer(os.path.join(output_path, f"{name}.xyz"), atoms)
            out.append((iso, atoms))
        return out

    def _node_group(self):
        """Validated symmetry group acting on the nodes (operations that are
        also symmetries of the FG slots, as everywhere else)."""
        from scipy.optimize import linear_sum_assignment

        _, nodes = self.building_blocks()
        atoms = self.to_ase()
        n_nodes = len(nodes)
        if atoms.pbc.any():
            from ase.geometry import find_mic

            centroids = []
            for g in nodes:
                d, _ = find_mic(atoms.positions[g] - atoms.positions[g[0]], atoms.cell, atoms.pbc)
                centroids.append(atoms.positions[g[0]] + d.mean(axis=0))
            frac = np.linalg.solve(atoms.cell[:].T, np.array(centroids).T).T
            data = symmetry.spglib_dataset(atoms, symprec=1e-2)
            if data is None:
                raise RuntimeError("spglib could not determine the space group of this "
                                   "structure at symprec=1e-2.")
            ops = zip(data.rotations, data.translations)
            perms = []
            for rot, trans in ops:
                image = frac @ rot.T + trans
                diff = image[:, None, :] - frac[None, :, :]
                diff -= np.round(diff)
                dist = np.linalg.norm(diff @ np.asarray(atoms.cell[:]), axis=2)
                rows, cols = linear_sum_assignment(dist)
                if dist[rows, cols].max() > 1.0:
                    raise RuntimeError("a space-group operation does not map the nodes "
                                       "onto themselves.")
                perms.append(cols[np.argsort(rows)])
        else:
            pg = self.point_group()
            centroids = np.array([atoms.positions[g].mean(axis=0) for g in nodes]) - pg.centre
            anchors = atoms.positions[self._fg_anchor_indices] - pg.centre
            perms = []
            for rot in pg.rotations:
                im_a = anchors @ rot
                cost = np.linalg.norm(im_a[:, None] - anchors[None], axis=2)
                r, cc = linear_sum_assignment(cost)
                if cost[r, cc].max() > self.symmetry_anchor_tol:
                    continue
                im_n = centroids @ rot
                cost = np.linalg.norm(im_n[:, None] - centroids[None], axis=2)
                r, cc = linear_sum_assignment(cost)
                if cost[r, cc].max() > self.symmetry_anchor_tol:
                    continue
                perms.append(cc[np.argsort(r)])
        return rgroup.prepare_group(perms, n_nodes, 1), nodes

    def count_node_defect_isomers(self, n_missing_nodes: int = 1) -> int:
        """Symmetry-unique ways of removing ``n_missing_nodes`` nodes (linkers
        unfunctionalised), via Burnside's lemma on the node permutations.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> cage.count_node_defect_isomers(1), cage.count_node_defect_isomers(2)   # Td: all equivalent
        (1, 1)
        """
        group, nodes = self._node_group()
        n = len(nodes)
        if not 0 < n_missing_nodes <= n:
            raise ValueError(f"n_missing_nodes must be between 1 and {n}.")
        patterns = {"D": n_missing_nodes, "": n - n_missing_nodes}
        patterns = {k: v for k, v in patterns.items() if v}
        return rgroup.count_rgroup_isomers(group, n, 1, linker_patterns=patterns)

    def enumerate_node_defect_isomers(self, n_missing_nodes: int = 1,
                                      output_path: Optional[str] = None,
                                      generate_files: bool = True) -> list:
        """
        The structures counted by :meth:`count_node_defect_isomers`: each
        missing node is deleted with all its atoms; the linker ends left
        behind stay open. FG slots are restored to H.

        Returns
        -------
        list of (tuple of removed node indices, ase.Atoms)

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
        >>> _ = cage.functionalise()
        >>> removed, atoms = cage.enumerate_node_defect_isomers(1, generate_files=False)[0]
        >>> removed, len(cage.to_ase()) - len(atoms)      # one 86-atom Zr6 node removed
        ((0,), 86)
        """
        group, nodes = self._node_group()
        n = len(nodes)
        patterns = {k: v for k, v in {"D": n_missing_nodes, "": n - n_missing_nodes}.items() if v}
        isomers = rgroup.enumerate_rgroup_isomers(group, n, 1, linker_patterns=patterns)
        base = self._ase_cage.copy()
        for slot, i_atom in enumerate(self._fg_anchor_indices):
            from cage_isomer_builder.utils.sites import clear_site
            clear_site(base, i_atom, "H", self._fg_anchor_h_positions[slot])
        if generate_files and output_path:
            os.makedirs(output_path, exist_ok=True)
        out = []
        for iso in isomers:
            removed = tuple(k for k, v in enumerate(iso.site_labels) if v)
            drop = {i for k in removed for i in nodes[k]}
            atoms = base[[i for i in range(len(base)) if i not in drop]]
            atoms.info["missing_nodes"] = " ".join(map(str, removed))
            if generate_files and output_path:
                from cage_isomer_builder.utils.sites import write_isomer
                write_isomer(os.path.join(output_path,
                                          "nodes-" + "-".join(map(str, removed)) + ".xyz"), atoms)
            out.append((removed, atoms))
        return out

    # ------------------------------------------------------------------
    #  Save
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the cage structure to file.

        If the cage was built with STK, STK's writer is used for PDB files.
        Otherwise, the ASE structure is written using ``ase.io.write``.

        Parameters
        ----------
        path : str
            Output file path, e.g. ``"base_structure/Tet2Di4.pdb"``.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> import os, tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), "cage.xyz")
        >>> cage.save(path)                               # any ASE format; PDB via STK
        >>> os.path.getsize(path) > 0
        True
        """
        path = str(path)

        if self._stk_cage is not None and path.lower().endswith(".pdb"):
            stk.PdbWriter().write(molecule=self._stk_cage, path=path)
            return

        ase_cage = self.to_ase()
        write(path, ase_cage)

    # ------------------------------------------------------------------
    #  Writers
    # ------------------------------------------------------------------

    def write_ams_run(self, path: Union[str, Path]) -> None:
        """
        Write an AMS geometry-optimisation run script for this cage, with Cartesian
        coordinates, bond orders from the STK bond matrix, lattice vectors for a
        periodic structure, and an ``Engine DFTB`` block.

        Parameters
        ----------
        path : str or Path
            Output file, e.g. ``"Tet2Di4.run"``.

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called (the bond orders come from STK).

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> import os, tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), "cage.run")
        >>> cage.write_ams_run(path)
        >>> "BondOrders" in open(path).read()
        True
        """
        if self._stk_cage is None:
            raise RuntimeError(
                "write_run() requires the STK cage. Call build() first."
            )

        atoms, bond_matrix = stk_2_ase_atoms(self._stk_cage)
        write_run(str(path), atoms, bond_matrix)

    def write_gulp_gin(
        self, path: Union[str, Path], lattice: str = "conv", mechanical: bool = False
    ) -> None:
        """
        Write a GULP input file for this cage, with UFF4MOF atom types from
        ``gulp_setup`` (as used by ``optimise(method="UFF4MOF")``), Cartesian
        coordinates, ``connect`` records from the STK bond matrix and a ``uff4mof``
        library block.

        Parameters
        ----------
        path : str or Path
            Output file, e.g. ``"Tet2Di4.gin"``.
        lattice : str, default "conv"
            ``"conv"`` (constant volume) or ``"conp"`` (constant pressure, the cell
            is optimised). Only matters for a periodic structure.
        mechanical : bool, default False
            Use ``gulp_setup``'s mechanical-properties input (looser convergence,
            different optimiser settings).

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called (the bond orders come from STK).

        Notes
        -----
        ``gulp_setup.mmanalysis.write_gin`` accepts bond orders 4, 3, 2, 1.5, 1, 0.5
        and 0.25; any other order raises ``KeyError``. STK bond matrices hold
        integer orders and 1.5 for aromatic bonds.

        Examples
        --------
        >>> from cage_isomer_builder import example_data
        >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
        >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
        ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
        >>> _ = cage.build()
        >>> import os, tempfile
        >>> path = os.path.join(tempfile.mkdtemp(), "cage.gin")
        >>> cage.write_gulp_gin(path)
        >>> open(path).read().startswith("opti")
        True
        """
        if self._stk_cage is None:
            raise RuntimeError(
                "write_gulp_run() requires the STK cage. Call build() first."
            )

        from gulp_setup.mmanalysis import analyze_mm, write_gin

        atoms, bond_matrix = stk_2_ase_atoms(self._stk_cage)
        _, mmtypes = analyze_mm(atoms, bond_order=bond_matrix)
        write_gin(str(path), atoms, bond_matrix, mmtypes, lattice=lattice, mechanical=mechanical)
