#!/usr/bin/env python3
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

import os
import re
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

from cage_isomer_builder.utils import symmetry
from cage_isomer_builder.utils.functionalise import full_functionalisation
from cage_isomer_builder.utils.geometry import bonded_pairs
from cage_isomer_builder.utils.read_write import (
    generate_isomer_structure_file,
    stk_2_ase_atoms,
    write_run,
)
from cage_isomer_builder.utils.util import (
    count_unique_isomers,
    fg2fg_distance_count,
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

DUMMY_SMARTS = "[*][C]"

# Noble-gas symbols used as geometry/orientation markers in MOF node files.
# They are stripped from building blocks before STK construction so they
# don't appear in cage output or bias the centroid calculation.
_NOBLE_GAS_SYMBOLS = frozenset({'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'})

_SINGLE_ATOM_RE = re.compile(r"^\[([A-Z][a-z]?)\]$")

_STRUCTURE_EXTENSIONS = {
    ".xyz", ".pdb", ".cif", ".mol", ".sdf", ".mol2", ".gjf", ".com"
}


def make_dummy_factory() -> stk.SmartsFunctionalGroupFactory:
    """
    Factory for dummy atoms attached to any atom.

    Atom 0: dummy atom, deleted.
    Atom 1: real attachment atom, used as bonder.
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

        metal = [i.index for i in molecule_to_write if i.symbol in transition_metals(
        ) and i.index not in porphyrin_checker]
        if len(metal) > 0:
            mof_metal.append(molecule_to_write)
        else:
            mof_linker.append(molecule_to_write)
        building_unit_regions[idx] = molecule_to_write.info['atom_indices_mapping']
    return mof_metal, mof_linker, building_unit_regions

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
    """

    c_components, breaking_point, porpyrin, regions, _ =\
        secondary_building_units(ase_atom)
    _, organic_sbu, _ =\
        find_unique_building_units(c_components, breaking_point, ase_atom, porpyrin, regions)
    select_sbu = [linker for linker in organic_sbu if len(linker) >= 6]
    linker = select_sbu[0]
    linker_atom_indices = linker.info['atom_indices_mapping']
    metal_atom_indicces = [i for i in range(len(ase_atom))
                           if i not in sum(linker_atom_indices, [])
                           ]
    return linker_atom_indices, metal_atom_indicces


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
    """Return True if value looks like a molecular structure file path."""
    if not isinstance(value, (str, Path)):
        return False

    path = Path(value)
    return path.suffix.lower() in _STRUCTURE_EXTENSIONS


def is_ase_atoms(value: Any) -> bool:
    """Return True if value is an ASE Atoms object."""
    return isinstance(value, Atoms)


def is_smiles_like(value: Any) -> bool:
    """Return True if value should be interpreted as a SMILES string."""
    return isinstance(value, str) and not is_file_like(value)


def atoms_to_smiles(atoms: Atoms) -> str:
    """
    Convert ASE Atoms to SMILES using mofstructure/OpenBabel.
    """
    smiles = ase_2_pybel(atoms).write("smi")
    if not smiles:
        raise ValueError("Could not convert ASE Atoms object to SMILES.")
    return smiles


def file_to_atoms(path: Union[str, Path]) -> Atoms:
    """Read a molecular structure file with ASE."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file not found: {path}")
    return read(str(path))


def file_to_smiles(path: Union[str, Path]) -> str:
    """
    Convert a molecular structure file to SMILES using ASE + mofstructure.
    """
    atoms = file_to_atoms(path)
    return atoms_to_smiles(atoms)


def mol_input_to_smiles(mol: MoleculeInput) -> str:
    """
    Convert SMILES, file path, or ASE Atoms into a SMILES string.
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
    """
    atoms = mol_input_to_atoms_or_none(mol)
    if atoms is None:
        return mol_input_to_smiles(mol), None

    return atoms_to_smiles(atoms), atoms.get_positions()


def has_dummy_marker_atoms(atoms: Atoms) -> bool:
    """Return True if ASE Atoms contains dummy attachment markers."""
    return any(atom.symbol == "X" for atom in atoms)


def remove_dummy_marker_atoms(atoms: Atoms) -> tuple[Atoms, list[int]]:
    """
    Remove dummy marker and noble-gas atoms; return bonder indices.

    ``X`` atoms mark attachment directions. Noble-gas atoms (He, Ne, …) are
    orientation/geometry markers added by MOF deconstruction tools.  Both are
    removed before STK construction. Each ``X`` marker is paired with the
    nearest non-hydrogen atom in the stripped molecule, which becomes the
    bonder for the corresponding functional group.
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

    Uses ``BuildingBlock.init()`` directly (no SMILES round-trip) so that
    the original atom ordering and exact 3D coordinates are preserved.
    This is essential for large inorganic clusters (e.g. UiO-66 Zr nodes)
    where OpenBabel → RDKit canonicalisation would reorder atoms and
    silently assign bonder functional groups to the wrong positions.
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
    """
    return "[*]" in smiles or "[#0]" in smiles or "*" in smiles


def is_single_atom_smiles(smiles: str) -> bool:
    """Return True for bracketed single-atom SMILES such as [Zn], [Pd], [Zr]."""
    return bool(_SINGLE_ATOM_RE.match(smiles.strip()))


def is_single_atom_input(mol: MoleculeInput) -> bool:
    """
    Return True if input represents exactly one atom.
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
    4. If dummy atoms are present:
       use dummy SMARTS factory [#0][*].
    5. If user provided fg:
       use that factory.
    6. Otherwise search FG_FACTORY_MAP.
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
            smiles=smiles,
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
    Statistics for a cage topology computed after functionalisation.

    Attributes
    ----------
    n_linkers : int
        Number of ditopic linkers in the cage.
    n_fg_slots : int
        Total number of functional group anchor slots (n_linkers × 4).
    n_unique_isomers : int
        Number of symmetry-unique isomers (Burnside's lemma result).
    distance_keys : list of float
        Sorted unique inter-linker FG-to-FG distances in Angstroms.
    total_fg_pairs : list of int
        Total active FG pair count at each distance.
    endo_endo_pairs : list of int
        Pair counts where both anchors point inward (endo-endo).
    endo_exo_pairs : list of int
        Pair counts with one inward and one outward anchor (endo-exo).
    exo_exo_pairs : list of int
        Pair counts where both anchors point outward (exo-exo).
    kde_x : numpy.ndarray
        Distance grid at which the KDE was evaluated.
    kde_y : numpy.ndarray
        KDE density values rescaled to total pair count.
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
    Abstract base class for building cage structures with isomer enumeration.

    Each subclass hard-codes one STK topology graph and its symmetry
    transformation library. The shared workflow (build → functionalise →
    enumerate → statistics → SMILES) lives here.

    Typical usage
    -------------
    >>> cage = Tet2Di4CageBuilder(linker_smiles='Nc1ccc(N)cc1')
    >>> cage.build()
    >>> cage.functionalise()
    >>> isomers = cage.enumerate_isomers(output_path='Tet2Di4_isomers')
    >>> stats   = cage.get_statistics()
    >>> smi     = cage.to_smiles()

    For cages without an STK builder (e.g. Tet6Di12):
    >>> cage = Tet6Di12CageBuilder.load_from_file('base_structure/Tet6Di12.pdb')
    >>> cage.functionalise()
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
        Initialize a cage builder.

        The linker and node may be provided as a SMILES string, an ASE
        ``Atoms`` object, or a molecular structure file readable by ASE
        (e.g. ``.xyz``, ``.mol``, ``.mol2``, ``.sdf``, ``.pdb`` or ``.cif``).

        Functional groups are assigned automatically using the following
        priority:

            1. **Single atom** (nodes only)
            If the node represents a single atom (e.g. ``"[Zn]"`` or an
            ASE object containing one atom), an ``stk.SingleAtom``
            building block is created automatically.

            2. **Dummy atoms**
            If dummy atoms (``[*]`` or ``[#0]``) are present, the SMARTS
            factory

                ``[#0][*]``

            is used automatically.

            3. **User-specified factory**
            If ``linker_fg`` or ``node_fg`` is supplied, that factory is
            used.

            4. **Automatic factory detection**
            If no factory is supplied, all factories in
            ``FG_FACTORY_MAP`` are searched until a compatible functional
            group is found.

        Parameters
        ----------
        linker : MoleculeInput, optional
            Linker building block.

            May be one of

            * a SMILES string,
            * an ASE ``Atoms`` object, or
            * a molecular structure file readable by ASE.

            Defaults to ``default_linker``.

        node : MoleculeInput, optional
            Cage node.

            May be one of

            * a SMILES string,
            * an ASE ``Atoms`` object, or
            * a molecular structure file readable by ASE.

            Defaults to ``default_node``.

        linker_fg : str or stk.FunctionalGroupFactory, optional
            Functional group factory assigned to the linker.

            May be

            * a key in ``FG_FACTORY_MAP``,
            * an ``stk.FunctionalGroupFactory`` instance, or
            * ``None`` to enable automatic detection.

            Defaults to ``default_linker_fg``.

        node_fg : str or stk.FunctionalGroupFactory, optional
            Functional group factory assigned to the node.

            Use ``"single_atom"`` for isolated metal nodes or ``None`` for
            automatic detection.

            Defaults to ``default_node_fg`` only when the default node is
            used. User-supplied nodes default to automatic detection.

        node_fg_count : int, optional
            Number of functional groups assigned to a single-atom node.

            Ignored for multi-atom nodes.

            Defaults to ``default_node_fg_count``.

        smarts_pattern : str, optional
            SMARTS pattern used when ``linker_fg`` or ``node_fg`` is
            ``"smarts"``.

        optimizer_step : float, default=0.01
            Step size used by the STK ``Collapser`` optimizer.

        distance_threshold : float, default=5.0
            Distance threshold (Å) supplied to the STK ``Collapser``
            optimizer.

        scale_multiplier : float, default=1.0
            Forwarded to the STK topology graph's own ``scale_multiplier``.
            STK sizes the *entire* topology graph - node-to-node distances
            included - by multiplying its unit vertex positions by
            ``scale_multiplier * max(bb.get_maximum_diameter() for bb in
            (node_bb, linker_bb))``. For a bulky multi-atom node next to a
            much shorter linker (e.g. a UiO-66 Zr6 SBU with a short
            dicarboxylate), the *node's* diameter alone sets the scale for
            every edge, including ones spanned by the short linker - so the
            initial placement comes out stretched to several times the
            linker's natural length, forcing :meth:`optimise` into a drastic
            global contraction instead of a local bond-length correction.
            Lowering ``scale_multiplier`` shrinks the initial placement
            before construction, at the source, rather than compensating for
            it after the fact via ``expansion_factor``.

            Rule of thumb: build once at the default 1.0, note the resulting
            node-to-node distance ``d0`` (e.g. via each node subunit's
            centroid), and the distance ``d_target`` you actually want (a
            reasonable estimate is the linker's own end-to-end length plus
            twice a typical M-L bond length); then set
            ``scale_multiplier = d_target / d0``.

        expansion_factor : float, default=1.0
            After STK assembly, each building-block subunit is translated
            rigidly so that its distance from the cage centroid is multiplied
            by this factor. Internal geometries are preserved exactly.

            Use values slightly above 1.0 when bulky multi-atom nodes (e.g.
            Zr6 clusters with CH3-capped peripheral connections) physically
            overlap their neighbours in the initial STK placement.  A value
            too large will leave a visible gap between the node and linker
            bond sites; use the smallest value that eliminates the clash.

            Rule of thumb: if the node's outer radius is ``r_max`` and the
            initial vertex-to-vertex distance is ``d``, the minimum factor is
            ``2*r_max / d``.  For ``uio66_tri_node.xyz`` with CH3 caps at
            5.97 Å and d ≈ 11.66 Å, a factor of **1.15** works well.

        linker_smiles : str, optional
            Deprecated alias for ``linker``.

            Included for backwards compatibility with earlier releases.

        node_smiles : str, optional
            Deprecated alias for ``node``.

            Included for backwards compatibility with earlier releases.
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
        Return the STK topology graph for this cage topology.

        Default implementation instantiates ``self._stk_topology_cls`` with
        the supplied building blocks, the instance's ``scale_multiplier``
        (STK's own sizing of the topology graph - see the constructor
        docstring), and, when ``self._use_collapser`` is True, a
        ``stk.Collapser`` optimizer using the instance's ``optimizer_step``
        and ``distance_threshold`` parameters.

        Subclasses that need non-standard construction (e.g. custom topology
        graphs not in ``stk.cage``) should override this method directly.
        All others only need to set ``_stk_topology_cls`` as a class attribute.
        """
        if self._stk_topology_cls is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set _stk_topology_cls "
                "or override _build_stk_topology."
            )
        kwargs: dict = {
            'building_blocks': (node_bb, linker_bb),
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
        Return the symmetry transformation library for this cage.

        Dispatches on periodicity before anything else: a periodic
        structure (``pbc`` set, e.g. a real MOF loaded via
        :meth:`load_from_file`) is related to itself by its
        crystallographic *space* group — translations and non-symmorphic
        operations included — not the finite molecular *point* group every
        other code path here assumes. Using the point-group model on a
        periodic input doesn't just miss a few equivalences: on a real
        MOF unit cell it left the isomer count too large by roughly nine
        orders of magnitude (192 true space-group operations vs. the ~24
        a point-group fit found in 4 node positions alone). So periodic
        structures always go through
        :func:`~cage_isomer_builder.utils.symmetry.spacegroup_transformation_library`
        (spglib-based) regardless of topology, and only finite structures
        reach :meth:`_get_point_group_transformation_library`.

        Parameters
        ----------
        fg_anchors : ase.Atoms
            FG anchor atoms ("At" atoms) as returned by ``full_functionalisation``.

        Returns
        -------
        list[tuple[int]]
            Transformation library in zipped format — one entry per FG slot,
            each a tuple of images under the non-identity symmetry operations.
        """
        ase_cage = self.to_ase()
        if ase_cage.pbc.any():
            return symmetry.spacegroup_transformation_library(
                ase_cage, self._fg_anchor_indices
            )
        return self._get_point_group_transformation_library(fg_anchors)

    def _get_point_group_transformation_library(self, fg_anchors: Atoms):
        """
        Return the point-group transformation library for this topology
        (finite structures only — see :meth:`_get_transformation_library`).

        Default implementation: uses the node centre-of-mass positions stored
        in ``self._node_centers`` (set by :meth:`functionalise`) together with
        :func:`~cage_isomer_builder.utils.symmetry.universal_transformation_library`
        to derive all symmetry operations via SVD over node permutations.

        This works correctly for any topology where the point group acts
        faithfully on the node positions — i.e. no non-identity symmetry
        operation leaves every node centre fixed.  Concretely:

        * 4 nodes at tetrahedral vertices  (Td)
        * 6 nodes at octahedral vertices   (Oh)
        * 8 nodes at cubic vertices        (Oh)

        For prism / bipyramid topologies (Tet2Di4, Tri2Di3, …) and for large
        cages with > 8 nodes, subclasses must override this method with a
        topology-specific hardcoded function.

        Parameters
        ----------
        fg_anchors : ase.Atoms
            FG anchor atoms ("At" atoms) as returned by ``full_functionalisation``.

        Returns
        -------
        list[tuple[int]]
            Transformation library in zipped format — one entry per FG slot,
            each a tuple of images under the non-identity symmetry operations.

        Raises
        ------
        NotImplementedError
            If ``_node_centers`` is not set or contains more than 8 centres.
        """
        if self._node_centers is not None and 2 <= len(self._node_centers) <= 8:
            return symmetry.universal_transformation_library(
                fg_anchors, self._node_centers
            )
        raise NotImplementedError(
            f"{type(self).__name__} must override _get_point_group_transformation_library "
            "or ensure _node_centers is populated with 2–8 node CoMs before "
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
        Translate each STK building-block subunit rigidly outward.

        Each subunit's centroid is moved to
        ``cage_centroid + expansion_factor * (subunit_centroid - cage_centroid)``.
        Atom positions within a subunit are shifted by the same vector, so
        internal bond lengths and angles are unchanged.

        Subunits are identified via STK ``atom_infos`` — not by distance-based
        BFS — so this works even when large nodes overlap in the initial
        STK placement (which would break NeighborList-based cluster detection).
        """
        subunits: dict[int, list[int]] = {}
        for atom_info in self._stk_cage.get_atom_infos():
            if atom_info.get_building_block() is None:
                continue
            # get_building_block_id() is unique per placement instance, so the
            # 4 Zr6 node placements get 4 distinct keys even though they all
            # reference the same BuildingBlock Python object.
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
        """
        node_bb = self._make_node_bb()
        linker_bb = self._make_linker_bb()

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
    ) -> Atoms:
        """
        Run ASE geometry optimisation on the cage.

        Two backends are available, selected via ``method``:

        * ``"UFF4MOF"`` (default) - a classical force field evaluated from
          the STK bond matrix and explicit UFF atom types (see
          :mod:`cage_isomer_builder.utils.optimise`). Bond connectivity is
          taken from the STK-constructed cage and respected exactly, which
          matters when inter-atomic distances in the initial geometry are
          atypical (bulky multi-atom nodes, an expanded cage, ...) and would
          confuse a distance-only bond perception.
        * ``"GFN1-xTB"`` / ``"GFN2-xTB"`` - a tblite GFN-xTB electronic
          structure calculation. Unlike UFF4MOF, xTB needs no explicit
          bond-order matrix or atom typing - it derives its own electronic
          structure directly from atomic numbers and positions - but for the
          same reason it cannot be told to honour a specific bond matrix.

        Call this after :meth:`build` and before :meth:`functionalise`.
        If called after :meth:`functionalise`, the At marker atoms are
        stripped before optimisation (they are not real chemistry) and
        :meth:`functionalise` should be re-run afterwards.

        Parameters
        ----------
        fmax : float, default 0.05
            Force convergence threshold in eV/Å.
        steps : int, default 500
            Maximum number of optimisation steps.
        optimizer_cls : ASE optimizer class, optional
            Any ``ase.optimize`` class (``LBFGS``, ``BFGS``, ``FIRE``, …).
            Defaults to ``LBFGS``.
        trajectory : str, optional
            Path for an ASE ``.traj`` trajectory file.
        rigid_sbus : bool, default False
            Treat every node/linker SBU as a rigid body (see
            :class:`~cage_isomer_builder.utils.rigid_body.FixRigidBodies`),
            relaxing only inter-SBU placement. Metal nodes are kept fully
            rigid (their own cluster bonding is what full relaxation tends
            to get wrong), but each linker's donor atom(s) - whichever atom
            carries the bond crossing into a node - are excluded from the
            linker's rigid group. A node's binding vectors are fixed by its
            rigid shape, and a rigid linker's 6 whole-body DOF generally
            can't satisfy every M-L bond/angle around a node at once, which
            is what bends the M-L bond; freeing just the donor atom gives
            that bond the local freedom it needs without touching the rest
            of the linker (ring shape, aromaticity, ...). Set to False to
            relax every atom independently instead.
        include_charges : bool, default True
            Store per-atom partial charges on the returned ``Atoms`` (via
            ``set_initial_charges``). For ``method="GFN1-xTB"``/``"GFN2-xTB"``
            these are a byproduct of the optimisation's own electronic
            structure. UFF4MOF itself has no notion of partial charges, so
            for ``method="UFF4MOF"`` a tblite GFN2-xTB single-point
            calculation is run on the *pre-optimisation* geometry to obtain
            them; those same fixed charges are then also fed into
            :class:`~cage_isomer_builder.utils.optimise.UFF4MOFCalculator`
            to add its Coulomb term (``ke*qi*qj/r`` between the same
            non-bonded pairs the van der Waals term already covers), rather
            than optimising with electrostatics entirely absent. Setting
            this to ``False`` skips both - no charges are computed or
            reported, and ``method="UFF4MOF"`` optimises without a Coulomb
            term at all.
        charge : float, default 0.0
            Total charge of the structure, passed to tblite - either as the
            xTB optimisation's own charge, or (for ``method="UFF4MOF"``) as
            the charge of the pre-optimisation single-point calculation used
            to seed the Coulomb term.
        method : str, default "UFF4MOF"
            ``"UFF4MOF"``, or a tblite xTB Hamiltonian name such as
            ``"GFN1-xTB"`` / ``"GFN2-xTB"``.

        Returns
        -------
        ase.Atoms
            Optimised cage structure. ``self._ase_cage`` is updated in-place.
        """
        from cage_isomer_builder.utils.rigid_body import FixRigidBodies

        if self._stk_cage is not None:
            # Always derive atoms from the original STK cage so that At
            # marker atoms from functionalise() are never included.
            ase_atoms, bond_matrix = stk_2_ase_atoms(self._stk_cage)
        else:
            ase_atoms = self.to_ase().copy()
            # Strip At atoms (FG markers) — not parameterised by any FF.
            non_at = [i for i, a in enumerate(ase_atoms) if a.symbol != 'At']
            ase_atoms = ase_atoms[non_at]
            bond_matrix = None

        if rigid_sbus:
            if self._stk_cage is None:
                raise RuntimeError(
                    "rigid_sbus=True requires the STK cage to identify SBU "
                    "groupings (unavailable after load_from_file()). Pass "
                    "rigid_sbus=False to relax every atom independently instead."
                )
            bb_atoms, bb_is_node = self._classify_atoms_by_bb(ase_atoms)

            # Nodes stay fully rigid. Linkers drop their donor atom(s) - any
            # atom with a bond crossing into a node - out of the rigid group,
            # so each M-L bond keeps the local freedom a whole-rigid-body fit
            # can't give it (see the rigid_sbus docstring above).
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
                from cage_isomer_builder.utils.tblite_charges import predict_tblite_charges

                # Computed once, up front, from the pre-optimisation geometry,
                # and held fixed for the whole run (standard UFF/QEq
                # practice) so they can also drive UFF4MOF's Coulomb term -
                # which is otherwise entirely absent from the force field.
                charges = predict_tblite_charges(ase_atoms, charge=charge)

            optimised = run_uff4mof_optimisation(
                atoms=ase_atoms,
                bond_matrix=bond_matrix,
                fmax=fmax,
                steps=steps,
                optimizer_cls=optimizer_cls,
                trajectory=trajectory,
                charges=charges,
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
            )

        if include_charges and charges is not None:
            optimised.set_initial_charges(charges)

        self._ase_cage = optimised
        return optimised


    # ------------------------------------------------------------------
    #  Load existing structure
    # ------------------------------------------------------------------

    @classmethod
    def load_from_file(cls, path: str, **init_kwargs):
        """
        Create a builder from an existing molecular structure.

        This method bypasses STK construction and instead loads an existing
        cage structure directly into the builder. The loaded structure may
        subsequently be functionalised, analysed, or enumerated.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to a molecular structure readable by ASE (e.g. PDB, CIF,
            XYZ, MOL or SDF).

        **init_kwargs
            Additional keyword arguments forwarded to the constructor.

        Returns
        -------
        CageBuilder
            Builder initialized from the supplied structure.
        """
        instance = cls.__new__(cls)
        CageBuilder.__init__(instance, **init_kwargs)
        instance._ase_cage = read(path)
        return instance

    # ------------------------------------------------------------------
    #  Conversion
    # ------------------------------------------------------------------

    def to_ase(self) -> Atoms:
        """
        Return the cage as an ASE Atoms object.
        Converts from the STK cage if not already done.
        """
        if self._ase_cage is not None:
            return self._ase_cage
        if self._stk_cage is None:
            raise RuntimeError(
                "No structure available. Call build() or load_from_file() first."
            )
        self._ase_cage, _ = stk_2_ase_atoms(self._stk_cage)
        return self._ase_cage

    def get_bond_matrix(self) -> np.ndarray:
        """
        This cage's bond-order matrix, taken directly from the STK
        construction rather than re-perceived from geometry - the same
        matrix write_run/write_gulp_gin use, for the same reason: a real
        bond order survives even where two bonded atoms sit unusually far
        apart, which a distance-based guess on the whole structure can
        miss (or, on a larger assembly like a host-guest complex, wrongly
        add between atoms that are merely close together but not bonded).

        Needed by anything building on top of this cage's own bonding
        without re-perceiving it - e.g.
        utils.functionalise.place_guest_in_host's host_bond_matrix
        parameter, which keeps the host's real bonds and the union of
        each guest's own (separately, safely perceived) internal bonds,
        rather than running one neighbour-list pass over the merged
        host+guest structure.

        Returns
        -------
        np.ndarray, shape (n_atoms, n_atoms)

        Raises
        ------
        RuntimeError
            If build() hasn't been called - a structure loaded via
            load_from_file() has no STK-derived bond order information.
        """
        if self._stk_cage is None:
            raise RuntimeError(
                "get_bond_matrix() requires the STK cage. Call build() "
                "first - a structure loaded via load_from_file() has no "
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
        """
        return ase_2_pybel(self.to_ase()).write('smi').strip()

    # ------------------------------------------------------------------
    #  Functionalisation
    # ------------------------------------------------------------------

    def _classify_atoms_by_bb(self, ase_cage):
        """
        Group atom indices by building-block placement instance via STK tracking.

        Uses ``get_building_block_id()`` — unique per placement, so four Zr6
        node placements get four distinct keys even though they reference the
        same Python ``BuildingBlock`` object.  Each instance is classified as
        a node (contains a transition metal) or a linker (purely organic).

        Parameters
        ----------
        ase_cage : ase.Atoms

        Returns
        -------
        bb_atoms : dict[int, list[int]]
            Map from placement id → list of atom indices.
        bb_is_node : dict[int, bool]
            Map from placement id → True if the instance contains any metal.
        """
        tm_set = set(transition_metals())
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
        Return linker and metal atom index lists via STK atom-info tracking.

        Robust even when building blocks physically overlap, because it uses
        ``get_building_block_id()`` rather than distance-based connectivity.

        Returns
        -------
        linker_atom_indices : list[list[int]]
            One sub-list per linker instance.
        metal_atom_indices : list[int]
            Flat list of atom indices in node (metal-containing) instances.
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
        Return one centre per node instance, using the node's points of
        extension (atoms that bond to linker building blocks) as the
        representative position.

        For a single-atom node the bonder atom IS the metal atom, so this
        is identical to using the metal position.  For a multi-atom node
        (e.g. a Zr6 cluster with CH3 acetate caps) it takes the mean of
        the 3 carboxylate-C atoms that bond to the BDC linkers, which gives
        the functional centroid of the SBU connection sites and is
        unaffected by the size or orientation of the capping groups.

        Falls back to the transition-metal centroid if no inter-BB bonds are
        found for a given node placement.

        Returns
        -------
        node_centers : np.ndarray of shape (n_nodes, 3), or None
        """
        bb_atoms, bb_is_node = self._classify_atoms_by_bb(ase_cage)

        # Reverse map: cage atom_id → bb_id
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

        tm_set = set(transition_metals())
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

    def functionalise(self):
        """
        Functionalise the cage and identify functional-group anchor sites.

        Returns
        -------
        fg_anchors : ase.Atoms
            Functionalised ASE structure containing the generated anchor atoms.

        fg_anchor_indices : list[int]
            Indices of the anchor atoms used for isomer enumeration.
        """
        # Work on a copy so the raw (pre-At) cage is preserved if needed,
        # then store the At-decorated version as the canonical _ase_cage so
        # enumerate_isomers and generate_isomer_structure_file see the markers.
        functionalised = self.to_ase().copy()

        # When the STK cage is available, use its atom-info tracking to
        # identify subunits.  This is robust even when large building blocks
        # physically overlap (which breaks mofstructure's connectivity analysis
        # and the BFS node-center finder).
        if self._stk_cage is not None:
            linker_atom_indices, metal_atom_indices = self._get_sbus_from_stk(functionalised)
            self._node_centers = self._get_node_centers_from_stk(functionalised)
        else:
            linker_atom_indices, metal_atom_indices = get_sbus_indices(functionalised)
            self._node_centers = get_node_centers(functionalised, metal_atom_indices)

        self._linker_atom_indices = linker_atom_indices
        self._fg_anchors, self._fg_anchor_indices, self._fg_anchor_h_positions = full_functionalisation(
            functionalised, sum(linker_atom_indices, [])
        )
        self._fg_per_linker = len(self._fg_anchor_indices) // len(linker_atom_indices)
        self._ase_cage = functionalised

        return self._fg_anchors, self._fg_anchor_indices

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
        Enumerate symmetry-unique isomers using the transformation library.

        Walks candidate isomers with iter_unique_isomers, which checks each
        candidate directly against its own symmetry orbit instead of
        allocating a fg_per_linker**n_linkers flag array - the array is
        what used to raise MemoryError on a periodic structure with more
        than a couple dozen linkers (see docs/isomers.md). Memory use here
        is O(1) in n_linkers regardless of limit; runtime is unchanged
        (still one orbit check per raw combination visited) unless you
        pass limit, which stops early instead of walking the full space.

        Parameters
        ----------
        output_path : str, optional
            Directory to write XYZ structure files. No files are written
            if None or if generate_files is False.
        generate_files : bool
            Write one XYZ file per unique isomer (default True).
        limit : int, optional
            Stop after this many unique isomers instead of walking the
            entire fg_per_linker**n_linkers space. Use this for cages/MOFs
            where the full count (from count_unique_isomers_burnside) is
            too large to fully enumerate - canonical representatives occur
            roughly 1-in-|symmetry group order| along the walk, not
            1-in-fg_per_linker**n_linkers, so a modest limit is normally
            fast even when the full space isn't.

        Returns
        -------
        unique_isomers : list of list of int
            Each element is a base-4 isomer descriptor.
        """
        if self._fg_anchors is None:
            self.functionalise()

        n_fg_slots = len(self._fg_anchors)
        fg_per_linker = self._fg_per_linker if self._fg_per_linker is not None else 4
        n_linkers = n_fg_slots // fg_per_linker
        tl = self._get_transformation_library(self._fg_anchors.copy())

        if generate_files and output_path:
            os.makedirs(output_path, exist_ok=True)

        unique_isomers = []
        for _, descriptor in iter_unique_isomers(tl, n_linkers, fg_per_linker, limit=limit):
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

    def count_unique_isomers_burnside(self) -> int:
        """
        Count unique isomers via Burnside's lemma without full enumeration.
        Useful for large cages where iterating over 4^n_linkers is impractical.

        Returns
        -------
        n_unique : int
        """
        if self._fg_anchors is None:
            self.functionalise()
        n_fg_slots = len(self._fg_anchors)
        fg_per_linker = self._fg_per_linker if self._fg_per_linker is not None else 4
        n_linkers = n_fg_slots // fg_per_linker
        tl = self._get_transformation_library(self._fg_anchors.copy())
        raw_tl = list(zip(*tl))
        return count_unique_isomers(raw_tl, n_linkers, n_fg_slots, fg_per_linker)

    # ------------------------------------------------------------------
    #  Statistics
    # ------------------------------------------------------------------

    def get_statistics(self, n_kde_bins: int = 200) -> CageStatistics:
        """
        Compute FG-to-FG distance statistics and a Gaussian KDE for this cage.

        Parameters
        ----------
        n_kde_bins : int
            Number of evaluation points for the KDE curve (default 200).

        Returns
        -------
        stats : CageStatistics
        """
        if self._fg_anchors is None:
            self.functionalise()

        n_fg_slots = len(self._fg_anchors)
        fg_per_linker = self._fg_per_linker if self._fg_per_linker is not None else 4
        n_linkers = n_fg_slots // fg_per_linker
        n_unique = self.count_unique_isomers_burnside()

        distance_keys, total, endo_endo, endo_exo, exo_exo = fg2fg_distance_count(
            self._fg_anchors, self._fg_anchor_indices, fg_per_linker
        )
        kde_x, kde_y = kernel_density_estimation(distance_keys, total, n_kde_bins)

        return CageStatistics(
            n_linkers=n_linkers,
            n_fg_slots=n_fg_slots,
            n_unique_isomers=n_unique,
            distance_keys=distance_keys,
            total_fg_pairs=total,
            endo_endo_pairs=endo_endo,
            endo_exo_pairs=endo_exo,
            exo_exo_pairs=exo_exo,
            kde_x=kde_x,
            kde_y=kde_y,
        )

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
        """
        path = str(path)

        if self._stk_cage is not None and path.lower().endswith(".pdb"):
            stk.PdbWriter().write(molecule=self._stk_cage, path=path)
            return

        ase_cage = self.to_ase()
        write(path, ase_cage)

    # ------------------------------------------------------------------
    #  write_run
    # ------------------------------------------------------------------

    def write_ams_run(self, path: Union[str, Path]) -> None:
        """
        Write an AMS geometry-optimisation run script for this cage.

        The generated ``.run`` file contains:

        * ``Task GeometryOptimization``
        * Cartesian coordinates under ``System / Atoms``
        * Bond orders under ``System / BondOrders`` (from the STK bond matrix)
        * Lattice vectors if the structure is periodic
        * ``Engine DFTB`` block

        Parameters
        ----------
        path : str or Path
            Output file path, e.g. ``"Tet2Di4.run"``.

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called — the bond matrix is
            only available from the STK constructed molecule.
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
        Write a GULP molecular-mechanics input file for this cage.

        Uses ``gulp_setup`` - the same UFF4MOF atom-typing backend behind
        ``optimise(method="UFF4MOF")`` - to assign UFF4MOF atom types and
        write a ready-to-run ``.gin`` file: cartesian coordinates, explicit
        ``connect`` bond records taken from the STK bond matrix (so bond
        order is respected exactly rather than re-perceived from geometry),
        and a ``uff4mof`` species/library block.

        Parameters
        ----------
        path : str or Path
            Output file path, e.g. ``"Tet2Di4.gin"``.
        lattice : str, default "conv"
            GULP lattice-optimisation keyword: ``"conv"`` for constant
            volume (no cell optimisation) or ``"conp"`` for constant
            pressure (cell optimisation). Only relevant if the structure
            is periodic - a freshly built cage isn't, so this has no
            effect unless you've set a cell on it yourself.
        mechanical : bool, default False
            Use ``gulp_setup``'s "mechanical properties" input variant
            (looser convergence, different optimiser settings) instead of
            the standard geometry-optimisation one.

        Raises
        ------
        RuntimeError
            If :meth:`build` has not been called — the bond matrix is
            only available from the STK constructed molecule.

        Notes
        -----
        ``gulp_setup.mmanalysis.write_gin`` maps each bond order to a GULP
        bond-type keyword via a fixed lookup ({4, 3, 2, 1.5, 1, 0.5, 0.25});
        a bond order outside that set raises ``KeyError``. The STK bond
        matrix normally only contains integer orders (plus 1.5 for
        perceived aromaticity), so this doesn't come up in practice.
        """
        if self._stk_cage is None:
            raise RuntimeError(
                "write_gulp_run() requires the STK cage. Call build() first."
            )

        from gulp_setup.mmanalysis import analyze_mm, write_gin

        atoms, bond_matrix = stk_2_ase_atoms(self._stk_cage)
        _, mmtypes = analyze_mm(atoms, bond_order=bond_matrix)
        write_gin(str(path), atoms, bond_matrix, mmtypes, lattice=lattice, mechanical=mechanical)
