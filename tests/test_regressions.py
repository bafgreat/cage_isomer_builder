#!/usr/bin/env python3
"""
Regression tests for problems found while documenting the package.
"""

import contextlib
import inspect
import io
import warnings

import numpy as np
import pytest
from ase.build import molecule

import cage_isomer_builder.cage.library as library
from cage_isomer_builder.cage.base import CageBuilder, autodetect_building_block
from cage_isomer_builder.utils.pointgroup import detect_point_group
from cage_isomer_builder.utils.read_write import stk_2_ase_atoms


def test_point_group_of_small_molecules() -> None:
    """Every atom of water/NH3 is within 1 A of the centre; they used to
    come out as C1."""
    names = {"H2O": "C2v", "NH3": "C3v", "CH4": "Td", "C6H6": "D6h", "CO2": "Dinfh"}
    for name, group in names.items():
        if group == "Dinfh":
            with pytest.raises(ValueError, match="linear"):
                detect_point_group(molecule(name))
        else:
            assert detect_point_group(molecule(name)).name == group, name


def test_smiles_dummy_atoms_mark_connection_points() -> None:
    """SMILES ``[*]`` dummies used to fail inside STK (atomic number 0)."""
    for smiles in ("[*]c1ccc([*])cc1", "*c1ccc(*)cc1", "[1*]c1ccc([2*])cc1"):
        assert autodetect_building_block(smiles).get_num_functional_groups() == 2
    cage = library.Tet2Di4CageBuilder(linker="[*]c1ccc([*])cc1")
    cage.build()
    assert cage.to_ase().get_chemical_formula() == "C24H16Zr2"


ALL_BUILDERS = [cls for _, cls in inspect.getmembers(library, inspect.isclass)
                if issubclass(cls, CageBuilder) and cls is not CageBuilder]


@pytest.mark.parametrize("builder", ALL_BUILDERS, ids=lambda c: c.__name__)
def test_every_topology_builds_with_its_defaults(builder) -> None:
    cage = builder()
    cage.build()
    assert len(cage.to_ase()) > 0


@pytest.mark.parametrize("name", ["Tri4Tri4CageBuilder", "Tri1Tri1CageBuilder"])
def test_equal_topicity_blocks_only_bond_to_each_other(name) -> None:
    """Two tritopic building blocks: every bond between building blocks must
    join a node to a linker (the vertex graph is bipartite)."""
    cage = getattr(library, name)()
    cage.build()
    atoms, bonds = stk_2_ase_atoms(cage._stk_cage)
    blocks, is_node = cage._classify_atoms_by_bb(atoms)
    owner = {i: k for k, members in blocks.items() for i in members}
    for i, j in zip(*np.nonzero(np.triu(bonds))):
        if owner.get(i) is not None and owner.get(j) is not None and owner[i] != owner[j]:
            assert is_node[owner[i]] != is_node[owner[j]]


def test_tri4tri4_isomers() -> None:
    cage = library.Tri4Tri4CageBuilder()
    cage.build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cage.functionalise()
    assert cage.point_group().name == "Td"
    assert cage.count_unique_isomers_burnside() == 6


def test_optimise_can_be_silent_and_aliases_do_not_warn() -> None:
    from cage_isomer_builder.utils.optimise import run_uff4mof_optimisation

    methanol = molecule("CH3OH")              # gulp_setup types its O as O_HH (= O_3)
    bonds = np.zeros((6, 6))
    for i, j in ((0, 1), (0, 3), (0, 4), (0, 5), (1, 2)):
        bonds[i, j] = bonds[j, i] = 1.0
    out = io.StringIO()
    with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(out):
        warnings.simplefilter("always")
        run_uff4mof_optimisation(methanol, bonds, logfile=None)
    assert out.getvalue() == ""
    assert not [w for w in caught if "published" in str(w.message)]


def test_spglib_calls_raise_no_deprecation_warnings() -> None:
    from cage_isomer_builder import example_data
    from cage_isomer_builder.utils.symmetry import primitive_cell, spglib_dataset

    uio66 = example_data.read("RUBTAK01")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert spglib_dataset(uio66).international == "Fm-3m"
        assert len(primitive_cell(uio66)) == 110


def test_example_data_ships_every_documented_structure() -> None:
    from cage_isomer_builder import example_data

    for name in ("uio66_tri_node", "uio66_tetra_node", "uio66_sbu", "bdc", "biphenyl", "NH2",
                 "methane", "ibuprofen", "RUBTAK01", "ABIXEQ", "ABINOQ"):
        assert example_data.path(name).exists()
    with pytest.raises(KeyError):
        example_data.path("not-a-structure")


def test_writers_use_the_optimised_geometry(tmp_path) -> None:
    """save(), write_gulp_gin() and write_ams_run() used to write the STK
    construction geometry even after optimise()."""
    from ase.io import read

    from cage_isomer_builder import example_data

    cage = library.Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
                                      linker=example_data.path("bdc"), scale_multiplier=0.225)
    cage.build()
    optimised = cage.optimise(include_charges=False, logfile=None).positions.copy()
    cage.functionalise()                       # R-site markers are written as their H
    expected = cage.to_ase(hydrogens=True)
    assert np.allclose(expected.positions, optimised)

    cage.save(tmp_path / "cage.pdb")
    pdb = read(tmp_path / "cage.pdb")
    assert pdb.get_chemical_symbols() == expected.get_chemical_symbols()
    assert np.abs(pdb.positions - expected.positions).max() < 1e-3   # PDB keeps 3 decimals

    cage.write_ams_run(tmp_path / "cage.run")
    text = (tmp_path / "cage.run").read_text()
    assert "OptimizeLattice" not in text       # a finite cage has no lattice
    rows = text.split("Atoms\n")[1].split("\nEnd")[0].split("\n")
    coords = np.array([[float(v) for v in row.split()[1:]] for row in rows if row.strip()])
    assert np.abs(coords - expected.positions).max() < 1e-6
