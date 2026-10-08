"""
Tests for reducing a periodic structure to its primitive cell
(utils.symmetry.primitive_cell and CageBuilder.load_from_file(primitive=True)).

The main check does not trust spglib: it tiles the primitive cell back up to
the original cell and requires every original atom to be reproduced.
"""

from pathlib import Path

import numpy as np
import pytest
from ase.build import make_supercell, molecule
from ase.geometry import find_mic
from ase.io import read

from cage_isomer_builder.cage import CageBuilder
from cage_isomer_builder.utils.symmetry import primitive_cell

CIF = Path(__file__).parent / "data" / "RUBTAK01.cif"


def test_primitive_cell_tiles_back_to_the_original():
    original = read(str(CIF))
    prim = primitive_cell(original)

    # RUBTAK01 is face-centred cubic (Fm-3m): four primitive cells per CIF cell.
    assert len(original) == 4 * len(prim)
    assert original.cell.volume == pytest.approx(4 * prim.cell.volume, rel=1e-6)

    # The original lattice must be a whole-number combination of the primitive one.
    transform = original.cell[:] @ np.linalg.inv(prim.cell[:])
    assert np.allclose(transform, np.round(transform), atol=1e-6)
    tiled = make_supercell(prim, np.round(transform))

    # Every original atom has a same-element atom of the tiled cell on top of it.
    assert sorted(tiled.numbers) == sorted(original.numbers)
    for position, number in zip(original.positions, original.numbers):
        same = tiled.positions[tiled.numbers == number]
        _, distances = find_mic(same - position, original.cell, pbc=True)
        assert distances.min() < 0.05


def test_already_primitive_is_unchanged_in_size():
    prim = primitive_cell(read(str(CIF)))
    assert len(primitive_cell(prim)) == len(prim)


def test_non_periodic_structure_raises():
    with pytest.raises(ValueError, match="periodic"):
        primitive_cell(molecule("C6H6"))


def test_load_from_file_primitive_makes_enumeration_possible():
    full = CageBuilder.load_from_file(str(CIF))
    full.functionalise()
    prim = CageBuilder.load_from_file(str(CIF), primitive=True)
    _, anchors = prim.functionalise()

    assert len(prim.to_ase()) == len(full.to_ase()) // 4
    assert len(anchors) == len(full._fg_anchor_indices) // 4
    assert prim.count_rgroup_isomers("A") < full.count_rgroup_isomers("A")

    n_ab = prim.count_rgroup_isomers("AB")
    isomers = prim.enumerate_rgroup_isomers("AB", generate_files=False)
    assert len(isomers) == n_ab
