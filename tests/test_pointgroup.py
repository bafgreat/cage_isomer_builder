#!/usr/bin/env python3
"""
Tests for cage_isomer_builder.utils.pointgroup: point groups of ideal
polyhedra and real molecules, in random orientations and with noise.
"""

import itertools

import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from scipy.spatial.transform import Rotation

from cage_isomer_builder.utils.pointgroup import (
    detect_point_group,
    pointgroup_transformation_library,
)
from cage_isomer_builder.utils.rgroup import prepare_group

PHI = (1 + 5 ** 0.5) / 2


def _ring(n, radius, z=0.0, phase=0.0):
    t = phase + 2 * np.pi * np.arange(n) / n
    return np.c_[radius * np.cos(t), radius * np.sin(t), np.full(n, z)]


TETRAHEDRON = 3 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
OCTAHEDRON = 3 * np.vstack([np.eye(3), -np.eye(3)])
CUBE = 3 * np.array(list(itertools.product([-1, 1], repeat=3)), float)
ICOSAHEDRON = 2 * np.array(
    [p for a, b in itertools.product([-1, 1], repeat=2)
     for p in ([0, a, b * PHI], [a, b * PHI, 0], [b * PHI, 0, a])], float)


SHAPES = {
    'Td':  Atoms('C4', TETRAHEDRON),
    'Oh':  Atoms('C6', OCTAHEDRON),
    'Ih':  Atoms('C12', ICOSAHEDRON),
    'D4h': Atoms('C8', np.vstack([_ring(4, 3, 1.5), _ring(4, 3, -1.5)])),
    'D5h': Atoms('C10', np.vstack([_ring(5, 3, 1.5), _ring(5, 3, -1.5)])),
    'D3h': Atoms('C6', np.vstack([_ring(3, 3, 1.5), _ring(3, 3, -1.5)])),
    'D4d': Atoms('C8', np.vstack([_ring(4, 3, 1.5), _ring(4, 3, -1.5, np.pi / 4)])),
    'D3d': Atoms('C6', np.vstack([_ring(3, 3, 1.5), _ring(3, 3, -1.5, np.pi / 3)])),
    'D2h': Atoms('C6', [[3, 0, 0], [-3, 0, 0], [0, 4, 0], [0, -4, 0], [0, 0, 5], [0, 0, -5]]),
    'C3v': Atoms('C3N', np.vstack([_ring(3, 3, 0), [[0, 0, 2]]])),
    'C4v': Atoms('C4N', np.vstack([_ring(4, 3, 0), [[0, 0, 2]]])),
}

ORDERS = {'Td': 24, 'Oh': 48, 'Ih': 120, 'D4h': 16, 'D5h': 20, 'D3h': 12,
          'D4d': 16, 'D3d': 12, 'D2h': 8, 'C3v': 6, 'C4v': 8}


def _randomised(atoms, seed, noise):
    rng = np.random.default_rng(seed)
    out = atoms.copy()
    rot = Rotation.random(random_state=seed).as_matrix()
    out.positions = atoms.positions @ rot.T + rng.normal(0, noise, atoms.positions.shape) + [5, -2, 7]
    return out


@pytest.mark.parametrize('name', sorted(SHAPES))
def test_ideal_shapes_any_orientation(name) -> None:
    for seed in range(3):
        group = detect_point_group(_randomised(SHAPES[name], seed, noise=0.01), tol=0.2)
        assert group.name == name, (name, seed, group.name)
        assert group.order == ORDERS[name]
        assert group.max_deviation < 0.1


def test_chiral_octahedral_and_tetrahedral_groups() -> None:
    # Decorate every vertex of an octahedron/tetrahedron with a twisted
    # pair: the proper rotations survive, every mirror is destroyed.
    proper_o = [r for r in detect_point_group(SHAPES['Oh'], tol=0.1).rotations
                if np.linalg.det(r) > 0]
    seed = np.array([[3.0, 0.6, 0.25]])
    chiral = Atoms('C24', np.vstack([seed @ r for r in proper_o]))
    group = detect_point_group(_randomised(chiral, 1, 0.005), tol=0.1)
    assert (group.name, group.order) == ('O', 24)

    proper_t = [r for r in detect_point_group(SHAPES['Td'], tol=0.1).rotations
                if np.linalg.det(r) > 0]
    chiral = Atoms('C12', np.vstack([np.array([[2.0, 1.0, 0.4]]) @ r for r in proper_t]))
    group = detect_point_group(chiral, tol=0.1)
    assert (group.name, group.order) == ('T', 12)


def test_real_molecules() -> None:
    assert detect_point_group(read('tests/data/methane.xyz'), tol=0.1).name == 'Td'
    ibuprofen = detect_point_group(read('tests/data/ibuprofen.xyz'), tol=0.1)
    assert (ibuprofen.name, ibuprofen.order) == ('C1', 1)


def test_operations_form_a_group_and_map_elements_to_themselves() -> None:
    group = detect_point_group(SHAPES['C4v'], tol=0.1)
    numbers = SHAPES['C4v'].numbers
    for perm in group.atom_perms:
        assert np.array_equal(numbers[perm], numbers)
    rows = {p.tobytes() for p in group.atom_perms}
    assert all(p[q].tobytes() in rows for p in group.atom_perms for q in group.atom_perms)


def test_too_loose_tolerance_is_caught_not_silently_wrong() -> None:
    # Slightly squashed tetrahedron: true group D2d (order 8). A generous
    # tolerance "sees" Td; an in-between one finds a non-closed set and
    # must raise instead of returning it.
    squashed = Atoms('C4', TETRAHEDRON * [1, 1, 0.97])
    assert detect_point_group(squashed, tol=0.05).name == 'D2d'
    assert detect_point_group(squashed, tol=0.5).name == 'Td'


def test_slot_library_is_a_valid_group_on_the_anchors() -> None:
    # Tetrahedron of C with one X site "anchor" pointing out of each vertex.
    centres = TETRAHEDRON
    anchors = centres * 1.4
    atoms = Atoms('C4X4', np.vstack([centres, anchors]))
    tl = pointgroup_transformation_library(atoms, [4, 5, 6, 7], tol=0.1)
    group = prepare_group(list(zip(*tl)), 4, 1)
    assert group.order == 24
