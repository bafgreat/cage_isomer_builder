#!/usr/bin/env python3
"""
Every symmetry library in cage_isomer_builder.utils.symmetry, and the
point-group detector, checked against the exact group it should produce.

The test structure is a perfectly symmetric point set: the orbit of one
generic point under the ideal group, in the orientation each library
assumes (principal axis z, sigma_v normal x; see _DIHEDRAL_COSETS). The
expected permutations are computed directly from the group matrices, so a
library that is missing operations, has foreign ones, or isn't closed fails.
"""

import numpy as np
import pytest
from ase import Atoms

from cage_isomer_builder.utils import symmetry as S
from cage_isomer_builder.utils.pointgroup import detect_point_group
from cage_isomer_builder.utils.rgroup import prepare_group

PHI = (1 + 5 ** 0.5) / 2
_AXES = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}


def _rot(axis, degrees):
    a = np.asarray(_AXES[axis] if isinstance(axis, str) else axis, dtype=float)
    a /= np.linalg.norm(a)
    t = np.radians(degrees)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def _mirror(normal):
    n = np.asarray(_AXES[normal] if isinstance(normal, str) else normal, dtype=float)
    n /= np.linalg.norm(n)
    return np.eye(3) - 2 * np.outer(n, n)


GENERATORS = {
    "D2h": [_rot("z", 180), _rot("x", 180), -np.eye(3)],
    "D3h": [_rot("z", 120), _mirror("x"), _mirror("z")],
    "D4h": [_rot("z", 90), _mirror("x"), _mirror("z")],
    "D5h": [_rot("z", 72), _mirror("x"), _mirror("z")],
    "D4d": [_rot("z", 45) @ _mirror("z"), _mirror("x")],
    "Td":  [_rot([1, 1, 1], 120), _rot("z", 180), _mirror([1, -1, 0])],
    "Oh":  [_rot("z", 90), _rot([1, 1, 1], 120), -np.eye(3)],
    "Ih":  [_rot([0, PHI, 1], 72), _rot([0, -PHI, 1], 72), -np.eye(3)],
}
ORDERS = {"D2h": 8, "D3h": 12, "D4h": 16, "D5h": 20, "D4d": 16, "Td": 24,
          "Oh": 48, "Ih": 120}


def _close(generators):
    group = [np.eye(3)]
    frontier = list(group)
    while frontier:
        new = []
        for a in frontier:
            for g in generators:
                b = g @ a
                if not any(np.allclose(b, c, atol=1e-8) for c in group):
                    group.append(b)
                    new.append(b)
        frontier = new
    return group


def _ideal_orbit(group_name, seed=0):
    """Generic orbit (anchors as At atoms) and its exact permutation group."""
    group = _close(GENERATORS[group_name])
    assert len(group) == ORDERS[group_name]
    rng = np.random.default_rng(seed)
    while True:
        p = rng.normal(size=3)
        p = 12 * p / np.linalg.norm(p)
        pts = np.array([g @ p for g in group])
        gaps = np.linalg.norm(pts[:, None] - pts[None], axis=2) + 99 * np.eye(len(pts))
        if gaps.min() > 2.5:
            break
    perms = {
        tuple(int(np.argmin(np.linalg.norm(pts - x, axis=1))) for x in pts @ g.T)
        for g in group
    }
    return Atoms(f"At{len(pts)}", pts), perms


def _library_perms(library, n):
    ops = list(zip(*library)) if library and len(library[0]) else []
    return {tuple(row) for row in prepare_group(ops, n, 1).perms}


TOPOLOGY_LIBRARIES = [
    ("D4h", S.tet2di4_transformation_library),
    ("D3h", S.tet3_3di3_transformation_library),
    ("D4h", S.tet4_4di8_transformation_library),
    ("D5h", S.tet5di10_transformation_library),
    ("Oh", S.tet6di12_transformation_library),
    ("D4d", S.tet8di16_transformation_library),
    ("D4h", S.tet16di32_transformation_library),
    ("Oh", S.tet24di48_transformation_library),
    ("D3h", S.tri2di3_transformation_library),
    ("D2h", S.tri4_2di6_transformation_library),
    ("D3h", S.tri6di9_transformation_library),
    ("Oh", S.tri8di12_transformation_library),
    ("Ih", S.tri20di30_transformation_library),
    ("Oh", S.tet6tri8_transformation_library),
    ("D2h", S.tri2_2tri2_transformation_library),
]


@pytest.mark.parametrize("group_name,library", TOPOLOGY_LIBRARIES,
                         ids=[f.__name__ for _, f in TOPOLOGY_LIBRARIES])
def test_topology_library_is_exactly_its_group(group_name, library) -> None:
    for seed in (0, 1):
        anchors, expected = _ideal_orbit(group_name, seed)
        assert _library_perms(library(anchors), len(anchors)) == expected


NODE_LAYOUTS = {
    "Td": 5 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float),
    "Oh": 5 * np.vstack([np.eye(3), -np.eye(3)]),
}


@pytest.mark.parametrize("group_name", sorted(NODE_LAYOUTS))
def test_node_centre_library_on_3d_layouts(group_name) -> None:
    anchors, expected = _ideal_orbit(group_name)
    library = S.universal_transformation_library(anchors, NODE_LAYOUTS[group_name])
    assert _library_perms(library, len(anchors)) == expected


def test_node_centre_library_on_cube_nodes() -> None:
    anchors, expected = _ideal_orbit("Oh")
    cube = 5 * np.array([[x, y, z] for x in (1, -1) for y in (1, -1) for z in (1, -1)], float)
    library = S.universal_transformation_library(anchors, cube)
    assert _library_perms(library, len(anchors)) == expected


@pytest.mark.parametrize("nodes", [
    np.array([[0, 0, 5], [0, 0, -5]], float),                          # Pd2L4 lantern
    5 * np.array([[1, 0, 0], [-0.5, 0.866, 0], [-0.5, -0.866, 0]]),    # triangle
    5 * np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], float),  # square
], ids=["line", "triangle", "square"])
def test_node_centre_library_refuses_flat_layouts(nodes) -> None:
    # On a flat layout, operations fixing every node are invisible and the
    # library would miss them (e.g. 2 of 16 D4h operations).
    anchors, _ = _ideal_orbit("D4h")
    with pytest.raises(ValueError, match="line or in a plane"):
        S.universal_transformation_library(anchors, nodes)


@pytest.mark.parametrize("group_name", sorted(GENERATORS))
def test_detector_finds_exact_group_in_any_orientation(group_name) -> None:
    anchors, expected = _ideal_orbit(group_name)
    q = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))[0]
    anchors.positions = anchors.positions @ q + [3.0, -1.0, 2.0]
    group = detect_point_group(anchors, tol=0.1)
    assert group.name == group_name
    assert {tuple(p) for p in group.atom_perms} == expected
