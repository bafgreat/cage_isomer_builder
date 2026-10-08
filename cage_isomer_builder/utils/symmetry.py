#!/usr/bin/env python3
from __future__ import print_function
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

import math
import numpy as np
from itertools import permutations
from cage_isomer_builder.utils import geometry


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Standard coset generators for D_nh symmetry:  E, σh, σv, σh·σv
_DIHEDRAL_COSETS = [
    (),
    ([0, 0, 1],),
    ([1, 0, 0],),
    ([0, 0, 1], [1, 0, 0]),
]

# C_nv cosets (no horizontal mirror):  E, σv
_CNV_COSETS = [
    (),
    ([1, 0, 0],),
]

_OH_AXES = ['z', '-z', 'y', '-y', 'x', '-x']


def _zrotation_library(fg_anchor, n_fold, cosets=None):
    """
    Library of an n-fold z rotation combined with a set of cosets.

    For each coset, the listed reflections are applied to a copy of
    ``fg_anchor``, then ``n_fold`` successive rotations of 360/n_fold degrees
    are recorded. The identity is removed. All prism and bipyramid topologies
    use this function.

    Parameters
    ----------
    fg_anchor : ase.Atoms
    n_fold : int
        Order of the principal rotation (3 for 120 degrees, 4 for 90, ...).
    cosets : list of tuple, optional
        Reflection normals applied before the rotations, one tuple per coset.
        Defaults to ``_DIHEDRAL_COSETS``.

    Returns
    -------
    list of tuple
        Slot-major transformation library.
    """
    if cosets is None:
        cosets = _DIHEDRAL_COSETS
    angle = 360.0 / n_fold
    transformation_lib = []
    for reflections in cosets:
        t_fg = fg_anchor.copy()
        for normal in reflections:
            t_fg = geometry.reflect_positions(t_fg, normal)
        for step in range(n_fold):
            if step != 0:
                t_fg.rotate(angle, 'z')
            transformation_lib.append(
                geometry.project_permutation(fg_anchor, t_fg)
            )
    transformation_lib.pop(0)   # remove identity
    return list(zip(*transformation_lib))


def _oh_library(fg_anchor):
    """
    Oh (octahedral) library: six axis orientations, four 90-degree z rotations
    and two cosets (plain and YZ-reflected) give 48 operations, 47 without the
    identity.

    Returns
    -------
    list of tuple
        Slot-major transformation library.
    """
    transformation_lib = []
    for axis in _OH_AXES:
        t_fg = fg_anchor.copy()
        t_fg.rotate(axis, 'z', center=[0, 0, 0])
        for deg in range(4):
            if deg != 0:
                t_fg.rotate(90, 'z')
            transformation_lib.append(
                geometry.project_permutation(fg_anchor, t_fg)
            )
    for axis in _OH_AXES:
        t_fg = geometry.reflect_positions(fg_anchor.copy(), [1, 0, 0])
        t_fg.rotate(axis, 'z', center=[0, 0, 0])
        for deg in range(4):
            if deg != 0:
                t_fg.rotate(90, 'z')
            transformation_lib.append(
                geometry.project_permutation(fg_anchor, t_fg)
            )
    transformation_lib.pop(0)
    return list(zip(*transformation_lib))


# ---------------------------------------------------------------------------
# Public API: one function per cage topology
# ---------------------------------------------------------------------------

def tet2di4_transformation_library(fg_anchor):
    """
    D4h: 4-fold z rotations and 4 cosets, 15 non-identity operations.

    Examples
    --------
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("D4h")        # a perfectly D4h-symmetric test set
    >>> library = symmetry.tet2di4_transformation_library(anchors)
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (16, 16)
    """
    return _zrotation_library(fg_anchor, n_fold=4)


def tet3_3di3_transformation_library(fg_anchor):
    """
    D3h: 3-fold z rotations and 4 cosets, 11 non-identity operations.

    Examples
    --------
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("D3h")        # a perfectly D3h-symmetric test set
    >>> library = symmetry.tet3_3di3_transformation_library(anchors)
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (12, 12)
    """
    return _zrotation_library(fg_anchor, n_fold=3)


def tet4_4di8_transformation_library(fg_anchor):
    """
    D4h: 4-fold z rotations and 4 cosets, 15 non-identity operations.

    Examples
    --------
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("D4h")        # a perfectly D4h-symmetric test set
    >>> library = symmetry.tet4_4di8_transformation_library(anchors)
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (16, 16)
    """
    return _zrotation_library(fg_anchor, n_fold=4)


def tet5di10_transformation_library(fg_anchor):
    """
    D5h: 5-fold z rotations and 4 cosets, 19 non-identity operations.

    Examples
    --------
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("D5h")        # a perfectly D5h-symmetric test set
    >>> library = symmetry.tet5di10_transformation_library(anchors)
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (20, 20)
    """
    return _zrotation_library(fg_anchor, n_fold=5)


def tet6di12_transformation_library(fg_anchor):
    """
    Oh: 47 non-identity operations.

    Examples
    --------
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("Oh")        # a perfectly Oh-symmetric test set
    >>> library = symmetry.tet6di12_transformation_library(anchors)
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (48, 48)
    """
    return _oh_library(fg_anchor)


def tet8di16_transformation_library(fg_anchor):
    """
    D4d: two sets of 4-fold z rotations, the second offset by a z flip and a
    45-degree turn, and a YZ-plane reflection; 15 non-identity operations.

    Examples
    --------
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("D4d")        # a perfectly D4d-symmetric test set
    >>> library = symmetry.tet8di16_transformation_library(anchors)
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (16, 16)
    """
    transformation_lib = []

    t_fg = fg_anchor.copy()
    for deg in range(4):
        if deg != 0:
            t_fg.rotate(90, 'z')
        transformation_lib.append(geometry.project_permutation(fg_anchor, t_fg))

    t_fg.rotate([0, 0, -1], [0, 0, 1], center=[0, 0, 0])
    t_fg.rotate(45, 'z')
    for deg in range(4):
        if deg != 0:
            t_fg.rotate(90, 'z')
        transformation_lib.append(geometry.project_permutation(fg_anchor, t_fg))

    t_fg = geometry.reflect_positions(fg_anchor.copy(), [1, 0, 0])
    for deg in range(4):
        if deg != 0:
            t_fg.rotate(90, 'z')
        transformation_lib.append(geometry.project_permutation(fg_anchor, t_fg))

    t_fg.rotate([0, 0, -1], [0, 0, 1], center=[0, 0, 0])
    t_fg.rotate(45, 'z')
    for deg in range(4):
        if deg != 0:
            t_fg.rotate(90, 'z')
        transformation_lib.append(geometry.project_permutation(fg_anchor, t_fg))

    transformation_lib.pop(0)
    return list(zip(*transformation_lib))


def tet16di32_transformation_library(fg_anchor):
    """
    D4h: 4-fold z rotations and 4 cosets, 15 non-identity operations.

    Examples
    --------
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("D4h")        # a perfectly D4h-symmetric test set
    >>> library = symmetry.tet16di32_transformation_library(anchors)
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (16, 16)
    """
    return _zrotation_library(fg_anchor, n_fold=4)


def tet24di48_transformation_library(fg_anchor):
    """
    Oh: 47 non-identity operations.

    Examples
    --------
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("Oh")        # a perfectly Oh-symmetric test set
    >>> library = symmetry.tet24di48_transformation_library(anchors)
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (48, 48)
    """
    return _oh_library(fg_anchor)


def tri2di3_transformation_library(fg_anchor):
    """
    D3h: 3-fold z rotations and 4 cosets, 11 non-identity operations.

    Examples
    --------
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("D3h")        # a perfectly D3h-symmetric test set
    >>> library = symmetry.tri2di3_transformation_library(anchors)
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (12, 12)
    """
    return _zrotation_library(fg_anchor, n_fold=3)


def tri4di6_transformation_library(fg_anchor, rotation_axes):
    """
    Td: the full tetrahedral group, from the four node-centre positions.

    All 24 permutations of the node centres are tried; SVD gives the matching
    3x3 transformation, which is applied to the FG anchors. No determinant
    correction is applied, so the improper operations (sigma_d, S4) are included
    and the full Td group is obtained.

    Parameters
    ----------
    fg_anchor : ase.Atoms
    rotation_axes : array-like, shape (4, 3)
        Positions of the four node centres.

    Returns
    -------
    list of tuple
        Slot-major transformation library, 23 non-identity operations.

    Examples
    --------
    >>> import numpy as np
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("Td")        # a perfectly Td-symmetric test set
    >>> library = symmetry.tri4di6_transformation_library(anchors, 5 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]))
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (24, 24)
    """
    node_positions = np.array(rotation_axes, dtype=float)
    node_center = node_positions.mean(axis=0)
    node_centered = node_positions - node_center

    identity = tuple(range(len(fg_anchor)))
    seen = set()
    transformation_lib = []

    for perm in permutations(range(4)):
        target = node_centered[list(perm)]
        H = node_centered.T @ target
        U, _S, Vt = np.linalg.svd(H)
        Rt = U @ Vt

        t_fg = fg_anchor.copy()
        t_fg.positions = (t_fg.positions - node_center) @ Rt + node_center

        p = tuple(geometry.project_permutation(fg_anchor, t_fg))
        if p not in seen:
            seen.add(p)
            if p != identity:
                transformation_lib.append(p)

    if not transformation_lib:
        return [() for _ in range(len(fg_anchor))]
    return list(zip(*transformation_lib))


def tri4_2di6_transformation_library(fg_anchor):
    """
    D2h: 2-fold z rotations and 4 cosets, 7 non-identity operations.

    Examples
    --------
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("D2h")        # a perfectly D2h-symmetric test set
    >>> library = symmetry.tri4_2di6_transformation_library(anchors)
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (8, 8)
    """
    return _zrotation_library(fg_anchor, n_fold=2)


def tri6di9_transformation_library(fg_anchor):
    """
    D3h: 3-fold z rotations and 4 cosets, 11 non-identity operations.

    Examples
    --------
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("D3h")        # a perfectly D3h-symmetric test set
    >>> library = symmetry.tri6di9_transformation_library(anchors)
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (12, 12)
    """
    return _zrotation_library(fg_anchor, n_fold=3)


def tri8di12_transformation_library(fg_anchor):
    """
    Oh: 47 non-identity operations.

    Examples
    --------
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("Oh")        # a perfectly Oh-symmetric test set
    >>> library = symmetry.tri8di12_transformation_library(anchors)
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (48, 48)
    """
    return _oh_library(fg_anchor)


def _axis_rotation(axis, degrees):
    """3x3 rotation (row-vector convention: x' = x @ R) about ``axis``."""
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    t = np.radians(degrees)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return (np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k).T


def _matrix_group_library(fg_anchor, generators):
    """
    Transformation library from a point group given by generator matrices
    (about the origin): the group is closed explicitly, so every operation
    is present exactly once, then each is projected onto the anchors.
    """
    group = [np.eye(3)]
    frontier = list(group)
    while frontier:
        new = []
        for g in frontier:
            for gen in generators:
                h = g @ gen
                if not any(np.allclose(h, k, atol=1e-6) for k in group):
                    group.append(h)
                    new.append(h)
        frontier = new

    transformation_lib = []
    for rot in group[1:]:
        t_fg = fg_anchor.copy()
        t_fg.positions = fg_anchor.positions @ rot
        transformation_lib.append(geometry.project_permutation(fg_anchor, t_fg))
    return list(zip(*transformation_lib))


def tri20di30_transformation_library(fg_anchor):
    """
    Ih: the icosahedral group (order 120), 119 non-identity operations.

    Generated from two 72-degree rotations about adjacent C5 axes of the
    icosahedron, (0, phi, 1) and (0, -phi, 1), and the inversion, and closed
    explicitly.

    Examples
    --------
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("Ih")        # a perfectly Ih-symmetric test set
    >>> library = symmetry.tri20di30_transformation_library(anchors)
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (120, 120)
    """
    phi = (1 + 5 ** 0.5) / 2
    return _matrix_group_library(fg_anchor, [
        _axis_rotation([0, phi, 1], 72),
        _axis_rotation([0, -phi, 1], 72),
        -np.eye(3),
    ])


# ---------------------------------------------------------------------------
# Aliases: same symmetry group, different topology name
# ---------------------------------------------------------------------------

def tet6tri8_transformation_library(fg_anchor):
    """
    Oh: same symmetry as tri8di12.

    Examples
    --------
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("Oh")        # a perfectly Oh-symmetric test set
    >>> library = symmetry.tet6tri8_transformation_library(anchors)
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (48, 48)
    """
    return _oh_library(fg_anchor)


def tri4tri4_transformation_library(fg_anchor, rotation_axes):
    """
    Td: same symmetry as tri4di6.

    Examples
    --------
    >>> import numpy as np
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("Td")        # a perfectly Td-symmetric test set
    >>> library = symmetry.tri4tri4_transformation_library(anchors, 5 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]))
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (24, 24)
    """
    return tri4di6_transformation_library(fg_anchor, rotation_axes)


def tri1tri1_transformation_library(fg_anchor):
    """
    C3v: 3-fold z rotations and 2 cosets (E and sigma_v), 5 non-identity operations.

    Examples
    --------
    >>> from cage_isomer_builder.utils import symmetry
    >>> library = symmetry.tri1tri1_transformation_library(symmetry.ideal_orbit("D3h"))
    >>> len(library[0]) + 1                    # operations incl. identity
    6
    """
    return _zrotation_library(fg_anchor, n_fold=3, cosets=_CNV_COSETS)


def tri2_2tri2_transformation_library(fg_anchor):
    """
    D2h: same symmetry as tri4_2di6.

    Examples
    --------
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("D2h")        # a perfectly D2h-symmetric test set
    >>> library = symmetry.tri2_2tri2_transformation_library(anchors)
    >>> len(library), len(library[0]) + 1              # slots, operations incl. identity
    (8, 8)
    """
    return _zrotation_library(fg_anchor, n_fold=2)


# ---------------------------------------------------------------------------
# Universal (geometry-derived) library
# ---------------------------------------------------------------------------

_PHI = (1 + 5 ** 0.5) / 2


def _axis_rotation_matrix(axis, degrees):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    t = np.radians(degrees)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def _mirror_matrix(normal):
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return np.eye(3) - 2 * np.outer(n, n)


_IDEAL_GENERATORS = {
    "D2h": lambda: [_axis_rotation_matrix([0, 0, 1], 180), _axis_rotation_matrix([1, 0, 0], 180),
                    -np.eye(3)],
    "D3h": lambda: [_axis_rotation_matrix([0, 0, 1], 120), _mirror_matrix([1, 0, 0]),
                    _mirror_matrix([0, 0, 1])],
    "D4h": lambda: [_axis_rotation_matrix([0, 0, 1], 90), _mirror_matrix([1, 0, 0]),
                    _mirror_matrix([0, 0, 1])],
    "D5h": lambda: [_axis_rotation_matrix([0, 0, 1], 72), _mirror_matrix([1, 0, 0]),
                    _mirror_matrix([0, 0, 1])],
    "D4d": lambda: [_axis_rotation_matrix([0, 0, 1], 45) @ _mirror_matrix([0, 0, 1]),
                    _mirror_matrix([1, 0, 0])],
    "Td": lambda: [_axis_rotation_matrix([1, 1, 1], 120), _axis_rotation_matrix([0, 0, 1], 180),
                   _mirror_matrix([1, -1, 0])],
    "Oh": lambda: [_axis_rotation_matrix([0, 0, 1], 90), _axis_rotation_matrix([1, 1, 1], 120),
                   -np.eye(3)],
    "Ih": lambda: [_axis_rotation_matrix([0, _PHI, 1], 72),
                   _axis_rotation_matrix([0, -_PHI, 1], 72), -np.eye(3)],
}


def ideal_orbit(group, radius=12.0, seed=0):
    """
    A perfectly symmetric test structure: the orbit of one generic point
    under an ideal point group, as ``X`` atoms, in the orientation the
    per-topology libraries of this module assume (principal axis z, a
    vertical mirror with normal x; Td/Oh/Ih in their standard cubic
    setting). Every point is at least 2.5 Angstrom from every other, so
    anchor matching is unambiguous.

    Use it to check a transformation library: on the orbit, a correct
    library returns exactly ``|G| - 1`` non-identity operations.

    Parameters
    ----------
    group : {"D2h", "D3h", "D4h", "D5h", "D4d", "Td", "Oh", "Ih"}
    radius : float, default 12.0
        Distance of the points from the origin (Angstrom).
    seed : int, default 0
        Chooses the generic point.

    Returns
    -------
    ase.Atoms
        ``|G|`` atoms, one per group element.

    Examples
    --------
    >>> from cage_isomer_builder.utils.symmetry import ideal_orbit
    >>> len(ideal_orbit("D4h")), len(ideal_orbit("Ih"))
    (16, 120)
    """
    from ase import Atoms

    if group not in _IDEAL_GENERATORS:
        raise ValueError(f"group={group!r}; choose from {sorted(_IDEAL_GENERATORS)}.")
    generators = _IDEAL_GENERATORS[group]()
    elements = [np.eye(3)]
    frontier = list(elements)
    while frontier:
        new = []
        for a in frontier:
            for g in generators:
                b = g @ a
                if not any(np.allclose(b, c, atol=1e-8) for c in elements):
                    elements.append(b)
                    new.append(b)
        frontier = new
    rng = np.random.default_rng(seed)
    while True:
        p = rng.normal(size=3)
        p = radius * p / np.linalg.norm(p)
        pts = np.array([g @ p for g in elements])
        gaps = np.linalg.norm(pts[:, None] - pts[None], axis=2) + 1e9 * np.eye(len(pts))
        if gaps.min() > 2.5:
            return Atoms(numbers=np.zeros(len(pts), dtype=int), positions=pts)


def universal_transformation_library(fg_anchor, node_centers, tol=0.5):
    """
    Transformation library from the node-centre positions.

    Works for any topology whose point group acts faithfully on the node
    centres, i.e. no non-identity operation fixes every centre:

    * 4 nodes at tetrahedral vertices (Td, 23 non-identity operations)
    * 6 nodes at octahedral vertices (Oh, 47 non-identity operations)
    * 8 nodes at cube vertices (Oh, 47 non-identity operations)

    Prism and bipyramid topologies have operations that fix all node centres
    (e.g. the C4 rotation of a D4h bipyramid), which this function cannot see;
    use their topology-specific library.

    Algorithm
    ---------
    For every permutation of the N node centres:

    1. Compute the best-fit orthogonal transformation by SVD.
    2. Keep it only if every transformed centre lands within ``tol`` of its
       target.
    3. Apply it to the FG anchors and record the permutation.
    4. Remove duplicates and the identity.

    Parameters
    ----------
    fg_anchor : ase.Atoms
    node_centers : array-like, shape (N, 3)
        Node centres, N at most 8.
    tol : float, default 0.5
        Matching tolerance (Angstrom).

    Returns
    -------
    list of tuple
        Slot-major transformation library.

    Raises
    ------
    ValueError
        If the centres lie on a line or in a plane, or there are more than 8.

    Examples
    --------
    >>> import numpy as np
    >>> from cage_isomer_builder.utils import symmetry
    >>> anchors = symmetry.ideal_orbit("Oh")
    >>> octahedron = 5 * np.vstack([np.eye(3), -np.eye(3)])     # six node centres
    >>> library = symmetry.universal_transformation_library(anchors, octahedron)
    >>> len(library[0]) + 1
    48
    """
    node_positions = np.array(node_centers, dtype=float)
    n_nodes = len(node_positions)

    spread = np.linalg.svd(node_positions - node_positions.mean(axis=0), compute_uv=False)
    if n_nodes < 4 or spread[2] < 0.1 * spread[0]:
        raise ValueError(
            "universal_transformation_library: the node centres lie on a line or "
            "in a plane, so operations that fix every node (e.g. the mirror "
            "through that plane, or rotations about the line) are invisible here "
            "and the result would be missing operations. Use point-group "
            "detection (CageBuilder.symmetry_method = 'detect') instead."
        )
    if n_nodes > 8:
        raise ValueError(
            f"universal_transformation_library: {n_nodes} nodes give "
            f"{math.factorial(n_nodes):,} permutations (the limit is 8 nodes, "
            "40,320 permutations). Use point-group detection "
            "(symmetry_method='detect') for larger cages."
        )

    node_center = node_positions.mean(axis=0)
    node_centered = node_positions - node_center
    identity = tuple(range(len(fg_anchor)))
    seen = set()
    transformation_lib = []

    for perm in permutations(range(n_nodes)):
        target = node_centered[list(perm)]
        H = node_centered.T @ target
        U, _S, Vt = np.linalg.svd(H)
        Rt = U @ Vt  # no determinant correction, so improper operations are kept

        if not np.allclose(node_centered @ Rt, target, atol=tol):
            continue  # not a valid symmetry operation

        t_fg = fg_anchor.copy()
        t_fg.positions = (t_fg.positions - node_center) @ Rt + node_center

        p = tuple(geometry.project_permutation(fg_anchor, t_fg))
        if p not in seen:
            seen.add(p)
            if p != identity:
                transformation_lib.append(p)

    if not transformation_lib:
        return [() for _ in range(len(fg_anchor))]
    return list(zip(*transformation_lib))


def _spglib_call(function, *args, **kwargs):
    """
    Call a spglib function, returning None on failure under both of
    spglib's error conventions (old: returns None and warns; new: raises
    SpglibError), without its deprecation notice about the old convention.
    """
    import warnings

    import spglib

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Set OLD_ERROR_HANDLING",
                                category=DeprecationWarning)
        try:
            return getattr(spglib, function)(*args, **kwargs)
        except Exception as exc:          # spglib.error.SpglibError
            if type(exc).__name__ == "SpglibError":
                return None
            raise


def spglib_dataset(atoms, symprec=1e-2):
    """
    spglib symmetry dataset of a periodic structure, or None if spglib
    finds none at this tolerance.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.utils.symmetry import spglib_dataset
    >>> spglib_dataset(example_data.read("RUBTAK01")).international
    'Fm-3m'
    """
    cell = (np.asarray(atoms.cell[:], dtype=float), atoms.get_scaled_positions(), atoms.numbers)
    return _spglib_call("get_symmetry_dataset", cell, symprec=symprec)


def primitive_cell(atoms, symprec=1e-2):
    """
    The primitive cell of a periodic structure: the smallest cell that
    repeats to give the whole crystal (via spglib).

    The atom positions are kept as they are (not moved to ideal symmetric
    positions), so the geometry is unchanged. Only the cell, positions and
    elements are kept; other per-atom arrays and ``info`` are dropped.

    Parameters
    ----------
    atoms : ase.Atoms
        Periodic structure.
    symprec : float, default 1e-2
        spglib symmetry tolerance, the same default as
        :func:`spacegroup_transformation_library`.

    Returns
    -------
    ase.Atoms
        The primitive cell. If ``atoms`` is already primitive, a copy with
        the same number of atoms.

    Raises
    ------
    ValueError
        If ``atoms`` is not periodic, or spglib finds no symmetry at this
        tolerance.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.utils.symmetry import primitive_cell
    >>> uio66 = example_data.read("RUBTAK01")            # conventional fcu cell
    >>> len(uio66), len(primitive_cell(uio66))           # four lattice points per cell
    (440, 110)
    """
    from ase import Atoms

    if not atoms.pbc.all():
        raise ValueError("primitive_cell needs a periodic structure (pbc in all directions).")
    cell = (np.asarray(atoms.cell[:], dtype=float), atoms.get_scaled_positions(), atoms.numbers)
    reduced = _spglib_call("standardize_cell", cell, to_primitive=True, no_idealize=True,
                           symprec=symprec)
    if reduced is None:
        raise ValueError("spglib could not find the symmetry of this structure; try a looser symprec.")
    lattice, scaled, numbers = reduced
    return Atoms(numbers=numbers, scaled_positions=scaled, cell=lattice, pbc=True)


def spacegroup_transformation_library(cage_atoms, fg_anchor_indices, symprec=1e-2, atom_tol=0.5):
    """
    Transformation library of a periodic structure from its space group
    (spglib).

    A unit cell's FG anchors are related by the full space group, including
    translations, screw axes and glide planes. A point group about one centre
    misses most of these operations, so a periodic structure must use this
    function.

    Algorithm
    ---------
    1. Get every (rotation, translation) operation of the space group from
       spglib, in fractional coordinates.
    2. Apply each operation to the anchors' fractional positions and match every
       image to the anchor it coincides with (minimum image).
    3. Remove the identity and duplicate permutations.

    Parameters
    ----------
    cage_atoms : ase.Atoms
        The full periodic structure (with ``pbc`` and a cell); the space group
        is computed from all its atoms.
    fg_anchor_indices : sequence of int
        Indices of the FG anchors in ``cage_atoms``, in slot order.
    symprec : float, default 1e-2
        spglib tolerance. Real, slightly noisy structures usually need a looser
        value than spglib's default of 1e-5.
    atom_tol : float, default 0.5
        Distance (Angstrom) within which an image matches an anchor. Anchors are
        always much further apart than this.

    Returns
    -------
    list of tuple
        Slot-major transformation library, in the same format as
        universal_transformation_library.

    Raises
    ------
    RuntimeError
        If spglib finds no space group, or an operation does not map the
        anchors onto themselves within ``atom_tol`` (usually ``symprec`` is too
        loose for the structure).

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.utils.symmetry import primitive_cell, spacegroup_transformation_library
    >>> cell = primitive_cell(example_data.read("RUBTAK01"))
    >>> zr = [a.index for a in cell if a.symbol == "Zr"]
    >>> library = spacegroup_transformation_library(cell, zr)    # how the 6 Zr permute
    >>> len(library), len(library[0]) + 1                         # Fm-3m: 48 point operations
    (6, 48)
    """
    lattice = np.asarray(cage_atoms.cell[:], dtype=float)
    dataset = spglib_dataset(cage_atoms, symprec=symprec)
    if dataset is None:
        raise RuntimeError(
            "spglib could not determine a space group for this structure - "
            "try a looser symprec."
        )

    anchor_frac = cage_atoms.get_scaled_positions()[list(fg_anchor_indices)]
    n_anchors = len(anchor_frac)
    identity = tuple(range(n_anchors))
    seen = set()
    transformation_lib = []

    for rotation, translation in zip(dataset.rotations, dataset.translations):
        image_frac = anchor_frac @ rotation.T + translation

        perm = []
        for i in range(n_anchors):
            diff_frac = anchor_frac - image_frac[i]
            diff_frac -= np.round(diff_frac)   # minimum image, fractional
            diff_cart = diff_frac @ lattice
            dists = np.linalg.norm(diff_cart, axis=1)
            j = int(np.argmin(dists))
            if dists[j] > atom_tol:
                raise RuntimeError(
                    "A space-group operation did not map the FG anchor set "
                    "onto itself (closest match "
                    f"{dists[j]:.3f} Å > atom_tol={atom_tol}). The structure "
                    "doesn't satisfy the symmetry spglib detected at this "
                    "symprec; try tightening symprec or loosening atom_tol."
                )
            perm.append(j)

        p = tuple(perm)
        if p not in seen:
            seen.add(p)
            if p != identity:
                transformation_lib.append(p)

    if not transformation_lib:
        return [() for _ in range(n_anchors)]
    return list(zip(*transformation_lib))
