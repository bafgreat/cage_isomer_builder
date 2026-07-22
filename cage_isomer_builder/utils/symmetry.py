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
    n-fold z-rotation × cosets transformation library.

    For each coset, applies the listed reflections to a fresh copy of
    fg_anchor, then records ``n_fold`` successive rotations of 360/n_fold
    degrees.  The first entry (identity) is removed.

    This single function replaces every prism/bipyramid topology-specific
    function that previously contained the same loop with only the angle
    and step-count varying.

    Parameters
    ----------
    fg_anchor : ase.Atoms
    n_fold : int
        Order of the principal rotation (3 → 120°, 4 → 90°, …).
    cosets : list of tuple, optional
        Each tuple is a sequence of reflection normals applied before the
        rotations.  Defaults to _DIHEDRAL_COSETS.

    Returns
    -------
    list of tuple  — slot-major transformation library.
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
    Oh (octahedral) transformation library.

    Six canonical axis orientations × four 90° z-rotations × two cosets
    (plain and YZ-reflected): 6 × 4 × 2 = 48 ops; identity removed → 47.

    Returns
    -------
    list of tuple  — slot-major.
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
# Public API — one function per cage topology
# ---------------------------------------------------------------------------

def tet2di4_transformation_library(fg_anchor):
    """D4h — 4-fold z × 4 cosets → 15 non-identity ops."""
    return _zrotation_library(fg_anchor, n_fold=4)


def tet3_3di3_transformation_library(fg_anchor):
    """D3h — 3-fold z × 4 cosets → 11 non-identity ops."""
    return _zrotation_library(fg_anchor, n_fold=3)


def tet4_4di8_transformation_library(fg_anchor):
    """D4h — 4-fold z × 4 cosets → 15 non-identity ops."""
    return _zrotation_library(fg_anchor, n_fold=4)


def tet5di10_transformation_library(fg_anchor):
    """D5h — 5-fold z × 4 cosets → 19 non-identity ops."""
    return _zrotation_library(fg_anchor, n_fold=5)


def tet6di12_transformation_library(fg_anchor):
    """Oh — 47 non-identity ops."""
    return _oh_library(fg_anchor)


def tet8di16_transformation_library(fg_anchor):
    """
    Staggered D4d — two groups of 4-fold z-rotations, each offset by a
    z-flip + 45° tilt, mirrored by a YZ-plane reflection → 15 non-identity ops.
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
    """D4h — 4-fold z × 4 cosets → 15 non-identity ops."""
    return _zrotation_library(fg_anchor, n_fold=4)


def tet24di48_transformation_library(fg_anchor):
    """Oh — 47 non-identity ops."""
    return _oh_library(fg_anchor)


def tri2di3_transformation_library(fg_anchor):
    """D3h — 3-fold z × 4 cosets → 11 non-identity ops."""
    return _zrotation_library(fg_anchor, n_fold=3)


def tri4di6_transformation_library(fg_anchor, rotation_axes):
    """
    Td — full tetrahedral group derived from the 4 node-centre positions.

    All 24 permutations of the node centres are enumerated; SVD gives the
    corresponding 3×3 transformation which is applied to the FG anchors.
    No determinant correction is used, so improper operations (σd, S4) are
    included alongside proper rotations — recovering the full Td group.

    Parameters
    ----------
    fg_anchor : ase.Atoms
    rotation_axes : array-like, shape (4, 3)
        Positions of the four tetrahedral node centres.

    Returns
    -------
    list of tuple  — slot-major, 23 non-identity ops.
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
    """D2h — 2-fold z × 4 cosets → 7 non-identity ops."""
    return _zrotation_library(fg_anchor, n_fold=2)


def tri6di9_transformation_library(fg_anchor):
    """D3h — 3-fold z × 4 cosets → 11 non-identity ops."""
    return _zrotation_library(fg_anchor, n_fold=3)


def tri8di12_transformation_library(fg_anchor):
    """Oh — 47 non-identity ops."""
    return _oh_library(fg_anchor)


def tri20di30_transformation_library(fg_anchor):
    """
    Ih — icosahedral group (order 120), 119 non-identity ops.

    Optional YZ-reflection × optional x-rotation × five 72° rotations about
    each of the six C5 axes of the icosahedron.  One duplicate at index 29
    is discarded.
    """
    c1, c2 = 8.988, 14.543
    c5_axes = [
        [[ 0, -c2,  c1], [ 0,  c2, -c1]],
        [[ 0,  c2,  c1], [ 0, -c2, -c1]],
        [[ c1,  0,  c2], [-c1,  0, -c2]],
        [[-c1,  0,  c2], [ c1,  0, -c2]],
        [[ c2, -c1,  0], [-c2,  c1,  0]],
        [[ c2,  c1,  0], [-c2, -c1,  0]],
    ]
    transformation_lib = []
    t_fg = fg_anchor.copy()
    for r in range(2):
        if r == 1:
            t_fg = geometry.reflect_positions(t_fg, [1, 0, 0])
        for p in range(2):
            if p == 1:
                t_fg.rotate('-x', 'x', center=[0, 0, 0])
            for _ in range(5):
                for v in c5_axes:
                    t_fg.rotate(72, v=v[1], center=[0, 0, 0])
                    transformation_lib.append(
                        geometry.project_permutation(fg_anchor, t_fg)
                    )
    transformation_lib.pop(29)
    return list(zip(*transformation_lib))


# ---------------------------------------------------------------------------
# Aliases — same symmetry group, different topology name
# ---------------------------------------------------------------------------

def tet6tri8_transformation_library(fg_anchor):
    """Oh — same symmetry as tri8di12."""
    return _oh_library(fg_anchor)


def tri4tri4_transformation_library(fg_anchor, rotation_axes):
    """Td — same symmetry as tri4di6."""
    return tri4di6_transformation_library(fg_anchor, rotation_axes)


def tri1tri1_transformation_library(fg_anchor):
    """C3v — 3-fold z × 2 cosets (E + σv) → 5 non-identity ops."""
    return _zrotation_library(fg_anchor, n_fold=3, cosets=_CNV_COSETS)


def tri2_2tri2_transformation_library(fg_anchor):
    """D2h — same as tri4_2di6."""
    return _zrotation_library(fg_anchor, n_fold=2)


# ---------------------------------------------------------------------------
# Universal (geometry-derived) library
# ---------------------------------------------------------------------------

def universal_transformation_library(fg_anchor, node_centers, tol=0.5):
    """
    Derive the symmetry transformation library from SBU centre-of-mass positions.

    Works for any topology where the point group acts faithfully on the node
    positions — i.e. no non-identity symmetry operation fixes every node
    centre.  This covers:

    * 4 nodes at tetrahedral vertices  (Td,  23 non-identity ops)
    * 6 nodes at octahedral vertices   (Oh,  47 non-identity ops)
    * 8 nodes at cubic vertices        (Oh,  47 non-identity ops)

    Limitation: prism / bipyramid topologies have operations (e.g. the C4
    rotation of a D4h bipyramid) that fix all node positions — those are
    invisible here.  Use the topology-specific function for those cases.

    Algorithm
    ---------
    For every permutation of the N node centres:

    1. Compute the best-fit rigid transformation via SVD.
    2. Validate: rotate source nodes and check that every result lands within
       ``tol`` Å of its target.  Rejects non-symmetry permutations.
    3. Apply the validated rotation to the FG anchors; record the permutation.
    4. Deduplicate and discard identity.

    Parameters
    ----------
    fg_anchor : ase.Atoms
    node_centers : array-like, shape (N, 3)
        Metal SBU centre-of-mass positions.  N must be ≤ 8.
    tol : float
        Validation distance tolerance in Å (default 0.5).

    Returns
    -------
    list of tuple  — slot-major transformation library.

    Raises
    ------
    ValueError
        If more than 8 node centres are supplied.
    """
    node_positions = np.array(node_centers, dtype=float)
    n_nodes = len(node_positions)

    if n_nodes > 8:
        raise ValueError(
            f"universal_transformation_library: {n_nodes} nodes → "
            f"{math.factorial(n_nodes):,} permutations (limit: 8 / 40 320). "
            "Use a topology-specific function for larger cages."
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
        Rt = U @ Vt  # no det correction → includes improper operations

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


def spacegroup_transformation_library(cage_atoms, fg_anchor_indices, symprec=1e-2, atom_tol=0.5):
    """
    Transformation library for a periodic structure, from its actual
    crystallographic space group (via spglib) rather than a molecular
    point group.

    universal_transformation_library and every per-topology function in
    this module derive a *finite* point group (rotations/reflections
    about a molecular centre) - correct for a discrete, STK-built cage,
    but wrong for a periodic MOF: a unit cell's FG anchors are related
    to each other by the full space group, including translations
    (and non-symmorphic operations like screw axes/glide planes), most
    of which a point-group-only model can't see at all. Using the wrong
    model doesn't just miss a few equivalences - on a real MOF unit
    cell this left the isomer count too large by roughly nine orders of
    magnitude (192 true space-group operations vs. the ~24 a
    point-group fit found in the node positions alone).

    Algorithm
    ---------
    1. Get every (rotation, translation) operation of ``cage_atoms``'s
       space group from spglib, in fractional coordinates.
    2. For each op, apply it to every FG anchor's fractional position,
       wrap into the unit cell, and match each image back to the anchor
       whose (periodic-wrapped) position it coincides with.
    3. Discard the identity operation and any duplicate permutation.

    Parameters
    ----------
    cage_atoms : ase.Atoms
        The full periodic structure (must have ``pbc`` and a real cell -
        this is what the space group is computed from, not just the
        anchor atoms).
    fg_anchor_indices : sequence of int
        Indices into ``cage_atoms`` of the FG anchor atoms, in the same
        order ``enumerate_isomers`` indexes isomer descriptor slots.
    symprec : float, default 1e-2
        Tolerance (fractional coordinate units, roughly) passed to
        spglib for detecting the space group. Real (possibly slightly
        noisy) structures typically need this looser than spglib's own
        default (1e-5).
    atom_tol : float, default 0.5
        Cartesian distance (Angstrom) within which a symmetry image of
        one anchor is considered a match to another. Loose enough for
        typical refinement noise, tight enough that it won't be
        confused with a neighbouring anchor - real anchor atoms are
        never anywhere near this close to each other.

    Returns
    -------
    list of tuple — slot-major transformation library, same format as
    universal_transformation_library.

    Raises
    ------
    RuntimeError
        If spglib can't determine a space group, or if a symmetry
        operation doesn't map the anchor set onto itself within
        ``atom_tol`` (every operation of the structure's own space
        group must do this by definition - a failure here means the
        input structure doesn't actually satisfy the symmetry spglib
        detected, usually because symprec is too loose for how
        precisely the structure was built/relaxed).
    """
    import spglib

    lattice = np.asarray(cage_atoms.cell[:], dtype=float)
    dataset = spglib.get_symmetry_dataset(
        (lattice, cage_atoms.get_scaled_positions(), cage_atoms.numbers),
        symprec=symprec,
    )
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
                    "symprec - try tightening symprec or loosening atom_tol."
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
