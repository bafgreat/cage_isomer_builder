#!/usr/bin/env python3
"""
Point-group detection for finite structures (cages, molecules) from the
atoms, with no assumption about topology, orientation or the number of
nodes.

Why not spglib for finite cages?
--------------------------------
spglib finds the symmetry of a *crystal*. A lattice only allows rotations of
order 1, 2, 3, 4 and 6 (the crystallographic restriction), so spglib can
never report the C5 axes of an icosahedral (Ih) or D5h cage, or the S8 axis
of a D4d square antiprism. Periodic structures keep using spglib (see
``symmetry.spacegroup_transformation_library``); finite ones use this module.

Algorithm
---------
1. Centre the structure on its centroid. Every symmetry operation of a
   finite point set fixes its centroid, so all operations are 3x3
   orthogonal matrices about that point.
2. Pick two reference atoms ``a`` and ``b`` (not collinear with the centre)
   from the rarest (element, radius) classes. Any symmetry operation R must
   send ``a`` to an atom ``a'`` of the same element at the same radius, and
   ``b`` to a same-element atom ``b'`` at the same radius *and* the same
   distance from ``a'``. Two such pairs fix R up to handedness, so each
   candidate pair gives one proper and one improper candidate matrix. This
   search is exhaustive: no symmetry operation can be missed.
3. Validate each candidate on *every* atom: each image must land within
   ``tol`` of an atom of the same element, one-to-one. Then refine R by an
   orthogonal Procrustes fit over all matched atoms and record the largest
   remaining deviation.
4. Check that the accepted operations are closed under composition (a
   group). A tolerance that is too loose or too tight for a slightly
   distorted structure then raises an error instead of giving a wrong
   isomer count.

The result is the symmetry group as permutations of all atoms, which
restricts directly to the FG anchor slots.
"""
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PointGroup:
    """
    Symmetry of a finite structure.

    Attributes
    ----------
    name : str
        Schoenflies symbol, e.g. ``"Td"``, ``"D4h"``, ``"Ih"``.
    rotations : np.ndarray, shape (|G|, 3, 3)
        Orthogonal matrices (row-vector convention: ``x' = x @ R``), about
        ``centre``. Index 0 is the identity.
    atom_perms : np.ndarray, shape (|G|, n_atoms)
        ``atom_perms[g, i]`` is the atom that atom ``i`` is sent to.
    centre : np.ndarray, shape (3,)
    max_deviation : float
        Largest distance (Angstrom) between a transformed atom and the atom
        it was matched to, over all operations: how exact the symmetry is.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.pointgroup import detect_point_group
    >>> pg = detect_point_group(molecule("CH4"))
    >>> pg.name, pg.order, pg.rotations.shape
    ('Td', 24, (24, 3, 3))
    """
    name: str
    rotations: np.ndarray
    atom_perms: np.ndarray
    centre: np.ndarray
    max_deviation: float

    @property
    def order(self):
        """
        Number of symmetry operations, including the identity.

        Examples
        --------
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.pointgroup import detect_point_group
        >>> detect_point_group(molecule("C6H6")).order
        24
        """
        return len(self.rotations)


def _frame(u, v):
    e1 = u / np.linalg.norm(u)
    e2 = v - (v @ e1) * e1
    e2 /= np.linalg.norm(e2)
    return np.array([e1, e2, np.cross(e1, e2)])


def _procrustes(source, target, det):
    """Orthogonal M with determinant sign ``det`` minimising
    |source @ M - target| (Kabsch). Fixing the sign matters when the atoms
    are coplanar: the fit alone can't tell a rotation from its mirror image."""
    u, _, vt = np.linalg.svd(source.T @ target)
    d = np.sign(np.linalg.det(u @ vt)) * np.sign(det)
    return u @ np.diag([1.0, 1.0, d]) @ vt


def _same_matrix(r1, r2, atol=0.15):
    """Same operation, allowing for the fit noise of a real structure."""
    return np.abs(r1 - r2).max() < atol


def _match(positions, numbers, image, trees, tol):
    """Atom permutation sending each atom to the atom its image lands on,
    or None if any image has no same-element atom within tol, or two
    images land on the same atom."""
    perm = np.empty(len(positions), dtype=np.int64)
    worst = 0.0
    for z, (tree, members) in trees.items():
        mask = numbers == z
        dist, idx = tree.query(image[mask], distance_upper_bound=tol)
        if np.any(np.isinf(dist)):
            return None, None
        perm[mask] = members[idx]
        worst = max(worst, float(dist.max()))
    if len(np.unique(perm)) != len(perm):
        return None, None
    return perm, worst


def _same_shell(radii, numbers, i, tol):
    return np.nonzero((numbers == numbers[i]) & (np.abs(radii - radii[i]) <= tol))[0]


def detect_point_group(atoms, tol=0.3, min_radius=1.0):
    """
    Find every point symmetry operation of a finite structure.

    Parameters
    ----------
    atoms : ase.Atoms
        Finite structure. Element types matter: an operation must send every
        atom to an atom of the same element.
    tol : float, default 0.3
        Largest allowed distance (Angstrom) between a transformed atom and
        its partner. Must be well below the shortest distance between two
        same-element atoms (so matches are unambiguous) and above the
        structure's own distortion from perfect symmetry.
    min_radius : float, default 1.0
        Reference atoms closer than this to the centre are not used to fix
        orientations (their direction is too sensitive to noise). For a
        structure smaller than that (e.g. water), half the largest distance
        from the centre is used instead.

    Returns
    -------
    PointGroup

    Raises
    ------
    ValueError
        If the structure is (nearly) linear or the operations found do not
        form a closed group (see the module docstring).

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.pointgroup import detect_point_group
    >>> [detect_point_group(molecule(m)).name for m in ("H2O", "NH3", "C6H6", "CH4")]
    ['C2v', 'C3v', 'D6h', 'Td']
    >>> from cage_isomer_builder.utils.symmetry import ideal_orbit   # works for any order
    >>> detect_point_group(ideal_orbit("Ih"), tol=0.1).name
    'Ih'
    """
    positions = np.asarray(atoms.positions, dtype=float)
    numbers = np.asarray(atoms.numbers)
    n_atoms = len(positions)
    centre = positions.mean(axis=0)
    centred = positions - centre
    radii = np.linalg.norm(centred, axis=1)

    trees = {}
    for z in np.unique(numbers):
        members = np.nonzero(numbers == z)[0]
        trees[z] = (cKDTree(centred[members]), members)

    identity = np.arange(n_atoms)
    if radii.max() < 1e-6:                      # a single point: nothing to orient by
        return PointGroup('C1', np.eye(3)[None], identity[None], centre, 0.0)
    # Small molecules (water, NH3) have no atom min_radius from the centre;
    # use half the largest radius there instead of giving up.
    usable = np.nonzero(radii >= min(min_radius, 0.5 * radii.max()))[0]

    shell_size = {i: len(_same_shell(radii, numbers, i, tol)) for i in usable}
    order_a = sorted(usable, key=lambda i: (shell_size[i], -radii[i]))
    a = order_a[0]
    unit_a = centred[a] / radii[a]
    sin_ab = np.linalg.norm(np.cross(unit_a, centred[usable]), axis=1) / radii[usable]
    off_axis = usable[sin_ab > 0.2]
    if len(off_axis) == 0:
        raise ValueError("structure is (nearly) linear; point group detection "
                         "needs at least one atom off the line through the centre.")
    b = min(off_axis, key=lambda i: (shell_size[i], -abs(np.cross(unit_a, centred[i])).sum()))

    source = _frame(centred[a], centred[b])
    d_ab = np.linalg.norm(centred[a] - centred[b])
    flip = np.diag([1.0, 1.0, -1.0])

    # Operations are kept by matrix, not by atom permutation: when the
    # reference atoms are coplanar, the mirror through their plane moves no
    # atom but is a distinct symmetry operation.
    ops = [(np.eye(3), identity, 0.0)]
    for a2 in _same_shell(radii, numbers, a, tol):
        for b2 in _same_shell(radii, numbers, b, tol):
            if b2 == a2 or abs(np.linalg.norm(centred[a2] - centred[b2]) - d_ab) > 2 * tol:
                continue
            target = _frame(centred[a2], centred[b2])
            for hand in (np.eye(3), flip):
                rot = source.T @ hand @ target          # rows: x' = x @ rot
                perm, _ = _match(centred, numbers, centred @ rot, trees, tol)
                if perm is None:
                    continue
                refined = _procrustes(centred, centred[perm], np.linalg.det(rot))
                perm2, worst = _match(centred, numbers, centred @ refined, trees, tol)
                if perm2 is None or not np.array_equal(perm, perm2):
                    refined, worst = rot, float(np.linalg.norm(
                        centred @ rot - centred[perm], axis=1).max())
                if not any(_same_matrix(refined, op[0]) for op in ops):
                    ops.append((refined, perm, worst))

    rotations = np.array([op[0] for op in ops])
    atom_perms = np.array([op[1] for op in ops])
    max_dev = max(op[2] for op in ops)

    products = np.einsum('aij,bjk->abik', rotations, rotations).reshape(-1, 1, 3, 3)
    gap = np.abs(products - rotations[None]).max(axis=(2, 3)).min(axis=1)
    if gap.max() >= 0.15:                      # same threshold as _same_matrix
        raise ValueError(
            f"symmetry operations found at tol={tol} Angstrom are not closed "
            "under composition; the structure is too distorted for this "
            "tolerance. Optimise it, or adjust tol."
        )

    return PointGroup(_schoenflies(rotations), rotations, atom_perms, centre, max_dev)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def _axis_and_order(rot):
    """Axis and order of the proper part of an orthogonal matrix."""
    proper = rot * np.sign(np.linalg.det(rot))
    cos_t = np.clip((np.trace(proper) - 1) / 2, -1, 1)
    theta = np.arccos(cos_t)
    if theta < 1e-3:
        return None, 1
    w, v = np.linalg.eig(proper.T)
    axis = np.real(v[:, np.argmin(np.abs(w - 1))])
    axis /= np.linalg.norm(axis)
    return axis, int(round(2 * np.pi / theta))


def _schoenflies(rotations):
    """Schoenflies symbol of a finite point group given as matrices."""
    dets = np.sign(np.linalg.det(rotations))
    proper = rotations[dets > 0]
    improper = rotations[dets < 0]
    has_inversion = any(np.allclose(r, -np.eye(3), atol=1e-2) for r in improper)

    axes = []      # [axis, max order]
    for r in proper:
        axis, n = _axis_and_order(r)
        if axis is None:
            continue
        for entry in axes:
            if abs(abs(entry[0] @ axis) - 1) < 1e-2:
                entry[1] = max(entry[1], n)
                break
        else:
            axes.append([axis, n])

    mirror_normals = []
    for r in improper:
        axis, n = _axis_and_order(r)
        if axis is not None and n == 2:          # -C2 = reflection through plane _|_ axis
            mirror_normals.append(axis)

    def count(order):
        return sum(1 for _, n in axes if n == order)

    if count(5) >= 2:
        return 'Ih' if len(improper) else 'I'
    if sum(1 for _, n in axes if n >= 3) >= 4:
        if count(4):
            return 'Oh' if len(improper) else 'O'
        if not len(improper):
            return 'T'
        return 'Th' if has_inversion else 'Td'

    n = max((k for _, k in axes), default=1)
    n_proper = len(proper)
    if n == 1:
        if not len(improper):
            return 'C1'
        return 'Ci' if has_inversion else 'Cs'

    if n == 2 and n_proper == 4:                 # three C2 axes: D2 family
        if not len(improper):
            return 'D2'
        return 'D2h' if has_inversion else 'D2d'

    principal = next(a for a, k in axes if k == n)
    sigma_h = any(abs(abs(m @ principal) - 1) < 1e-2 for m in mirror_normals)
    if n_proper == 2 * n:
        if not len(improper):
            return f'D{n}'
        return f'D{n}h' if sigma_h else f'D{n}d'
    if not len(improper):
        return f'C{n}'
    if sigma_h:
        return f'C{n}h'
    if mirror_normals:
        return f'C{n}v'
    return f'S{2 * n}'


# ---------------------------------------------------------------------------
# Reference atoms
# ---------------------------------------------------------------------------

_TRANSITION_METALS = frozenset(
    list(range(21, 31)) + list(range(39, 49)) + list(range(57, 81)) + list(range(89, 113))
)


def reference_indices(atoms, reference="auto"):
    """
    Atoms the symmetry operations are detected from.

    ``"all"``: every atom (strict: the whole structure must be symmetric).
    ``"heavy"``: everything except H and the ``X`` site markers.
    ``"metals"``: transition-metal/lanthanide/actinide atoms.
    ``"auto"`` (default): metals unless they are fewer than three or all on
    one line, else heavy atoms.

    "all" is often too strict: after optimisation, linkers settle at
    slightly different ring twists (a 10-degree twist moves an aromatic H by
    about 0.4 A) while the metal framework keeps the cage's symmetry. The
    anchor slots are then matched under each framework operation with a
    looser tolerance (see :func:`pointgroup_transformation_library`).

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.pointgroup import reference_indices
    >>> reference_indices(molecule("C6H6"), "heavy").tolist()     # metals absent: heavy atoms
    [0, 1, 2, 3, 4, 5]
    """
    numbers = np.asarray(atoms.numbers)
    symbols = atoms.get_chemical_symbols()
    if reference == "all":
        return np.arange(len(atoms))
    heavy = np.array([i for i, s in enumerate(symbols) if s not in ("H", "X", "At")])
    metals = np.array([i for i, z in enumerate(numbers) if z in _TRANSITION_METALS])
    if reference == "heavy":
        return heavy
    if reference == "metals":
        return metals
    if reference == "auto":
        # Metals fix the orientation only if they do not all lie on one line
        # (e.g. the two Pd of a Pd2L4 lantern); otherwise use heavy atoms.
        if len(metals) >= 3:
            spread = np.linalg.svd(atoms.positions[metals] - atoms.positions[metals].mean(axis=0),
                                   compute_uv=False)
            if spread[1] > 0.5:
                return metals
        return heavy
    raise ValueError(f"reference={reference!r}; use 'auto', 'metals', 'heavy' or 'all'.")


# ---------------------------------------------------------------------------
# FG slots
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlotSymmetry:
    """
    Symmetry of the FG anchor slots.

    Attributes
    ----------
    library : list of tuple
        Slot-major transformation library (see symmetry.py).
    framework : PointGroup
        Point group of the reference atoms.
    n_kept : int
        Framework operations that also map the anchors onto themselves
        (including the identity). Fewer than ``framework.order`` means the
        anchors are less symmetric than the framework.
    max_anchor_deviation : float
        Largest anchor-to-partner distance over the kept operations.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.pointgroup import slot_symmetry
    >>> benzene = molecule("C6H6")
    >>> result = slot_symmetry(benzene, list(range(6, 12)), reference="heavy", tol=0.1)
    >>> result.framework.name, result.n_kept
    ('D6h', 24)
    """
    library: list
    framework: PointGroup
    n_kept: int
    max_anchor_deviation: float


def slot_symmetry(atoms, fg_anchor_indices, reference="auto", tol=0.5,
                  anchor_tol=1.0, point_group=None):
    """
    Symmetry operations of a finite structure restricted to its FG anchor
    slots.

    1. Detect the point group of the reference atoms (:func:`reference_indices`)
       at ``tol``, exhaustively and checked to be a group.
    2. For every operation, map the anchors with a one-to-one (Hungarian)
       assignment; keep the operation only if every anchor lands within
       ``anchor_tol`` of its partner.

    The kept set is checked for closure (and linker blocks) by the caller
    via ``rgroup.prepare_group``.

    Parameters
    ----------
    atoms : ase.Atoms
    fg_anchor_indices : sequence of int
        Anchor atoms in slot order.
    reference : str, default "auto"
    tol : float, default 0.5
        Matching tolerance (Angstrom) for the reference atoms.
    anchor_tol : float, default 1.0
        Matching tolerance (Angstrom) for the anchors. Must stay well below
        the anchor-anchor distance on one ring (~2.5 A for aromatic H).
    point_group : PointGroup, optional
        Reuse an already detected framework group.

    Returns
    -------
    SlotSymmetry

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.pointgroup import slot_symmetry
    >>> benzene = molecule("C6H6")                     # the six H as slots
    >>> result = slot_symmetry(benzene, list(range(6, 12)), reference="heavy", tol=0.1)
    >>> len(result.library), round(result.max_anchor_deviation, 6)
    (6, 0.0)
    """
    from scipy.optimize import linear_sum_assignment

    if point_group is None:
        ref = reference_indices(atoms, reference)
        point_group = detect_point_group(atoms[ref], tol=tol)
    anchors = np.asarray(atoms.positions[list(fg_anchor_indices)], dtype=float)
    centred = anchors - point_group.centre
    n = len(anchors)

    library, seen, worst, kept = [], set(), 0.0, 1
    for rot in point_group.rotations[1:]:
        image = centred @ rot
        cost = np.linalg.norm(image[:, None, :] - centred[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        dist = cost[rows, cols]
        if dist.max() > anchor_tol:
            continue
        kept += 1
        worst = max(worst, float(dist.max()))
        slots = tuple(int(c) for c in cols[np.argsort(rows)])
        if slots != tuple(range(n)) and slots not in seen:
            seen.add(slots)
            library.append(slots)
    library = list(zip(*library)) if library else [() for _ in range(n)]
    return SlotSymmetry(library, point_group, kept, worst)


def pointgroup_transformation_library(atoms, fg_anchor_indices, reference="auto",
                                      tol=0.5, anchor_tol=1.0, point_group=None):
    """
    Transformation library of a finite structure from its detected point
    group: the finite counterpart of
    ``symmetry.spacegroup_transformation_library``, in the same slot-major
    format. See :func:`slot_symmetry` for the parameters.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.pointgroup import pointgroup_transformation_library
    >>> library = pointgroup_transformation_library(molecule("C6H6"), list(range(6, 12)),
    ...                                             reference="heavy", tol=0.1)
    >>> sorted(set(op[0] for op in zip(*library)))     # slot 0 reaches every H (mirrors fix it)
    [0, 1, 2, 3, 4, 5]
    """
    return slot_symmetry(atoms, fg_anchor_indices, reference, tol, anchor_tol,
                         point_group).library
