#!/usr/bin/env python3
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.data import covalent_radii


def _unit(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def random_rotation_matrix():
    """
    A uniformly random 3x3 rotation matrix.

    Two independent random vectors are orthonormalised (Gram-Schmidt) and
    completed to a right-handed basis, which samples SO(3) uniformly.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        A proper rotation matrix (determinant +1).

    Examples
    --------
    >>> import numpy as np
    >>> from cage_isomer_builder.utils.geometry import random_rotation_matrix
    >>> r = random_rotation_matrix()
    >>> bool(np.allclose(r @ r.T, np.eye(3))), round(float(np.linalg.det(r)), 6)
    (True, 1.0)
    """
    v1 = np.random.randn(3)
    v1 /= np.linalg.norm(v1)
    v2 = np.random.randn(3)
    v2 -= np.dot(v2, v1) * v1
    v2 /= np.linalg.norm(v2)
    v3 = np.cross(v1, v2)
    return np.column_stack([v1, v2, v3])


def find_third_point(p1, p2, bond_length):
    """
    Extrapolate a third point P3 beyond P1, away from P2:
    ``R-P2-P1----P3``. Used to construct a second reference direction
    for aligning a fragment onto a single bonding site.

    Parameters
    ----------
    p1 : np.ndarray, shape (3,)
    p2 : np.ndarray, shape (3,)
    bond_length : float
        Distance of P3 from P1.

    Returns
    -------
    np.ndarray, shape (3,)
        Coordinates of P3.

    Examples
    --------
    >>> import numpy as np
    >>> from cage_isomer_builder.utils.geometry import find_third_point
    >>> find_third_point(np.array([1.0, 0, 0]), np.array([0.0, 0, 0]), 1.5).tolist()
    [2.5, 0.0, 0.0]
    """
    d = p2 - p1
    unit_vector = d / np.linalg.norm(d)
    return p1 - bond_length * unit_vector


def kabsch(coordinates1, coordinates2):
    """
    Kabsch algorithm: the optimal rotation matrix aligning one point
    set onto another.

    Parameters
    ----------
    coordinates1 : np.ndarray
        Coordinates of the first point set.
    coordinates2 : np.ndarray
        Coordinates of the second point set.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Rotation matrix that best aligns coordinates1 onto coordinates2.

    Examples
    --------
    >>> import numpy as np
    >>> from cage_isomer_builder.utils.geometry import kabsch
    >>> a = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    >>> rz = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])        # 90 degrees about z
    >>> bool(np.allclose(a @ kabsch(a, a @ rz), a @ rz))
    True
    """
    covariant_matrix = np.dot(np.transpose(coordinates1), coordinates2)
    left_singular_vectors, eigen_value, right_singular_vectors = np.linalg.svd(covariant_matrix)
    d = (np.linalg.det(left_singular_vectors) * np.linalg.det(right_singular_vectors)) < 0.0
    if d:
        eigen_value[-1] = -eigen_value[-1]
        left_singular_vectors[:, -1] = -left_singular_vectors[:, -1]
    rotation_matrix = np.dot(left_singular_vectors, right_singular_vectors)
    return rotation_matrix


def find_connected_atoms(atoms, atom_index):
    """
    Indices of every atom bonded to ``atom_index`` (e.g. the real atom
    a dummy "X" marker is attached to), via a per-atom covalent-radius
    cutoff.

    Parameters
    ----------
    atoms : ase.Atoms
    atom_index : int
        Index of the atom whose bonded neighbours are wanted.

    Returns
    -------
    np.ndarray of int
        Indices of atoms bonded to ``atom_index``.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.geometry import find_connected_atoms
    >>> sorted(find_connected_atoms(molecule("H2O"), 0).tolist())
    [1, 2]
    """
    cutoffs = covalent_radii[atoms.numbers] + 0.3
    # skin=0: NeighborList adds its skin to the cutoff, which on top of the
    # +0.3 above could pick up a non-bonded ring neighbour.
    nl = NeighborList(cutoffs=cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)
    indices, _ = nl.get_neighbors(atom_index)
    return indices


def bonded_pairs(atoms, mult=1.2, exclude=None):
    """
    All bonded atom pairs (i, j), i < j, from interatomic distances.

    Uses ``ase.neighborlist.natural_cutoffs`` (``mult`` times each covalent
    radius) with ``skin=0``. ASE's default skin of 0.3 is added to the bonding
    cutoff, which would bond the meta and para carbons of an aromatic ring
    (about 2.4 and 2.8 Angstrom apart) and corrupt the molecular graph.

    Parameters
    ----------
    atoms : ase.Atoms
    mult : float, default 1.2
        Covalent-radius multiplier passed to ``natural_cutoffs``.
    exclude : int, optional
        Atom index to skip (e.g. a marker atom whose bonds the caller tracks).

    Returns
    -------
    list of (int, int)
        Each bonded pair once, smaller index first.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.geometry import bonded_pairs
    >>> bonded_pairs(molecule("H2O"))
    [(0, 1), (0, 2)]
    >>> len(bonded_pairs(molecule("C6H6")))         # 6 C-C + 6 C-H, no meta/para pairs
    12
    """
    cutoffs = natural_cutoffs(atoms, mult=mult)
    nl = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)
    pairs = []
    for i in range(len(atoms)):
        if i == exclude:
            continue
        for j in nl.get_neighbors(i)[0]:
            if j > i and j != exclude:
                pairs.append((i, int(j)))
    return pairs


def align_fragment_orientation(host_system, sub_fragment, bond_length,
                               host_marker='X', fragment_marker='X',
                               host_index=None, host_neigh_index=None):
    """
    Rotate ``sub_fragment`` in place so that its marker bond lies along the
    bonding direction of a site in ``host_system``.

    A second reference point (``find_third_point``) is placed on the host's bond
    axis so that two point pairs can be aligned. Apply
    ``resolve_fragment_twist`` afterwards to fix the rotation about that axis.

    Parameters
    ----------
    host_system : ase.Atoms
        Host structure containing the bonding site.
    sub_fragment : ase.Atoms
        Fragment with one marker atom. Rotated in place.
    bond_length : float
        Distance of the reference point from the site atom.
    host_marker : str, default 'X'
        Symbol of the site atom in the host (``X``, as written by
        ``full_functionalisation``).
    fragment_marker : str, default 'X'
        Symbol of the marker atom in ``sub_fragment``.
    host_index : int, optional
        Index of the host site to use, instead of the first ``host_marker``
        atom. Needed when the host has several sites.
    host_neigh_index : int, optional
        Index of the site's bonded neighbour, instead of searching for it with
        ``find_connected_atoms``. A marker atom has no meaningful covalent
        radius, so a distance search can pick the wrong atom in a compact
        structure; pass the neighbour from the host's bond matrix when known.

    Returns
    -------
    sub_fragment : ase.Atoms
        The aligned fragment (the same object).
    sub_x : int
        Index of the marker atom in ``sub_fragment``.
    host_x : int
        Index of the site atom in ``host_system``.
    sub_neigh : int
        Index of the fragment atom bonded to its marker.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from ase.io import read
    >>> from cage_isomer_builder import example_data
    >>> host = molecule("C6H6")
    >>> host[6].symbol = "X"                       # a site on C 0
    >>> nh2 = example_data.read("NH2")             # N, H, H and its X attachment marker
    >>> from cage_isomer_builder.utils.geometry import align_fragment_orientation
    >>> fragment, sub_x, host_x, sub_neigh = align_fragment_orientation(
    ...     host, nh2.copy(), 1.5, host_index=6, host_neigh_index=0)
    >>> host_x, nh2[sub_neigh].symbol
    (6, 'N')
    """
    if host_index is not None:
        host_x = host_index
    else:
        host_x = [atom.index for atom in host_system if atom.symbol == host_marker][0]
    sub_x = [atom.index for atom in sub_fragment if atom.symbol == fragment_marker][0]

    sub_neigh = find_connected_atoms(sub_fragment, sub_x)[0]
    if host_neigh_index is not None:
        host_neigh = host_neigh_index
    else:
        host_neigh = find_connected_atoms(host_system, host_x)[0]

    third_point = find_third_point(host_system.positions[host_x],
                                   host_system.positions[host_neigh], bond_length)

    host_x_coords = np.array([host_system.positions[host_x].tolist(),
                              third_point.tolist()])
    sub_x_coords = sub_fragment.positions[[sub_x, sub_neigh]]

    host_x_coords -= host_system[host_x].position
    sub_x_coords -= sub_fragment[sub_x].position

    rotation_matrix = kabsch(sub_x_coords, host_x_coords)
    sub_fragment.positions -= sub_fragment[sub_x].position
    sub_fragment.positions = np.dot(sub_fragment.positions, rotation_matrix)
    sub_fragment.positions += sub_fragment[sub_x].position

    return sub_fragment, sub_x, host_x,  sub_neigh


def resolve_fragment_twist(host_system, sub_fragment, host_x, sub_neigh,
                           tolerance=0.75, n_angles=24, host_neigh_index=None):
    """
    Rotate a translated fragment about its bond axis to avoid clashing with the
    host.

    ``align_fragment_orientation`` fixes only the direction of the bond, so the
    rotation about it is arbitrary; for a fragment of more than one atom (e.g.
    NH2) that can point an atom into the host. ``n_angles`` rotations about the
    bond axis are tried and the one with the largest closest fragment-host
    approach (scaled by covalent radii) is kept.

    Parameters
    ----------
    host_system : ase.Atoms
        Host to avoid. Its site atom ``host_x`` is left out of the check, since
        ``sub_neigh`` sits at that position.
    sub_fragment : ase.Atoms
        Fragment already translated so that ``sub_fragment[sub_neigh]``
        coincides with ``host_system[host_x]``.
    host_x : int
        Index of the site atom in ``host_system``.
    sub_neigh : int
        Index of the fragment atom bonded to its marker; the pivot of the
        rotation.
    tolerance : float, default 0.75
        Fraction of the summed covalent radii used as the clash threshold.
    n_angles : int, default 24
        Number of evenly spaced trial angles in [0, 360).
    host_neigh_index : int, optional
        Index of ``host_x``'s bonded neighbour, as for
        :func:`align_fragment_orientation`.

    Returns
    -------
    ase.Atoms
        ``sub_fragment``, rotated in place to the best angle.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from ase.io import read
    >>> from cage_isomer_builder import example_data
    >>> host = molecule("C6H6")
    >>> host[6].symbol = "X"                       # a site on C 0
    >>> nh2 = example_data.read("NH2")             # N, H, H and its X attachment marker
    >>> from cage_isomer_builder.utils.geometry import (
    ...     align_fragment_orientation, resolve_fragment_twist, translate_fragment)
    >>> fragment, sub_x, host_x, sub_neigh = align_fragment_orientation(
    ...     host, nh2.copy(), 1.5, host_index=6, host_neigh_index=0)
    >>> fragment = translate_fragment(host, fragment, sub_neigh, host_x)
    >>> fragment = resolve_fragment_twist(host, fragment, host_x, sub_neigh, host_neigh_index=0)
    >>> import numpy as np
    >>> hydrogens = [i for i in range(len(fragment)) if fragment[i].symbol == "H"]
    >>> others = [i for i in range(len(host)) if i != host_x]
    >>> gaps = np.linalg.norm(fragment.positions[hydrogens][:, None]
    ...                       - host.positions[others][None], axis=2)
    >>> bool(gaps.min() > 1.5)                  # the NH2 hydrogens clear the ring
    True
    """
    others = [i for i in range(len(sub_fragment)) if i != sub_neigh]
    if not others:
        return sub_fragment

    # The bond axis runs from the host's neighbour atom (e.g. the ring
    # carbon) to the pivot; the pivot already sits on host_x, so host_x
    # itself cannot define the axis.
    if host_neigh_index is not None:
        host_neigh = host_neigh_index
    else:
        host_neigh = find_connected_atoms(host_system, host_x)[0]
    pivot = sub_fragment[sub_neigh].position
    axis_vector = pivot - host_system[host_neigh].position
    if np.linalg.norm(axis_vector) < 1e-8:
        return sub_fragment
    axis = _unit(axis_vector)

    host_mask = [i for i in range(len(host_system)) if i != host_x]
    if not host_mask:
        return sub_fragment
    host_pos = host_system.positions[host_mask]
    host_radii = covalent_radii[host_system.numbers[host_mask]]

    rel = sub_fragment.positions[others] - pivot
    frag_radii = covalent_radii[sub_fragment.numbers[others]]
    radii_sum = frag_radii[:, None] + host_radii[None, :]

    best_angle, best_score = 0.0, -np.inf
    for angle in np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False):
        trial = pivot + _rotate_about_axis(rel, axis, angle)
        dists = np.linalg.norm(trial[:, None, :] - host_pos[None, :, :], axis=-1)
        score = (dists - tolerance * radii_sum).min()
        if score > best_score:
            best_score, best_angle = score, angle

    new_positions = sub_fragment.positions.copy()
    new_positions[others] = pivot + _rotate_about_axis(rel, axis, best_angle)
    sub_fragment.positions = new_positions
    return sub_fragment


def _rotate_about_axis(vectors, axis, angle):
    """Rodrigues' rotation formula, applied to an (N, 3) array of vectors."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return (
        vectors * cos_a
        + np.cross(axis, vectors) * sin_a
        + axis * np.einsum('ij,j->i', vectors, axis)[:, None] * (1 - cos_a)
    )


def translate_fragment(host_system, sub_fragment, sub_neigh, host_x):
    """
    Translate an aligned fragment so its bonded atom coincides with the
    host's site atom, forming the new bond.

    Parameters
    ----------
    host_system : ase.Atoms
        Host structure containing the bonding site.
    sub_fragment : ase.Atoms
        Aligned fragment to translate.
    sub_neigh : int
        Index of the fragment atom bonded to its own dummy atom.
    host_x : int
        Index of the site atom in ``host_system``.

    Returns
    -------
    sub_fragment : ase.Atoms
        The translated fragment (same object, moved in place).

    Examples
    --------
    >>> from ase.build import molecule
    >>> from ase.io import read
    >>> from cage_isomer_builder import example_data
    >>> host = molecule("C6H6")
    >>> host[6].symbol = "X"                       # a site on C 0
    >>> nh2 = example_data.read("NH2")             # N, H, H and its X attachment marker
    >>> import numpy as np
    >>> from cage_isomer_builder.utils.geometry import align_fragment_orientation, translate_fragment
    >>> fragment, sub_x, host_x, sub_neigh = align_fragment_orientation(
    ...     host, nh2.copy(), 1.5, host_index=6, host_neigh_index=0)
    >>> fragment = translate_fragment(host, fragment, sub_neigh, host_x)
    >>> bool(np.allclose(fragment.positions[sub_neigh], host.positions[host_x]))  # N on the site
    True
    """
    translation_vector = host_system[host_x].position - sub_fragment[sub_neigh].position
    sub_fragment.positions += translation_vector
    return sub_fragment

def reflect_positions(ase_atom, reflection_normal):
    """
    Reflect all positions of ``ase_atom`` through the plane through the origin
    with normal ``reflection_normal`` (Householder reflection
    ``H = I - 2 n n^T``, i.e. ``x -> x - 2 (x . n) n`` for the unit normal n).

    Parameters
    ----------
    ase_atom : ase.Atoms
        Structure to reflect. Its positions are changed in place.
    reflection_normal : numpy.ndarray, shape (3,)
        Normal of the plane; it is normalised internally. ``[0, 0, 1]``
        reflects through the XY plane, ``[1, 0, 0]`` through YZ, ``[0, 1, 0]``
        through XZ.

    Returns
    -------
    ase.Atoms
        The reflected structure (the same object).

    Examples
    --------
    >>> from ase import Atoms
    >>> from cage_isomer_builder.utils.geometry import reflect_positions
    >>> reflect_positions(Atoms("H", [[1.0, 2.0, 3.0]]), [0, 0, 1]).positions.tolist()
    [[1.0, 2.0, -3.0]]
    """

    unit_normal = reflection_normal / np.linalg.norm(reflection_normal)

    positions = ase_atom.positions

    ase_atom.positions[:] -= np.outer(
        2 * np.dot(positions, unit_normal),
        unit_normal
        )

    return ase_atom


def project_permutation(original_fg_anchors,
                        transformed_fg_anchors,
                        max_distance=1.0,
                        ):
    """
    The permutation that maps an anchor set onto its transformed copy.

    The match is a one-to-one assignment minimising the total distance
    (Hungarian algorithm, ``scipy.optimize.linear_sum_assignment``). Matching
    each point to its nearest neighbour independently can give two points the
    same partner when anchors are close together (e.g. the eight positions of a
    biphenyl linker); the assignment is always a bijection.

    Parameters
    ----------
    original_fg_anchors : ase.Atoms
        The anchors before the symmetry operation.
    transformed_fg_anchors : ase.Atoms
        The anchors after it; same number of atoms.
    max_distance : float, default 1.0
        Largest allowed distance (Angstrom) between a transformed anchor and its
        match.

    Returns
    -------
    list of int
        ``permutation[i]`` is the index in ``original_fg_anchors`` matched to
        anchor i of ``transformed_fg_anchors``.

    Raises
    ------
    RuntimeError
        If the best assignment leaves a pair further apart than
        ``max_distance``: the operation is not a symmetry of the structure.

    Examples
    --------
    >>> from ase import Atoms
    >>> from cage_isomer_builder.utils.geometry import project_permutation
    >>> original = Atoms("X3", [[0, 0, 0], [3, 0, 0], [6, 0, 0]])
    >>> moved = Atoms("X3", [[6, 0, 0], [0, 0, 0], [3, 0, 0]])
    >>> project_permutation(original, moved)
    [2, 0, 1]
    """
    from scipy.optimize import linear_sum_assignment

    orig_positions = original_fg_anchors.positions
    trans_positions = transformed_fg_anchors.positions
    cost = np.linalg.norm(
        orig_positions[:, None, :] - trans_positions[None, :, :], axis=-1
    )
    orig_idx, trans_idx = linear_sum_assignment(cost)

    worst = cost[orig_idx, trans_idx].max() if len(orig_idx) else 0.0
    if worst > max_distance:
        raise RuntimeError(
            f"project_permutation: the best one-to-one match leaves a pair "
            f"{worst:.3f} A apart (more than max_distance={max_distance}), so "
            "this transformation does not map the anchors onto themselves and "
            "is not a symmetry of this structure."
        )

    permutation = np.empty(len(trans_positions), dtype=int)
    permutation[trans_idx] = orig_idx
    return permutation.tolist()


