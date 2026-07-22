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
    Return a uniformly random 3×3 rotation matrix.

    Uses Gram-Schmidt orthonormalisation of two independent random
    vectors to construct a right-handed orthonormal basis, which is
    equivalent to sampling uniformly from SO(3).

    Returns
    -------
    R : numpy.ndarray of shape (3, 3)
        A random proper rotation matrix (det = +1).
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
    """
    cutoffs = covalent_radii[atoms.numbers] + 0.3
    # skin=0.0 - NeighborList's default skin=0.3 is an MD rebuild buffer that
    # also gets added to the bonding cutoff itself, which combined with the
    # already-generous +0.3 per-atom cutoff above risks pulling in a second,
    # non-bonded atom (e.g. a ring meta-neighbour) ahead of the real one.
    nl = NeighborList(cutoffs=cutoffs, skin=0.0, self_interaction=False, bothways=True)
    nl.update(atoms)
    indices, _ = nl.get_neighbors(atom_index)
    return indices


def bonded_pairs(atoms, mult=1.2, exclude=None):
    """
    All bonded atom-index pairs (i, j), i < j, via distance-based perception.

    Uses ``ase.neighborlist.natural_cutoffs`` (``mult`` times each atom's
    covalent radius) with ``skin=0.0``. The default ``skin=0.3`` is meant as
    an MD rebuild buffer, but ``NeighborList.get_neighbors()`` also counts
    anything inside ``cutoff_i + cutoff_j + skin`` as bonded - for a typical
    aromatic ring (meta C...C ~2.42 Angstrom) that silently adds every
    meta/para ring pair as a spurious "bond" alongside the real ones,
    corrupting the whole molecular graph (extra angle/torsion terms, wrong
    coordination number fed into UFF typing). Leaving skin at 0 avoids that.

    Parameters
    ----------
    atoms : ase.Atoms
    mult : float, default 1.2
        Covalent-radius multiplier passed to ``natural_cutoffs``.
    exclude : int, optional
        Atom index to skip entirely (e.g. a dummy/anchor atom whose own
        bonds the caller tracks separately).

    Returns
    -------
    list of (int, int)
        Each bonded pair once, with the smaller index first.
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
    Rotate ``sub_fragment`` in place so its own dummy-marker bond axis
    aligns with a bonding site's axis in ``host_system``.

    A second reference point (``find_third_point``) is constructed along
    the host's own bond axis so the alignment has two point pairs to work
    with; ``resolve_fragment_twist`` should be applied afterward to fix
    the remaining rotational freedom around that axis.

    Parameters
    ----------
    host_system : ase.Atoms
        Host structure containing the bonding site.
    sub_fragment : ase.Atoms
        Fragment to align, containing one dummy marker atom. Rotated
        in place.
    bond_length : float
        Distance of the constructed reference point from the site atom.
    host_marker : str, default 'X'
        Element symbol marking the bonding site in the host (use 'At'
        for a host prepared via ``full_functionalisation``).
    fragment_marker : str, default 'X'
        Element symbol marking the bonding atom in ``sub_fragment``.
    host_index : int, optional
        Use this atom index as the host bonding site instead of searching
        for the first atom matching ``host_marker``. Needed when the host
        has more than one marker atom (e.g. several active "At" sites on
        one isomer) and each must be addressed individually.
    host_neigh_index : int, optional
        Use this atom index as the site's real bonded neighbour instead of
        searching for it with ``find_connected_atoms``. Needed when the
        site atom has an unusually large covalent radius (e.g. "At", used
        as the FG anchor marker) - ``find_connected_atoms``'s cutoff is
        generous enough that in a compact structure it can pick up some
        other nearby atom instead of the true bonded one. Pass the
        neighbour looked up from the host's own bond matrix when it's
        known, rather than trusting distance alone.

    Returns
    -------
    sub_fragment : ase.Atoms
        The aligned fragment (same object, rotated in place).
    sub_x : int
        Index of the dummy atom in ``sub_fragment``.
    host_x : int
        Index of the site atom in ``host_system``.
    sub_neigh : int
        Index of the fragment atom bonded to its own dummy atom.
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
    Rotate an already-translated fragment around its own bond axis to avoid
    clashing with the host.

    align_fragment_orientation only fixes the bond-axis direction (it
    aligns two points that are collinear with that axis - see
    find_third_point), so rotation of the fragment *around* that axis is
    left arbitrary. For a single-atom fragment that is harmless, but for
    anything with more atoms (e.g. NH2) an arbitrary twist can point a
    fragment atom straight into a nearby host atom. This samples n_angles
    evenly-spaced twists around the bond axis and keeps whichever one
    maximises the closest fragment-host approach (covalent-radius scaled),
    i.e. the least (ideally zero) clash.

    Parameters
    ----------
    host_system : ase.Atoms
        Host structure to avoid clashing with (its own dummy/site atom at
        ``host_x`` is excluded from the clash check, since the fragment's
        ``sub_neigh`` atom is deliberately sitting at that same position
        after translate_fragment).
    sub_fragment : ase.Atoms
        Fragment already translated so ``sub_fragment[sub_neigh]`` coincides
        with ``host_system[host_x]``.
    host_x : int
        Index of the site atom in ``host_system``.
    sub_neigh : int
        Index of the fragment atom bonded to its own dummy atom - the pivot
        the rest of the fragment rotates around.
    tolerance : float
        Fraction of the summed covalent radii used as the clash threshold
        (default 0.75, matching ``_atoms_overlap`` elsewhere in this package).
    n_angles : int
        Number of evenly-spaced trial angles in [0, 360) (default 24, i.e.
        every 15 degrees).
    host_neigh_index : int, optional
        Use this specific atom index as host_x's real bonded neighbour
        instead of searching for it with find_connected_atoms - see
        align_fragment_orientation's own host_neigh_index for why (host_x
        is typically "At", whose unusually large covalent radius makes
        find_connected_atoms prone to picking up the wrong nearby atom in
        a compact structure).

    Returns
    -------
    ase.Atoms
        ``sub_fragment``, rotated in place to the best-scoring twist angle.
    """
    others = [i for i in range(len(sub_fragment)) if i != sub_neigh]
    if not others:
        return sub_fragment

    # The bond axis must come from the host's real neighbour atom (e.g. the
    # ring carbon), not from host_x itself - translate_fragment already made
    # sub_fragment[sub_neigh] and host_system[host_x] coincide, so
    # (pivot - host_x position) is a zero vector and can't define an axis.
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
    """
    translation_vector = host_system[host_x].position - sub_fragment[sub_neigh].position
    sub_fragment.positions += translation_vector
    return sub_fragment

def reflect_positions(ase_atom, reflection_normal):
    """
    Function to reflect all atomic positions in an ase_atom
    (molecule/material) through a plane defined by its
    normal vector using the Householder reflection formula:
        H = I - 2 * n̂ * n̂ᵀ

    This translates to:

        new_position -> position - 2 * (position . n̂) * n̂

    where n̂ is the unit normal vector of the reflection plane.

    Parameters
    ----------
    ase_atom : ase.Atoms
        The structure whose atomic positions will be reflected.
        Positions are modified in place.
    reflection_normal : numpy.ndarray of shape (3,)
        Normal vector defining the reflection plane passing through
        the origin. Does not need to be a unit vector — it is
        normalised internally.
        Examples:
            [0, 0, 1] -> reflect through the XY plane
            [1, 0, 0] -> reflect through the YZ plane
            [0, 1, 0] -> reflect through the XZ plane

    Returns
    -------
    new_ase_atom : ase.Atoms
        The structure with reflected atomic positions.
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
    A function to find the permutation mapping between an original set
    of functional group anchor positions and a transformed
    (rotated/reflected) set, via a globally-optimal one-to-one match
    (the Hungarian algorithm) rather than nearest-neighbour-per-point.

    Matching each transformed point to its independently-nearest
    original point (the previous approach) is not guaranteed to be a
    bijection: two different transformed points can both be closest to
    the same original point, silently producing a "permutation" that
    isn't actually one - a real, confirmed failure mode once anchors
    are packed closely enough (e.g. 8 aromatic positions per linker on
    a biphenyl-type core, vs. the usual 4 on a plain benzene ring).
    ``scipy.optimize.linear_sum_assignment`` finds the assignment
    minimising total distance subject to being one-to-one, which is
    always a valid bijection; ``max_distance`` then catches the
    genuinely-different failure mode where even that best assignment
    doesn't actually line up (the input doesn't have the symmetry this
    transformation assumes).

    E.g:
        original:    [0, 1, 2, 3]
        transformed: [2, 0, 3, 1]

    Parameters
    ----------
    original_fg_anchors : ase.Atoms
        The original set of functional group anchor atoms (At atoms),
        before any symmetry transformation is applied.
    transformed_fg_anchors : ase.Atoms
        The transformed set of functional group anchor atoms (At atoms),
        after applying a rotation or reflection symmetry operation.
        Must have the same number of atoms as original_fg_anchors.
    max_distance : float, default 1.0
        Maximum acceptable distance (Angstrom) between a transformed
        anchor and its matched original anchor. Loose enough for minor
        numerical noise, tight enough to catch a genuine symmetry
        mismatch rather than silently returning a wrong permutation.

    Returns
    -------
    permutation : list of int
        A list of length len(original_fg_anchors) where permutation[i]
        is the index in original_fg_anchors that corresponds to position
        i in transformed_fg_anchors.
        e.g. permutation[0] = 2 means the first transformed anchor
        maps to the third original anchor.

    Raises
    ------
    RuntimeError
        If the best possible one-to-one match still leaves some pair
        further apart than ``max_distance`` - the transformed anchors
        don't actually coincide with the original set, so this isn't a
        valid symmetry operation for this structure.
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
            f"project_permutation: even the best one-to-one match leaves a "
            f"pair {worst:.3f} Å apart (> max_distance={max_distance}) - "
            "this transformation doesn't actually map the FG anchor set "
            "onto itself, so it isn't a valid symmetry operation for this "
            "structure's actual geometry."
        )

    permutation = np.empty(len(trans_positions), dtype=int)
    permutation[trans_idx] = orig_idx
    return permutation.tolist()


