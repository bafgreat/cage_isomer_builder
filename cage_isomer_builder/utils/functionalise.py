#!/usr/bin/env python3
"""
Attaching fragments at R sites and docking guests into a host.
"""
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

import math
import random
import warnings
from collections import namedtuple
from itertools import combinations

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from cage_isomer_builder.utils.cavity import vdw_radius

from cage_isomer_builder.utils.geometry import (
    align_fragment_orientation,
    bonded_pairs,
    random_rotation_matrix,
    resolve_fragment_twist,
    translate_fragment,
)

FunctionalisedIsomer = namedtuple(
    'FunctionalisedIsomer', ['atoms', 'bond_matrix', 'free_atom_indices', 'anchor_pairs']
)

HostGuestComplex = namedtuple(
    'HostGuestComplex', ['atoms', 'bond_matrix', 'guest_atom_indices', 'guest_labels']
)


def _atoms_overlap(pos_a, radii_a, pos_b, num_b, tolerance):
    """
    True if any atom of set A and any atom of set B are closer than
    ``tolerance * (r_a + r_b)`` (van der Waals radii). A's radii are passed in
    precomputed, because callers test many poses against the same set A.
    """
    radii_b = vdw_radius(num_b)
    for pb, rb in zip(pos_b, radii_b):
        dists = np.linalg.norm(pos_a - pb, axis=1)
        if np.any(dists < tolerance * (radii_a + rb)):
            return True
    return False


def _principal_extents(positions, radii):
    """
    Extent (max - min) of a set of atomic spheres along each of its principal
    axes (SVD of the positions). The smallest of the three is the narrowest
    cross-section the set has in any orientation, as needed for the window
    check. Each projected coordinate is padded by the atom's radius, so a flat
    molecule such as benzene still has the thickness of its atoms.
    """
    centred = positions - positions.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ vt.T
    radii = np.asarray(radii)[:, None]
    return (projected + radii).max(axis=0) - (projected - radii).min(axis=0)


def _host_pore_geometry(host, inner_radius_fraction):
    """
    Centre and radius of the docking sphere, and the host's largest window.

    The cavity is the largest sphere that fits inside the host (clearance to
    the nearest atom surface, van der Waals radii) and the window is the widest
    straight opening out of it (see :mod:`cage_isomer_builder.utils.cavity`).
    If the host's centre is inside an atom (no enclosed cavity, e.g. an open
    fragment), the mean atom distance from the centre of mass is used instead,
    with a warning.

    Returns
    -------
    cavity_centre : np.ndarray, shape (3,)
    cavity_radius : float
        ``inner_radius_fraction`` times the free cavity radius.
    window_diameters : np.ndarray or None
        Free diameter of the largest window, or None for a closed cage or when
        no cavity was found (the window check is then skipped).
    """
    from cage_isomer_builder.utils.cavity import largest_window, pore_sphere

    centre, diameter = pore_sphere(host)
    if diameter > 0:
        window = largest_window(host, centre)
        windows = None if window is None else np.array([window])
        return centre, inner_radius_fraction * diameter / 2, windows
    warnings.warn(
        "no enclosed cavity found at the host's centre (it lies inside an atom); "
        "using the mean atom distance from the centre of mass as the docking "
        "radius instead."
    )
    cage_com = host.get_center_of_mass()
    inner_radius = inner_radius_fraction * np.mean(
        np.linalg.norm(host.positions - cage_com, axis=1)
    )
    return cage_com, inner_radius, None


def _check_guest_fits_windows(guests, window_diameters):
    """
    Raise if a guest's narrowest cross-section is wider than the host's largest
    window, i.e. it cannot enter in any orientation. Passing the check means
    that entry is possible, not that it is guaranteed (a non-circular window can
    still block some orientations). A guest that cannot enter at all usually
    means the wrong host or guest, so this raises instead of warning.
    """
    if window_diameters is None or len(window_diameters) == 0:
        return
    max_window = float(np.max(window_diameters))
    for i, guest in enumerate(guests):
        radii = vdw_radius(guest.numbers)
        min_extent = float(_principal_extents(guest.positions, radii).min())
        if min_extent > max_window:
            raise ValueError(
                f"guests[{i}]'s narrowest cross-section ({min_extent:.2f} A) "
                f"is larger than the host's largest window ({max_window:.2f} A), "
                "so it cannot enter the cavity in any orientation."
            )


def _next_type_index(counts, ratios):
    """
    The guest type furthest below its target share so far (a weighted round
    robin), so guest types can be mixed by ratio without knowing the final
    count. Always 0 for one guest type.
    """
    weights = np.asarray(ratios, dtype=float)
    weights = weights / weights.sum()
    counts_arr = np.asarray(counts, dtype=float)
    proportion_so_far = counts_arr / (counts_arr.sum() + 1.0)
    return int(np.argmax(weights - proportion_so_far))


def _fibonacci_sphere_directions(n):
    """
    ``n`` points spread evenly over the unit sphere (Fibonacci spiral): a
    deterministic cover of all directions without the gaps of random draws.
    """
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * i
    return np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ], axis=1)


def _axis_angle_matrix(axis, angle):
    """Rotation matrix for a right-handed rotation of angle around axis (Rodrigues' formula)."""
    axis = axis / np.linalg.norm(axis)
    k0, k1, k2 = axis
    K = np.array([[0, -k2, k1], [k2, 0, -k0], [-k1, k0, 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _systematic_rotation_grid(n_axes=8, n_twists=3):
    """
    ``n_axes * n_twists`` rotation matrices (Fibonacci-sphere axes, evenly
    spaced twists about each), which cover orientations at one position more
    evenly than as many random rotations (see _attempt_random_placement).
    """
    axes = _fibonacci_sphere_directions(n_axes)
    twists = np.linspace(0, 2 * np.pi, n_twists, endpoint=False)
    return [_axis_angle_matrix(axis, twist) for axis in axes for twist in twists]


_ROTATION_GRID = _systematic_rotation_grid()


def _attempt_random_placement(guest_template,
                              cage_com,
                              inner_radius,
                              existing_positions,
                              existing_numbers,
                              overlap_tolerance,
                              max_attempts,
                              ):
    """
    Search for one non-overlapping copy of ``guest_template`` in the docking
    sphere: up to ``max_attempts // len(_ROTATION_GRID)`` random positions,
    each tried in every orientation of the rotation grid (randomly turned as a
    whole, so results depend on the seed).

    A position is skipped without trying orientations when some guest atom
    clashes with some existing atom in every orientation, i.e. when the
    existing atom is closer than ``tol * (r_e + r_j)`` minus that guest atom's
    distance from the guest centre.

    Returns
    -------
    ase.Atoms or None
        The placed guest copy, or None if no attempt succeeded, which tells
        _build_guest_packing that the packing has jammed.
    """
    guest_centred = guest_template.copy()
    guest_centred.translate(-guest_centred.get_center_of_mass())
    guest_radii = vdw_radius(guest_centred.numbers)
    guest_offsets = np.linalg.norm(guest_centred.positions, axis=1)
    existing_radii = vdw_radius(existing_numbers)

    n_position_tries = max(1, max_attempts // len(_ROTATION_GRID))
    for _ in range(n_position_tries):
        point = np.random.uniform(-inner_radius, inner_radius, 3)
        if np.linalg.norm(point) > inner_radius:
            continue
        target = cage_com + point

        # Hopeless if some guest atom j clashes with some existing atom e
        # in every orientation: even turned straight away from e it is at
        # most d_e + a_j from it (a_j = its distance from the guest's
        # centre), so a clash is certain when d_e + a_j < tol (r_e + r_j).
        d = np.linalg.norm(existing_positions - target, axis=1)
        reach = overlap_tolerance * (existing_radii[:, None] + guest_radii[None, :]) \
            - guest_offsets[None, :]
        if np.any(d[:, None] < reach):
            continue  # no rotation can rescue this position

        base_rotation = random_rotation_matrix()
        for grid_rotation in _ROTATION_GRID:
            g = guest_centred.copy()
            g.positions = g.positions @ (base_rotation @ grid_rotation).T
            g.translate(target)

            if _atoms_overlap(existing_positions, existing_radii,
                              g.positions, g.numbers, overlap_tolerance):
                continue

            return g

    return None


def _build_guest_packing(host, guests, ratios, cage_com, inner_radius,
                         overlap_tolerance, max_attempts):
    """
    Random Sequential Adsorption (RSA): place guest copies one after another at
    random positions and orientations inside the docking sphere (the type of
    each copy chosen by _next_type_index), accepting a copy only if no atom
    overlaps the host or an earlier copy (_atoms_overlap). Stops when one
    placement fails after ``max_attempts`` tries.

    RSA is fast and typically jams at 35-55 % volume fraction for near-spherical
    guests; optimisation-based packers such as PACKMOL reach higher densities
    at a much higher cost. Every accepted copy was checked against all earlier
    ones, so any subset of the packing is itself a valid placement.

    Returns
    -------
    list of (ase.Atoms, int)
        Each placed copy and the index of its guest type, in placement order.
    """
    positions = host.positions.copy()
    numbers = host.numbers.copy()
    type_counts = [0] * len(guests)
    accepted = []

    while True:
        type_idx = _next_type_index(type_counts, ratios)
        placement = _attempt_random_placement(
            guests[type_idx], cage_com, inner_radius,
            positions, numbers, overlap_tolerance, max_attempts,
        )
        if placement is None:
            return accepted

        accepted.append((placement, type_idx))
        type_counts[type_idx] += 1
        positions = np.vstack([positions, placement.positions])
        numbers = np.concatenate([numbers, placement.numbers])


def _vdw_radii(numbers):
    """Alvarez van der Waals radii (Angstrom), 2.0 where the table has none."""
    from ase.data.vdw_alvarez import vdw_radii

    radii = vdw_radii[numbers]
    return np.where(np.isfinite(radii), radii, 2.0)


#: Penalty (eV/A^2) on a guest centre outside the docking sphere.
_OUTSIDE_PENALTY = 100.0


class _EnergySearch:
    """
    Finds low-energy guest positions inside a host with a pair-energy
    model from :func:`~cage_isomer_builder.utils.dispersion.pair_model`
    (two-body D4, or UFF Lennard-Jones when dftd4 is not installed).

    Atoms already in place (the host, then each accepted guest) are kept
    as positions plus their index in the model, so the parameters of any
    pair can be looked up.
    """

    def __init__(self, host, guests, cage_com, inner_radius, n_samples,
                 contact_scale, dispersion):
        from cage_isomer_builder.utils.dispersion import pair_model

        self.model = pair_model(host, guests, dispersion)
        self.guests = []
        for guest in guests:
            centred = guest.copy()
            centred.translate(-centred.get_center_of_mass())
            self.guests.append(centred)
        self.cage_com = np.asarray(cage_com, dtype=float)
        self.inner_radius = inner_radius
        self.n_samples = n_samples
        self.n_starts = 30
        self.contact_scale = contact_scale

    def _min_distance(self, existing_numbers, guest_numbers):
        return self.contact_scale * (
            _vdw_radii(existing_numbers)[:, None] + _vdw_radii(guest_numbers)[None, :]
        )

    def _score(self, existing, type_idx, centres, rotations):
        """
        For each (centre, rotation) pose: pair energy plus penalties (eV),
        used to rank and move poses, and how far (Angstrom) the pose breaks
        a limit: the worst atom-pair overlap, or how far the guest's centre
        is outside the docking sphere.
        """
        source, positions, numbers = existing
        guest = self.guests[type_idx]
        poses = np.einsum('nj,pij->pni', guest.positions, rotations) + centres[:, None, :]
        energy, penalty, overlap = self.model.score(
            source, positions,
            self.model.offsets[type_idx + 1] + np.arange(len(guest)),
            poses, self._min_distance(numbers, guest.numbers),
        )
        outside = np.maximum(
            0.0, np.linalg.norm(centres - self.cage_com, axis=1) - self.inner_radius
        )
        total = energy + penalty + _OUTSIDE_PENALTY * outside ** 2
        return total, np.maximum(overlap, outside)

    def _refine(self, existing, type_idx, centre, rotation):
        """
        Local rigid-body minimisation (L-BFGS) over the guest's translation
        and a rotation vector. The gradient is taken by central
        differences, all 12 displaced poses scored in one kernel call.
        """
        from scipy.optimize import minimize
        from scipy.spatial.transform import Rotation

        h = 1e-4
        displaced = np.vstack([np.zeros(6), h * np.eye(6), -h * np.eye(6)])

        def poses(x):
            xs = x + displaced
            rot = Rotation.from_rotvec(xs[:, 3:]).as_matrix() @ rotation
            return centre + xs[:, :3], rot

        def objective(x):
            totals, _ = self._score(existing, type_idx, *poses(x))
            return totals[0], (totals[1:7] - totals[7:]) / (2 * h)

        result = minimize(objective, np.zeros(6), jac=True, method="L-BFGS-B")
        centres, rotations = poses(result.x)
        totals, limits = self._score(existing, type_idx, centres[:1], rotations[:1])
        return centres[0], rotations[0], totals[0], limits[0]

    def best_poses(self, existing, type_idx, n_keep, max_overlap=0.01):
        """
        Up to ``n_keep`` distinct low-energy poses (centre, rotation, score),
        lowest first. A pose is kept only if, after refinement, no pair is
        more than ``max_overlap`` Angstrom inside its minimum distance and
        its centre is no more than that outside the docking sphere. Poses
        whose centres are less than 1 Angstrom apart count as the same site.
        """
        points = np.random.uniform(-1.0, 1.0, (4 * self.n_samples, 3))
        points = points[np.linalg.norm(points, axis=1) <= 1.0][: self.n_samples]
        centres = self.cage_com + self.inner_radius * points
        rotations = np.array([
            random_rotation_matrix() @ grid for _ in centres for grid in _ROTATION_GRID
        ])
        centres = np.repeat(centres, len(_ROTATION_GRID), axis=0)
        totals, _ = self._score(existing, type_idx, centres, rotations)

        # Refine the best pose of many different sites, not many poses of one.
        starts = []
        for i in np.argsort(totals):
            if len(starts) == self.n_starts + n_keep:
                break
            if all(np.linalg.norm(centres[i] - centres[j]) >= 1.0 for j in starts):
                starts.append(i)
        refined = [self._refine(existing, type_idx, centres[i], rotations[i]) for i in starts]
        refined = sorted(
            (pose for pose in refined if pose[3] <= max_overlap), key=lambda pose: pose[2]
        )
        distinct = []
        for centre, rotation, total, _ in refined:
            if all(np.linalg.norm(centre - kept[0]) >= 1.0 for kept in distinct):
                distinct.append((centre, rotation, total))
        return distinct[:n_keep]

    def add(self, existing, type_idx, centre, rotation):
        """``existing`` plus the guest placed at this pose, and that guest as Atoms."""
        source, positions, numbers = existing
        guest = self.guests[type_idx].copy()
        guest.positions = guest.positions @ rotation.T + centre
        offset = self.model.offsets[type_idx + 1]
        return (
            (np.concatenate([source, offset + np.arange(len(guest))]),
             np.vstack([positions, guest.positions]),
             np.concatenate([numbers, guest.numbers])),
            guest,
        )


def _place_guests_by_energy(host, guests, ratios, n_guests, n_complexes,
                            cage_com, inner_radius, n_samples, contact_scale,
                            dispersion):
    """
    Build up to ``n_complexes`` low-energy host-guest complexes.

    Complex ``c`` puts its first guest at the ``c``-th best distinct site,
    then adds each further guest at the lowest-energy site left (the
    earlier guests count too). Every complex is then rescored with the
    model's full interaction energy. Returns the model's name and a list of
    (placed, energy), where placed is a list of (guest Atoms, type index),
    lowest energy first.
    """
    search = _EnergySearch(host, guests, cage_com, inner_radius, n_samples,
                           contact_scale, dispersion)
    host_state = (np.arange(len(host)), host.positions.copy(), host.numbers.copy())

    first_type = _next_type_index([0] * len(guests), ratios)
    first_sites = search.best_poses(host_state, first_type, n_complexes)
    if len(first_sites) < n_complexes:
        warnings.warn(
            f"Requested n_complexes={n_complexes} but only {len(first_sites)} "
            "distinct sites were found for the first guest; capping to "
            f"{len(first_sites)}. Raise n_samples or lower contact_scale for a "
            "wider search."
        )

    results = []
    short_of_guests = None
    for centre, rotation, _ in first_sites:
        state, guest = search.add(host_state, first_type, centre, rotation)
        placed = [(guest, first_type)]
        counts = [0] * len(guests)
        counts[first_type] += 1
        while len(placed) < n_guests:
            type_idx = _next_type_index(counts, ratios)
            poses = search.best_poses(state, type_idx, 1)
            if not poses:
                short_of_guests = len(placed)
                break
            state, guest = search.add(state, type_idx, poses[0][0], poses[0][1])
            placed.append((guest, type_idx))
            counts[type_idx] += 1
        energy = search.model.interaction_energy([host] + [g for g, _ in placed])
        results.append((placed, energy))

    if short_of_guests is not None:
        warnings.warn(
            f"Requested n_guests={n_guests} but no allowed site was left after "
            f"{short_of_guests} guest(s) in at least one complex. Lower "
            "contact_scale or raise n_samples to search harder."
        )
    return search.model.name, sorted(results, key=lambda item: item[1])


def _assemble_complex(host, host_bond_matrix, placed):
    """HostGuestComplex from the host and a list of (guest Atoms, type index)."""
    n_host = len(host)
    combined = host.copy()
    for g, _ in placed:
        combined = combined + g

    n_total = len(combined)
    bond_matrix = np.zeros((n_total, n_total))
    bond_matrix[:n_host, :n_host] = host_bond_matrix

    offset = n_host
    guest_labels = []
    for g, type_idx in placed:
        for i, j in bonded_pairs(g):
            bond_matrix[offset + i, offset + j] = bond_matrix[offset + j, offset + i] = 1.0
        offset += len(g)
        guest_labels.append(type_idx)

    guest_atom_indices = list(range(n_host, n_total))
    return HostGuestComplex(combined, bond_matrix, guest_atom_indices, guest_labels)


def _reject_site_markers(host, guests):
    """Docking needs real atoms: an R-site marker (X, atomic number 0) has
    no size or dispersion parameters."""
    for name, atoms in [("host", host)] + [(f"guests[{i}]", g) for i, g in enumerate(guests)]:
        n = int(np.sum(atoms.numbers == 0))
        if n:
            raise ValueError(
                f"{name} contains {n} dummy X atom(s) (R-site markers). Dock into a "
                "structure with real atoms: attach groups with functionalise_isomer_sites, "
                "or use CageBuilder.to_ase(hydrogens=True) for the parent cage."
            )


def _normalise_guests(guests, ratios):
    if isinstance(guests, Atoms):
        guests = [guests]
    if ratios is None:
        ratios = [1.0] * len(guests)
    elif len(ratios) != len(guests):
        raise ValueError(
            f"ratios has {len(ratios)} entries but guests has {len(guests)}."
        )
    return guests, ratios


def max_guests_in_host(host,
                       guests,
                       ratios=None,
                       max_attempts=1000,
                       overlap_tolerance=0.75,
                       inner_radius_fraction=0.9,
                       seed=None,
                       ):
    """
    Number of guest copies that fit in the host's cavity at once, from a Random
    Sequential Adsorption packing (see _build_guest_packing) with the same
    overlap test as :func:`place_guest_in_host`. The result depends on ``seed``
    and ``max_attempts``; it is not a guaranteed maximum.

    Before any placement, each guest is checked against the host's largest
    window (:func:`cage_isomer_builder.utils.cavity.largest_window`); a guest
    wider than every opening raises at once.

    Parameters
    ----------
    host : ase.Atoms
    guests : ase.Atoms or sequence of ase.Atoms
        One guest type, or several to mix.
    ratios : sequence of float, optional
        Relative share of each guest type (several guest types only). Default:
        equal shares.
    max_attempts : int, default 1000
        Consecutive failed placements after which the packing counts as jammed.
    overlap_tolerance : float, default 0.75
        Closest allowed approach of two atoms (host-guest or guest-guest) as a
        fraction of the sum of their van der Waals radii: 2.55 A for C...C and
        1.8 A for H...H at 0.75. 1.0 means van der Waals spheres may not overlap.
    inner_radius_fraction : float, default 0.9
        Fraction of the cavity radius in which guest centres are placed, a small
        margin from the wall (contacts are already limited by
        ``overlap_tolerance``). The cavity is the largest sphere inside the host
        (:func:`cage_isomer_builder.utils.cavity.pore_sphere`).
    seed : int, optional
        Seed for a reproducible result.

    Returns
    -------
    int
        The number of guest copies placed.

    Raises
    ------
    ValueError
        If a guest's narrowest cross-section is wider than the host's largest
        window (not raised for a closed host), or the host or a guest contains
        R-site markers.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
    >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
    ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
    >>> _ = cage.build()
    >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
    >>> host, methane = cage.to_ase(), example_data.read("methane")
    >>> from cage_isomer_builder.utils.functionalise import max_guests_in_host
    >>> max_guests_in_host(host, methane, seed=0)            # in its 7.5 A cavity
    4
    >>> max_guests_in_host(host, example_data.read("ibuprofen"))   # wider than every window
    Traceback (most recent call last):
    ...
    ValueError: guests[0]'s narrowest cross-section (5.83 A) is larger than the host's largest window (4.10 A), so it cannot enter the cavity in any orientation.
    """
    guests, ratios = _normalise_guests(guests, ratios)
    _reject_site_markers(host, guests)
    if seed is not None:
        np.random.seed(seed)

    cage_com, inner_radius, window_diameters = _host_pore_geometry(host, inner_radius_fraction)
    _check_guest_fits_windows(guests, window_diameters)
    packing = _build_guest_packing(
        host, guests, ratios, cage_com, inner_radius, overlap_tolerance, max_attempts
    )
    return len(packing)


def place_guest_in_host(host,
                        guests,
                        n_complexes=1,
                        n_guests=1,
                        ratios=None,
                        max_attempts=1000,
                        overlap_tolerance=0.75,
                        inner_radius_fraction=0.9,
                        host_bond_matrix=None,
                        seed=None,
                        method="rsa",
                        n_samples=1000,
                        contact_scale=0.75,
                        dispersion="auto",
                        ):
    """
    Build ``n_complexes`` host-guest complexes.

    Two placement methods:

    * ``method="rsa"`` (default): random placement wherever the guests fit.
    * ``method="energy"``: each guest at its lowest-energy site, scored with
      dispersion (see "Energy method").

    Strategy (RSA)
    --------------
    1. The docking sphere is the largest sphere inside the host
       (:func:`cage_isomer_builder.utils.cavity.pore_sphere`), or a
       mean-atom-distance sphere when the host has no enclosed cavity.
    2. A guest wider than the host's largest window
       (:func:`cage_isomer_builder.utils.cavity.largest_window`) is refused
       before any placement.
    3. One Random Sequential Adsorption packing of the cavity is built (see
       _build_guest_packing); any subset of it is a valid placement.
    4. Each complex takes a different random subset of ``n_guests`` copies.
       ``n_guests`` is capped to the packing size, and ``n_complexes`` to the
       number of distinct subsets, each with a warning.

    Parameters
    ----------
    host : ase.Atoms
        The cage.
    guests : ase.Atoms or sequence of ase.Atoms
        One guest type, or several to mix (the type of each copy is chosen by a
        ratio-weighted round robin).
    n_complexes : int, default 1
        Number of complexes.
    n_guests : int, default 1
        Guest copies per complex, capped to what fits.
    ratios : sequence of float, optional
        Relative share of each guest type. Default: equal shares.
    max_attempts : int, default 1000
        Consecutive failed placements after which the RSA packing counts as
        jammed.
    overlap_tolerance : float, default 0.75
        Closest allowed approach of two atoms as a fraction of the sum of their
        van der Waals radii (RSA).
    inner_radius_fraction : float, default 0.9
        Fraction of the cavity radius in which guest centres are placed.
        Smaller values shrink the usable cavity.
    host_bond_matrix : np.ndarray, shape (len(host), len(host)), optional
        The host's bond orders, e.g. from ``CageBuilder.get_bond_matrix()``.
        Default: perceived from the geometry by
        ``gulp_setup.mmanalysis.analyze_mm``. Each complex carries the host's
        bonds plus each guest copy's own bonds, perceived on the isolated guest.
        No bond is added between host and guest or between guests.
    seed : int, optional
        Seed for the packing, the subsets and the energy search.
    method : {"rsa", "energy"}, default "rsa"
    n_samples : int, default 1000
        Energy method: random positions tried per guest, each in 24
        orientations.
    contact_scale : float, default 0.75
        Energy method: closest allowed approach of two atoms as a fraction of
        the sum of their van der Waals radii. Both scoring models have their own
        repulsion, which normally keeps atoms further apart than this.
    dispersion : str or dict, default "auto"
        Energy method scoring: ``"auto"`` (DFT-D4 with PBE parameters if dftd4
        is installed, otherwise UFF Lennard-Jones with a warning; D4 is combined
        with the repulsive part of UFF's Lennard-Jones term), ``"uff"``, a D4
        method name from ``dispersion.D4_PARAMETERS`` (e.g. ``"pbe"``), or a
        dict with s6, s8, a1, a2 and optionally s9.

    Energy method
    -------------
    1. The C6 coefficients of the host and each guest type are computed once
       with dftd4 (UFF: per-element parameters).
    2. For each guest, ``n_samples`` random positions in the docking sphere are
       scored in 24 orientations each; the best pose of each of 30 sites is
       minimised as a rigid body (L-BFGS).
    3. Complex ``c`` puts its first guest at the ``c``-th best site; each
       further guest goes to the lowest-energy site left, so guest-guest
       dispersion counts too.
    4. Each complex is rescored (D4: a full dftd4 calculation with the
       three-body term, plus the UFF repulsion). The energy (eV) is stored in
       ``complex.atoms.info["interaction_energy"]`` and the model in
       ``complex.atoms.info["dispersion_model"]``. Complexes are returned lowest
       energy first; with fewer sites than ``n_complexes``, the result is capped
       with a warning.

    Returns
    -------
    list of HostGuestComplex
        Named tuples ``(atoms, bond_matrix, guest_atom_indices, guest_labels)``;
        ``guest_labels`` gives the guest type of each placed copy.

    Raises
    ------
    ValueError
        If a guest is wider than the host's largest window, the host or a guest
        contains R-site markers, or ``method`` is unknown.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
    >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
    ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
    >>> _ = cage.build()
    >>> _ = cage.optimise(include_charges=False, logfile=None)   # the UiO-66 tetrahedral cage
    >>> host, methane = cage.to_ase(), example_data.read("methane")
    >>> from cage_isomer_builder.utils.functionalise import place_guest_in_host
    >>> complexes = place_guest_in_host(host, methane, n_guests=2, n_complexes=2, seed=0,
    ...                                 host_bond_matrix=cage.get_bond_matrix())
    >>> len(complexes), len(complexes[0].guest_labels), len(complexes[0].atoms) - len(host)
    (2, 2, 10)
    >>> best = place_guest_in_host(host, methane, method="energy", n_samples=200, seed=0)[0]
    >>> best.atoms.info["dispersion_model"], bool(best.atoms.info["interaction_energy"] < 0)
    ('D4(pbe)+UFF repulsion', True)
    """
    guests, ratios = _normalise_guests(guests, ratios)
    _reject_site_markers(host, guests)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if host_bond_matrix is None:
        from gulp_setup.mmanalysis import analyze_mm
        host_bond_matrix, _ = analyze_mm(host)

    cage_com, inner_radius, window_diameters = _host_pore_geometry(host, inner_radius_fraction)
    _check_guest_fits_windows(guests, window_diameters)

    if method == "energy":
        model_name, results = _place_guests_by_energy(
            host, guests, ratios, n_guests, n_complexes, cage_com, inner_radius,
            n_samples, contact_scale, dispersion,
        )
        complexes = []
        for placed, energy in results:
            result = _assemble_complex(host, host_bond_matrix, placed)
            result.atoms.info["interaction_energy"] = energy
            result.atoms.info["dispersion_model"] = model_name
            complexes.append(result)
        return complexes
    if method != "rsa":
        raise ValueError(f"method={method!r}; use 'rsa' or 'energy'.")

    packing = _build_guest_packing(
        host, guests, ratios, cage_com, inner_radius, overlap_tolerance, max_attempts
    )

    if n_guests > len(packing):
        warnings.warn(
            f"Requested n_guests={n_guests} but RSA only packed "
            f"{len(packing)} guest copies into the cavity before jamming "
            f"(overlap_tolerance={overlap_tolerance}, max_attempts="
            f"{max_attempts}); capping to {len(packing)}. Call "
            "max_guests_in_host() first to know the capacity, or raise "
            "max_attempts or lower overlap_tolerance for a denser search."
        )
    n_guests_actual = min(n_guests, len(packing))

    total_combos = math.comb(len(packing), n_guests_actual)
    if n_complexes > total_combos:
        warnings.warn(
            f"Requested n_complexes={n_complexes} but only {total_combos} "
            f"distinct {n_guests_actual}-guest combinations exist in the "
            f"{len(packing)}-guest packing; capping to {total_combos}."
        )
    n_complexes_actual = min(n_complexes, total_combos)

    # Enumerate and shuffle when there are few combinations or many are
    # requested; otherwise rejection-sample, where repeats are then rare.
    if total_combos <= 5000 or n_complexes_actual > total_combos * 0.1:
        all_combos = list(combinations(range(len(packing)), n_guests_actual))
        random.shuffle(all_combos)
        chosen_combos = all_combos[:n_complexes_actual]
    else:
        seen = set()
        chosen_combos = []
        while len(chosen_combos) < n_complexes_actual:
            combo = tuple(sorted(
                np.random.choice(len(packing), size=n_guests_actual, replace=False)
            ))
            if combo in seen:
                continue
            seen.add(combo)
            chosen_combos.append(combo)

    return [
        _assemble_complex(host, host_bond_matrix, [packing[idx] for idx in combo])
        for combo in chosen_combos
    ]


def functionalise_host(host_system,
                       sub_fragment,
                       bond_length=1.5,
                       host_marker='X',
                       fragment_marker='X',
                       guests=None,
                       n_complexes=1,
                       n_guests=1,
                       ratios=None,
                       max_attempts=1000,
                       overlap_tolerance=0.75,
                       inner_radius_fraction=0.9,
                       seed=None,
                       ):
    """
    Attach a fragment at the single site of a host and, optionally, dock guests
    in the result (see :func:`place_guest_in_host`).

    The fragment is aligned to the host's site, moved so that its bonding atom
    sits on the site, and merged into the host; both marker atoms are removed.

    Parameters
    ----------
    host_system : ase.Atoms
        Host with one site atom.
    sub_fragment : ase.Atoms
        Fragment with one marker atom at its bonding end.
    bond_length : float, default 1.5
        Length (Angstrom) of the new host-fragment bond.
    host_marker : str, default 'X'
        Symbol of the site atom in the host (``cage.functionalise()`` writes
        ``X`` sites).
    fragment_marker : str, default 'X'
        Symbol of the marker atom in the fragment.
    guests : ase.Atoms or sequence of ase.Atoms, optional
        Guests to dock.
    n_complexes, n_guests, ratios, max_attempts, overlap_tolerance, inner_radius_fraction, seed
        As for :func:`place_guest_in_host`.

    Returns
    -------
    ase.Atoms, HostGuestComplex or list of HostGuestComplex
        The functionalised host if ``guests`` is None; otherwise one complex
        (``n_complexes == 1``) or a list of complexes.

    Notes
    -----
    With guests, the bond matrix of each complex is perceived from the
    functionalised host's geometry. To use exact bonds, attach the fragment and
    call :func:`place_guest_in_host` with ``host_bond_matrix`` yourself.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.utils.functionalise import functionalise_host
    >>> host = molecule("C6H6")
    >>> host[6].symbol = "X"                            # one site, on C 0
    >>> aniline = functionalise_host(host, example_data.read("NH2"))
    >>> aniline.get_chemical_formula()
    'C6H7N'
    """
    aligned_sub_fragment, sub_x, host_x, sub_neigh = align_fragment_orientation(
        host_system, sub_fragment, bond_length,
        host_marker=host_marker,
        fragment_marker=fragment_marker,
    )
    translated_fragment = translate_fragment(
        host_system, aligned_sub_fragment, sub_neigh, host_x
    )
    translated_fragment = resolve_fragment_twist(
        host_system, translated_fragment, host_x, sub_neigh
    )

    sub_frag = [i.index for i in translated_fragment if i.index != sub_x]
    new_fragment = translated_fragment[sub_frag]

    host_frag = [i.index for i in host_system if i.index != host_x]
    host_system = host_system[host_frag]

    functionalised_host = host_system + new_fragment

    if guests is None:
        return functionalised_host

    complexes = place_guest_in_host(
        functionalised_host,
        guests,
        n_complexes=n_complexes,
        n_guests=n_guests,
        ratios=ratios,
        max_attempts=max_attempts,
        overlap_tolerance=overlap_tolerance,
        inner_radius_fraction=inner_radius_fraction,
        seed=seed,
    )
    return complexes[0] if n_complexes == 1 else complexes


def _allocate_counts(ratios, n_sites):
    """
    Integer counts from relative ratios that add up to ``n_sites``
    (largest-remainder method; plain rounding may not add up).
    """
    weights = np.asarray(ratios, dtype=float)
    weights = weights / weights.sum()
    raw = weights * n_sites
    counts = np.floor(raw).astype(int)
    remainder = n_sites - counts.sum()
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        for i in order[:remainder]:
            counts[i] += 1
    return counts.tolist()


def _fragments_by_rgroup(host_system, fragments_by_name):
    """
    Order a ``{group name: fragment}`` dict by R-group number, using the
    names stored in ``host_system.info["rgroup_labels"]`` (e.g. "A B" means
    A = R1, B = R2). Keys may be those names or "R1", "R2", ...
    """
    names = str(host_system.info.get('rgroup_labels', '')).split()
    n_groups = max(len(names), int(np.max(host_system.arrays.get('rgroup', [0]))))
    ordered = []
    for number in range(1, n_groups + 1):
        keys = [f"R{number}"] + ([names[number - 1]] if number <= len(names) else [])
        found = [k for k in keys if k in fragments_by_name]
        if not found:
            raise ValueError(
                f"no fragment given for R{number}"
                + (f" ({names[number - 1]!r})" if number <= len(names) else "")
                + f"; got keys {sorted(fragments_by_name)}."
            )
        ordered.append(fragments_by_name[found[0]])
    unknown = set(fragments_by_name) - set(names) - {f"R{i}" for i in range(1, n_groups + 1)}
    if unknown:
        raise ValueError(
            f"fragment keys {sorted(unknown)} match no R-group in this isomer "
            f"(groups: {' '.join(names) or 'none recorded'})."
        )
    return ordered


def functionalise_isomer_sites(host_system,
                               fragments,
                               ratios=None,
                               site_marker='X',
                               fragment_marker='X',
                               bond_length=1.5,
                               seed=None,
                               host_bond_matrix=None,
                               ):
    """
    Attach real functional-group fragments at every site of an isomer,
    optionally mixing several fragment types by ratio.

    Sites are R sites (``X`` atoms with a non-zero ``"rgroup"`` value, see
    :mod:`cage_isomer_builder.utils.sites`) or, for a structure without that
    array, atoms of symbol ``site_marker``. Each fragment carries one marker
    atom ``fragment_marker`` at its bonding end (e.g. ``Atoms('XNHH', ...)`` for
    NH2). This is the multi-site version of :func:`functionalise_host`.

    Parameters
    ----------
    host_system : ase.Atoms
        The isomer.
    fragments : ase.Atoms, sequence of ase.Atoms, or dict
        One fragment for every site, or several to mix. For an R-group isomer
        (written by ``CageBuilder.enumerate_rgroup_isomers``) pass a dict keyed
        by group name or by ``"R1"``, ``"R2"``, ...; each site then gets its
        group's fragment.
    ratios : sequence of float, optional
        Relative number of sites for each fragment (several fragments only). The
        counts add up to the number of sites (largest-remainder method); which
        site gets which fragment is random (see ``seed``). Default: equal
        shares. Ignored for an R-group isomer when not given: each site then
        gets ``fragments[label - 1]``.
    site_marker : str, default 'X'
        Symbol of a site atom in ``host_system``.
    fragment_marker : str, default 'X'
        Symbol of the marker atom in each fragment.
    bond_length : float, default 1.5
        Length (Angstrom) of each new host-fragment bond.
    seed : int, optional
        Seed for the random assignment of fragments to sites.
    host_bond_matrix : np.ndarray, shape (len(host_system), len(host_system)), optional
        The host's bond orders. When given, the bonds of the decorated structure
        are tracked too (each fragment's bonds and the new bond at each site),
        as needed for a force-field run on the new groups only.

    Returns
    -------
    ase.Atoms or FunctionalisedIsomer
        The decorated structure, or, with ``host_bond_matrix``, a named tuple
        ``(atoms, bond_matrix, free_atom_indices, anchor_pairs)``:
        ``free_atom_indices`` are the new fragment atoms and ``anchor_pairs``
        the ``(host atom, fragment atom)`` of each new bond.

    Raises
    ------
    ValueError
        If ``host_system`` has no sites, or ``ratios`` does not match
        ``fragments``.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.utils.sites import mark_site
    >>> from cage_isomer_builder.utils.functionalise import functionalise_isomer_sites
    >>> isomer = molecule("C6H6")
    >>> mark_site(isomer, 6, 1)                         # R1 on C 0
    >>> mark_site(isomer, 9, 1)                         # R1 on C 3
    >>> isomer.info["rgroup_labels"] = "NH2"
    >>> decorated = functionalise_isomer_sites(isomer, {"NH2": example_data.read("NH2")})
    >>> decorated.get_chemical_formula()                # p-phenylenediamine
    'C6H8N2'
    """
    if isinstance(fragments, dict):
        fragments = _fragments_by_rgroup(host_system, fragments)
    if isinstance(fragments, Atoms):
        fragments = [fragments]

    from cage_isomer_builder.utils.sites import SITE_ARRAY, site_indices as _site_indices
    if site_marker == 'X' and SITE_ARRAY in host_system.arrays:
        site_indices = _site_indices(host_system)
    else:
        site_indices = [atom.index for atom in host_system if atom.symbol == site_marker]
    if not site_indices:
        raise ValueError(
            f"host_system has no {site_marker!r} sites to functionalise."
        )

    site_labels = host_system.arrays.get('rgroup')
    if site_labels is not None and ratios is None:
        # R-group isomer: the label at each site fixes its fragment exactly.
        assignment = [int(site_labels[i]) - 1 for i in site_indices]
        if any(a < 0 or a >= len(fragments) for a in assignment):
            raise ValueError(
                f"host_system's 'rgroup' labels need {max(assignment) + 1} "
                f"fragments (fragments[label - 1] per site) but {len(fragments)} "
                "were given, or a site has no label."
            )
    else:
        if ratios is None:
            ratios = [1.0] * len(fragments)
        elif len(ratios) != len(fragments):
            raise ValueError(
                f"ratios has {len(ratios)} entries but fragments has {len(fragments)}."
            )

        counts = _allocate_counts(ratios, len(site_indices))
        assignment = [i for i, count in enumerate(counts) for _ in range(count)]
        random.Random(seed).shuffle(assignment)

    # Process sites from the highest index down: each attachment removes the
    # site atom and appends the fragment, which leaves lower indices intact.
    site_and_fragment = sorted(zip(site_indices, assignment), reverse=True)

    track_bonds = host_bond_matrix is not None
    # provenance[k] is the original index of atom k of `functionalised`, or
    # a negative id for a new fragment atom. It follows every removal and
    # append, so all bonds can be mapped to final indices at the end.
    if track_bonds:
        provenance = list(range(len(host_system)))
        next_new_id = -1
        site_records = []

    functionalised = host_system.copy()
    remaining_sites = set(site_indices)
    for site_index, frag_idx in site_and_fragment:
        remaining_sites.discard(site_index)
        fragment_template = fragments[frag_idx]

        # The site's bonded neighbour (the ring carbon) as an index into the
        # current structure, where earlier removals have shifted indices.
        # With a bond matrix it is mapped through provenance; otherwise it is
        # the nearest atom that is neither H nor another site.
        if track_bonds:
            host_neigh_orig = int(np.nonzero(host_bond_matrix[site_index])[0][0])
            host_neigh_index = provenance.index(host_neigh_orig)
        else:
            others = [i for i in range(len(functionalised))
                      if i != site_index and i not in remaining_sites
                      and functionalised[i].symbol not in ('H', 'X')]
            gaps = np.linalg.norm(
                functionalised.positions[others] - functionalised.positions[site_index], axis=1)
            host_neigh_index = others[int(np.argmin(gaps))]

        aligned_sub_fragment, sub_x, host_x, sub_neigh = align_fragment_orientation(
            functionalised, fragment_template.copy(), bond_length,
            host_marker=site_marker,
            fragment_marker=fragment_marker,
            host_index=site_index,
            host_neigh_index=host_neigh_index,
        )
        translated_fragment = translate_fragment(
            functionalised, aligned_sub_fragment, sub_neigh, host_x
        )
        translated_fragment = resolve_fragment_twist(
            functionalised, translated_fragment, host_x, sub_neigh,
            host_neigh_index=host_neigh_index,
        )

        keep_local = [i.index for i in translated_fragment if i.index != sub_x]
        new_fragment = translated_fragment[keep_local]

        if track_bonds:
            n_new = len(new_fragment)
            new_ids = list(range(next_new_id, next_new_id - n_new, -1))
            next_new_id -= n_new
            # keep_local[k] is the template index of new_fragment's atom k;
            # it locates sub_neigh, the atom bonded to the site.
            anchor_new_id = new_ids[keep_local.index(sub_neigh)]

            local_to_new_id = dict(zip(keep_local, new_ids))
            fragment_bonds = [
                (local_to_new_id[i], local_to_new_id[j])
                for i, j in bonded_pairs(fragment_template, exclude=sub_x)
            ]
            site_records.append({
                'host_neigh_orig': host_neigh_orig,
                'anchor_new_id': anchor_new_id,
                'new_ids': new_ids,
                'fragment_bonds': fragment_bonds,
            })

            keep_host = [i.index for i in functionalised if i.index != host_x]
            provenance = [provenance[i] for i in keep_host] + new_ids
        else:
            keep_host = [i.index for i in functionalised if i.index != host_x]

        functionalised = functionalised[keep_host]
        functionalised = functionalised + new_fragment

    if not track_bonds:
        return functionalised

    final_index_of = {orig_id: pos for pos, orig_id in enumerate(provenance)}
    n_atoms = len(functionalised)
    bond_matrix = np.zeros((n_atoms, n_atoms))

    host_i, host_j = np.nonzero(host_bond_matrix)
    for i, j in zip(host_i.tolist(), host_j.tolist()):
        if i in final_index_of and j in final_index_of:
            bond_matrix[final_index_of[i], final_index_of[j]] = host_bond_matrix[i, j]

    anchor_pairs = []
    for record in site_records:
        fi = final_index_of[record['host_neigh_orig']]
        fj = final_index_of[record['anchor_new_id']]
        bond_matrix[fi, fj] = bond_matrix[fj, fi] = 1.0
        anchor_pairs.append((fi, fj))

        for a, b in record['fragment_bonds']:
            fa, fb = final_index_of[a], final_index_of[b]
            bond_matrix[fa, fb] = bond_matrix[fb, fa] = 1.0

    free_atom_indices = [
        final_index_of[nid] for record in site_records for nid in record['new_ids']
    ]

    return FunctionalisedIsomer(functionalised, bond_matrix, free_atom_indices, anchor_pairs)


def full_functionalisation(cage, linker_indices, aromatic_only=True):
    """
    Mark every aromatic C-H of the linkers as an R site.

    Each H bonded to an aromatic linker carbon becomes a site atom (an ``X``
    with ``"rgroup"`` = 1, i.e. R1, see :mod:`cage_isomer_builder.utils.sites`),
    moved out along the C-H direction to 1.47 Angstrom (about a C-N bond) so
    that a fragment attached there starts at a sensible bond length.

    Parameters
    ----------
    cage : ase.Atoms
        The structure. Changed in place.
    linker_indices : sequence of int
        Atoms of the linkers; only their C-H bonds become sites.
    aromatic_only : bool, default True
        Only H on aromatic-ring carbons (planar 5- or 6-membered rings, see
        :func:`cage_isomer_builder.utils.features.aromatic_rings`). False also
        marks H on alkyl carbons, where every H of a CH2 or CH3 becomes a
        separate site.

    Returns
    -------
    fg_anchors : ase.Atoms
        The site atoms, in slot order.
    fg_anchor_indices : list of int
        Index of each site atom in ``cage``.
    fg_anchor_h_positions : np.ndarray, shape (n_sites, 3)
        Each site's original H position, so an inactive site can be turned
        back into H.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.functionalise import full_functionalisation
    >>> benzene = molecule("C6H6")
    >>> sites, indices, h_positions = full_functionalisation(benzene, list(range(12)))
    >>> len(indices), benzene.get_chemical_symbols()[6:]     # every aromatic C-H is a site
    (6, ['X', 'X', 'X', 'X', 'X', 'X'])
    """
    from cage_isomer_builder.utils.sites import mark_site

    linker_set = set(linker_indices)
    if aromatic_only:
        from cage_isomer_builder.utils.features import aromatic_rings, bond_graph

        ring_carbons = {k for ring in aromatic_rings(cage, bond_graph(cage)) for k in ring
                        if cage[k].symbol == 'C'}
        linker_set &= ring_carbons
    carbon_indices = neighbor_list('i', cage, {('C', 'H'): 1.3})
    hydrogen_indices = neighbor_list('j', cage, {('C', 'H'): 1.3})

    fg_anchor_indices = []
    fg_anchor_h_positions = []
    for i_carbon, i_hydrogen in zip(carbon_indices, hydrogen_indices):
        if cage[i_carbon].symbol == 'C' and i_carbon in linker_set:
            fg_anchor_h_positions.append(cage[i_hydrogen].position.copy())
            # Extend the bond to a C-N length, moving the H end (fix=0).
            cage.set_distance(i_carbon, i_hydrogen, 1.47, fix=0, mic=bool(cage.pbc.any()))
            # Rounded for numerical consistency in symmetry operations.
            mark_site(cage, i_hydrogen, 1,
                      [round(p, 3) for p in cage[i_hydrogen].position])
            fg_anchor_indices.append(int(i_hydrogen))

    fg_anchors = cage[fg_anchor_indices]

    return fg_anchors, fg_anchor_indices, np.array(fg_anchor_h_positions)
