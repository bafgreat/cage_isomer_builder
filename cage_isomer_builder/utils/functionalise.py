#!/usr/bin/env python3
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

import random
import warnings
from collections import namedtuple

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii

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


def _atoms_overlap(pos_a, num_a, pos_b, num_b, tolerance=0.75):
    """
    Return True if any atom pair between set A and set B is closer than
    tolerance * (r_a + r_b), using covalent radii.
    """
    radii_a = covalent_radii[num_a]
    radii_b = covalent_radii[num_b]
    for pb, rb in zip(pos_b, radii_b):
        dists = np.linalg.norm(pos_a - pb, axis=1)
        if np.any(dists < tolerance * (radii_a + rb)):
            return True
    return False


def _pywindow_molecule(atoms):
    import pywindow
    system = pywindow.MolecularSystem.load_system({
        'elements': np.array(atoms.get_chemical_symbols()),
        'coordinates': atoms.get_positions(),
    })
    return system.system_to_molecule()


def _principal_extents(positions, radii):
    """
    The extent (max - min) of a set of finite-radius spheres along each
    of its own principal axes (via SVD/PCA on the bare positions, not
    the arbitrary lab-frame x/y/z axes it happens to be sitting in) -
    the minimum of the three is the narrowest cross-section the set can
    ever present, regardless of how it's rotated, which is what a
    window-fit check needs.

    Padding each projected coordinate by that atom's own radius matters:
    without it, a perfectly planar molecule like benzene projects to
    exactly zero thickness along its normal (every nucleus really does
    lie in one plane), which would wrongly say it can slide through an
    arbitrarily small slit - real atoms aren't points.
    """
    centred = positions - positions.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ vt.T
    radii = np.asarray(radii)[:, None]
    return (projected + radii).max(axis=0) - (projected - radii).min(axis=0)


def _host_pore_geometry(host, inner_radius_fraction):
    """
    The docking sphere's centre/radius, and the host's window diameters
    if determinable.

    Tries pywindow's optimised intrinsic pore analysis first - a real
    geometric calculation of the actual cavity (centre and diameter) and
    of every window (aperture) into it, computed directly on this finite
    structure with no periodic-cell workaround. Falls back to the older,
    cruder "mean atom distance from the whole molecule's centre of mass"
    heuristic if pywindow can't find an enclosed cavity here (e.g. an
    open fragment rather than a real cage) - this keeps the function
    usable on any structure, not just ones pywindow's algorithm succeeds
    on.

    Returns
    -------
    cavity_centre : np.ndarray, shape (3,)
    cavity_radius : float
    window_diameters : np.ndarray or None
        None if pywindow found no enclosed cavity/windows here - callers
        should skip any window-based fit check in that case, not treat
        it as "no windows exist".
    """
    try:
        mol = _pywindow_molecule(host)
        mol.calculate_pore_diameter_opt()
        windows = mol.calculate_windows()
        pore = mol.properties['pore_diameter_opt']
    except Exception as exc:
        warnings.warn(
            f"pywindow pore analysis raised {exc!r} - falling back to the "
            "mean-atom-distance cavity heuristic. If this keeps happening "
            "for a structure you expect to be a real porous cage, that's "
            "worth investigating rather than ignoring."
        )
    else:
        if pore['diameter'] > 0:
            centre = np.asarray(pore['centre_of_mass'])
            radius = inner_radius_fraction * (pore['diameter'] / 2)
            return centre, radius, windows

    cage_com = host.get_center_of_mass()
    inner_radius = inner_radius_fraction * np.mean(
        np.linalg.norm(host.positions - cage_com, axis=1)
    )
    return cage_com, inner_radius, None


def _check_guest_fits_windows(guests, window_diameters):
    """
    Raise if any guest type's narrowest possible cross-section is bigger
    than the host's largest window - i.e. it cannot enter the cavity
    through any known opening, in any orientation. This is a fast,
    provable pre-check, not a substitute for the real overlap-checked
    placement: passing it means entry is geometrically possible, not
    guaranteed (a non-circular window can still block orientations this
    single-number comparison can't see).

    Deliberately raises rather than warns-and-continues (unlike running
    out of placement attempts): a guest that can never fit is far more
    likely to mean the wrong guest/host pairing than an expected outcome
    of a large request, and is worth stopping for.
    """
    if window_diameters is None or len(window_diameters) == 0:
        return
    max_window = float(np.max(window_diameters))
    for i, guest in enumerate(guests):
        radii = covalent_radii[guest.numbers]
        min_extent = float(_principal_extents(guest.positions, radii).min())
        if min_extent > max_window:
            raise ValueError(
                f"guests[{i}]'s narrowest cross-section ({min_extent:.2f} A) "
                f"is larger than the host's largest window ({max_window:.2f} A) "
                "- this guest cannot enter the cavity through any known "
                "opening, in any orientation."
            )


def _next_type_index(counts, ratios):
    """
    Pick the guest type most under-represented relative to its target
    ratio so far (a smooth weighted round-robin) - lets guest types be
    mixed by ratio without knowing the eventual total count in advance,
    which is required for max_guests_in_host (the total isn't known
    until placement itself stops succeeding). With one guest type this
    always returns 0.
    """
    weights = np.asarray(ratios, dtype=float)
    weights = weights / weights.sum()
    counts_arr = np.asarray(counts, dtype=float)
    proportion_so_far = counts_arr / (counts_arr.sum() + 1.0)
    return int(np.argmax(weights - proportion_so_far))


def _attempt_random_placement(guest_template,
                              cage_com,
                              inner_radius,
                              existing_positions,
                              existing_numbers,
                              overlap_tolerance,
                              max_attempts,
                              ):
    """
    Try up to max_attempts random positions/orientations inside the
    docking sphere for one copy of guest_template, rejecting any that
    overlap an atom in existing_positions/existing_numbers (host atoms
    plus every already-placed guest atom in this complex).

    Returns
    -------
    ase.Atoms or None
        The accepted, positioned guest copy, or None if no
        non-overlapping placement was found in max_attempts tries.
    """
    guest_centred = guest_template.copy()
    guest_centred.translate(-guest_centred.get_center_of_mass())

    for _ in range(max_attempts):
        point = np.random.uniform(-inner_radius, inner_radius, 3)
        if np.linalg.norm(point) > inner_radius:
            continue
        target = cage_com + point

        g = guest_centred.copy()
        g.positions = g.positions @ random_rotation_matrix().T
        g.translate(target)

        if _atoms_overlap(existing_positions, existing_numbers,
                          g.positions, g.numbers, overlap_tolerance):
            continue

        return g

    return None


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
                       inner_radius_fraction=0.6,
                       seed=None,
                       ):
    """
    Empirically determine how many guest copies fit inside host's cavity
    without steric overlap, rather than estimating it from cavity/guest
    volume - a closed-form packing formula is either optimistic or overly
    conservative for anything but a spherical guest, whereas this reuses
    the exact same rejection-sampled overlap check that actually builds
    the complexes (see place_guest_in_host), so "fits" here means exactly
    what it means there too.

    Keeps placing guest copies (drawn from ``guests`` by ``ratios`` if
    more than one type is given - see _next_type_index) until one fails
    to find a non-overlapping position/orientation within max_attempts
    tries, and returns how many were placed before that point. This is a
    lower bound in practice, not a guaranteed global optimum (rejection
    sampling can in principle miss a tight remaining gap), but converges
    to the true packing limit as max_attempts grows.

    Before any random sampling, this also checks whether each guest type
    can physically enter the cavity at all (see
    _check_guest_fits_windows): if pywindow can identify the host's
    actual window(s) (aperture into the cavity) and a guest's narrowest
    possible cross-section is bigger than the largest one, this raises
    immediately instead of burning max_attempts on placements that could
    never succeed.

    Parameters
    ----------
    host : ase.Atoms
    guests : ase.Atoms or sequence of ase.Atoms
        One guest type, or several to mix.
    ratios : sequence of float, optional
        Relative proportion of placed copies to give each guest type
        (only meaningful with more than one entry in ``guests``).
        Defaults to an even split.
    max_attempts : int, default 1000
        Random placement attempts before concluding capacity is reached.
    overlap_tolerance : float, default 0.75
        Fraction of the sum of covalent radii used as the hard-core
        exclusion distance.
    inner_radius_fraction : float, default 0.6
        Fraction of the docking sphere's radius to actually use - a
        safety margin against sampling too close to the cavity wall.
        The radius itself comes from pywindow's optimised intrinsic pore
        diameter when an enclosed cavity can be found (real computed
        geometry, not an approximation), falling back to the older mean-
        atom-distance-from-centre-of-mass heuristic otherwise (e.g. for
        an open, non-enclosing structure that isn't a real cage cavity).
    seed : int, optional
        Seed the random search for a reproducible answer.

    Returns
    -------
    int
        The maximum number of guest copies placed without overlap.

    Raises
    ------
    ValueError
        If a guest type's narrowest cross-section exceeds the host's
        largest known window - see _check_guest_fits_windows. Not raised
        if pywindow can't determine any windows for this host.
    """
    guests, ratios = _normalise_guests(guests, ratios)
    if seed is not None:
        np.random.seed(seed)

    cage_com, inner_radius, window_diameters = _host_pore_geometry(host, inner_radius_fraction)
    _check_guest_fits_windows(guests, window_diameters)
    positions = host.positions.copy()
    numbers = host.numbers.copy()
    type_counts = [0] * len(guests)
    n_placed = 0

    while True:
        type_idx = _next_type_index(type_counts, ratios)
        placement = _attempt_random_placement(
            guests[type_idx], cage_com, inner_radius,
            positions, numbers, overlap_tolerance, max_attempts,
        )
        if placement is None:
            return n_placed

        n_placed += 1
        type_counts[type_idx] += 1
        positions = np.vstack([positions, placement.positions])
        numbers = np.concatenate([numbers, placement.numbers])


def place_guest_in_host(host,
                        guests,
                        n_complexes=1,
                        n_guests=1,
                        ratios=None,
                        max_attempts=1000,
                        overlap_tolerance=0.75,
                        inner_radius_fraction=0.6,
                        host_bond_matrix=None,
                        seed=None,
                        ):
    """
    Quick stochastic docking: generate n_complexes host-guest structures
    by randomly placing n_guests copies of one or more guest types inside
    the cage interior.

    Strategy
    --------
    1. Compute a docking sphere for the cage interior: centre and radius
       from pywindow's optimised intrinsic pore analysis (a real cavity,
       geometrically determined) when an enclosed cavity can be found,
       falling back to a cruder mean-atom-distance-from-CoM heuristic
       otherwise - see _host_pore_geometry.
    2. If pywindow also found the host's window(s) (the aperture(s) into
       that cavity), reject up front - before any random sampling - any
       guest type whose narrowest possible cross-section is larger than
       the largest window, since it cannot enter in any orientation -
       see _check_guest_fits_windows. This is a fast, necessary-but-not-
       sufficient check: passing it means entry is geometrically
       possible, not guaranteed.
    3. For each complex, sample n_guests random positions inside the
       docking sphere (rejection sampling) and apply a uniformly random
       rotation to each guest - see _attempt_random_placement.
    4. Accept a placement only if the guest does not overlap with any
       host atom or any previously placed guest in the same complex.
    5. If n_guests exceeds what actually fits (placement fails
       max_attempts times in a row for the next copy), stop there rather
       than raising - a complex with fewer guests than requested is
       returned, with a warning, instead of an error. Use
       max_guests_in_host beforehand to know the real capacity, or to
       avoid the warning by passing an n_guests you know fits.

    Each of the n_complexes configurations uses an independent random
    draw, so positions are not repeated across complexes.

    Parameters
    ----------
    host : ase.Atoms
        The cage structure.
    guests : ase.Atoms or sequence of ase.Atoms
        One guest type (applied to every copy), or several to mix -
        which copy gets which type is chosen by _next_type_index
        (a ratio-weighted round robin), not assigned up front.
    n_complexes : int
        Number of independent host-guest configurations to produce
        (default 1).
    n_guests : int
        Number of guest copies requested per complex (default 1). Capped
        to whatever actually fits - see point 4 above.
    ratios : sequence of float, optional
        Relative proportion of guest copies to give each entry in
        ``guests`` (only meaningful with more than one guest type).
        Defaults to an even split.
    max_attempts : int
        Maximum random placement attempts per guest copy before giving
        up on it (default 1000).
    overlap_tolerance : float
        Fraction of the sum of covalent radii used as the hard-core
        exclusion distance (default 0.75). Lower values allow closer contacts.
    inner_radius_fraction : float
        Fraction of the docking sphere's radius to actually use (default
        0.6) - a safety margin against sampling too close to the cavity
        wall. Increase toward 1.0 to sample closer to it. The radius
        itself is pywindow's optimised pore diameter when an enclosed
        cavity is found, falling back to the older mean-atom-distance
        heuristic otherwise - see _host_pore_geometry.
    host_bond_matrix : np.ndarray, shape (len(host), len(host)), optional
        The host's own bond-order matrix (e.g. from
        CageBuilder.get_bond_matrix()). When given, each returned complex
        is a HostGuestComplex carrying the merged bond matrix - the
        host's real bonds plus each placed guest copy's own internal
        bonds (perceived on that isolated guest alone via bonded_pairs,
        never on the merged host+guest structure - see Notes). No bond
        is ever added between host and guest, or between two guest
        copies, since encapsulation is non-covalent. Pass this matrix
        straight to write_run/write_gulp_gin-style writers instead of
        letting them re-perceive bonds from geometry on the whole
        complex, which risks a spurious "bond" wherever a guest happens
        to sit close to the host wall or to another guest copy.
    seed : int, optional
        Seed the random search for reproducible output.

    Returns
    -------
    complexes : list of ase.Atoms, or list of HostGuestComplex
        n_complexes structures. Plain ase.Atoms if host_bond_matrix is
        None (unchanged from before); HostGuestComplex
        (atoms, bond_matrix, guest_atom_indices, guest_labels) if given -
        guest_labels is the index into ``guests`` for each placed copy,
        in placement order, so a mix's composition can be recovered.

    Raises
    ------
    ValueError
        If a guest type's narrowest cross-section exceeds the host's
        largest known window - see _check_guest_fits_windows. Not raised
        if pywindow can't determine any windows for this host.

    Notes
    -----
    Perceiving bonds on a guest in isolation (bonded_pairs(guest_type))
    is unambiguous - it's a small, complete molecule, so there's no risk
    of finding a bond across the cavity to the host or to a different
    guest copy. That's the whole reason this builds the merged bond
    matrix from known pieces (host_bond_matrix, plus one bonded_pairs
    call per guest type) instead of running bond perception once on the
    final merged structure, where a guest sitting near the host wall (or
    near another guest) could otherwise be misread as covalently bonded
    to it.
    """
    guests, ratios = _normalise_guests(guests, ratios)
    if seed is not None:
        np.random.seed(seed)

    cage_com, inner_radius, window_diameters = _host_pore_geometry(host, inner_radius_fraction)
    _check_guest_fits_windows(guests, window_diameters)
    track_bonds = host_bond_matrix is not None
    n_host = len(host)

    complexes = []
    for i_complex in range(n_complexes):
        positions = host.positions.copy()
        numbers = host.numbers.copy()
        placed = []  # list of (ase.Atoms copy, type_index)
        type_counts = [0] * len(guests)

        for _ in range(n_guests):
            type_idx = _next_type_index(type_counts, ratios)
            placement = _attempt_random_placement(
                guests[type_idx], cage_com, inner_radius,
                positions, numbers, overlap_tolerance, max_attempts,
            )
            if placement is None:
                break

            placed.append((placement, type_idx))
            type_counts[type_idx] += 1
            positions = np.vstack([positions, placement.positions])
            numbers = np.concatenate([numbers, placement.numbers])

        if len(placed) < n_guests:
            warnings.warn(
                f"Complex {i_complex + 1}: only {len(placed)}/{n_guests} guests "
                f"fit without overlap (each capped at {max_attempts} placement "
                f"attempts) - returning the {len(placed)} that fit. Call "
                f"max_guests_in_host() beforehand to know the real capacity."
            )

        combined = host.copy()
        for g, _ in placed:
            combined = combined + g

        if not track_bonds:
            complexes.append(combined)
            continue

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
        complexes.append(
            HostGuestComplex(combined, bond_matrix, guest_atom_indices, guest_labels)
        )

    return complexes


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
                       inner_radius_fraction=0.6,
                       seed=None,
                       ):
    """
    Functionalise a fragment onto a host cage and optionally dock one or
    more guest molecules at random positions inside the cage interior.

    The sub_fragment is aligned to the X dummy atom site in the host,
    translated so its bonding atom sits at the X position, and merged into
    the host (removing dummy atoms from both). If guests are given,
    place_guest_in_host is called to produce n_complexes independent
    host–guest configurations via stochastic docking.

    Parameters
    ----------
    host_system : ase.Atoms
        Host cage containing one dummy atom marking the functionalisation site.
    sub_fragment : ase.Atoms
        Fragment to attach, containing one dummy atom marking its bonding end.
    bond_length : float
        Desired bond length (Å) at the new host–fragment bond (default 1.5).
    host_marker : str
        Element symbol of the bonding-site dummy atom in the host. Default 'X'.
        Use 'At' when the host was prepared via cage.functionalise()
        (full_functionalisation), which marks sites with Astatine.
    fragment_marker : str
        Element symbol of the bonding-site dummy atom in sub_fragment (default 'X').
    guests : ase.Atoms or sequence of ase.Atoms, optional
        Guest molecule(s) to dock inside the cage - one type, or several
        to mix (see place_guest_in_host). Default is None.
    n_complexes : int
        Number of independent host–guest configurations to generate (default 1).
        Ignored when guests is None.
    n_guests : int
        Number of guest copies requested per complex (default 1), capped
        to whatever fits - see place_guest_in_host.
    ratios : sequence of float, optional
        Relative proportion of guest copies per entry in ``guests``
        (only meaningful with more than one guest type).
    max_attempts : int
        Maximum random placement attempts per guest copy (default 1000).
    overlap_tolerance : float
        Fraction of sum-of-covalent-radii used as hard-core exclusion (default 0.75).
    inner_radius_fraction : float
        Fraction of mean cage radius defining the docking sphere (default 0.6).
    seed : int, optional
        Seed the random search for reproducible output.

    Returns
    -------
    result : ase.Atoms or list of ase.Atoms
        If guests is None: a single functionalised host (ase.Atoms).
        If guests is given and n_complexes == 1: a single host–guest complex.
        If guests is given and n_complexes > 1: a list of n_complexes complexes.

    Notes
    -----
    This does not track a bond matrix through the fragment-attachment
    step (unlike functionalise_isomer_sites), so host_bond_matrix /
    HostGuestComplex tracking isn't available here even when guests is
    given - the fragment attachment shifts atom indices in a way this
    function doesn't record. If you need the tracked bond matrix for a
    complex, attach the fragment first (or use an already-complete host),
    get its bond matrix (e.g. CageBuilder.get_bond_matrix()), and call
    place_guest_in_host directly with host_bond_matrix set.
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
    Convert relative ratios into exact integer counts summing to n_sites,
    via the largest-remainder method (avoids the rounding drift of naive
    round()/int() truncation, which would not sum back to n_sites).
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


def functionalise_isomer_sites(host_system,
                               fragments,
                               ratios=None,
                               site_marker='At',
                               fragment_marker='X',
                               bond_length=1.5,
                               seed=None,
                               host_bond_matrix=None,
                               ):
    """
    Attach real functional-group fragments at every active site of an
    isomer, optionally mixing several fragment types by ratio.

    Each active site is an atom of symbol ``site_marker`` (an "At" atom,
    as produced by ``full_functionalisation``/``cage.functionalise()`` and
    left in place at the sites selected by a given isomer descriptor - see
    :func:`~cage_isomer_builder.utils.read_write.generate_isomer_structure_file`).
    Each fragment is a small molecule carrying exactly one dummy atom of
    symbol ``fragment_marker`` marking its own bonding end (e.g. an NH2
    fragment built as ``Atoms('XNHH', ...)``). This function is the
    multi-site, multi-fragment generalisation of :func:`functionalise_host`,
    which only ever handles a single site.

    Which fragment type lands on which site is chosen at random (see
    ``seed``): ``ratios`` only fixes how many of the active sites get each
    fragment overall, not which particular sites.

    Parameters
    ----------
    host_system : ase.Atoms
        The isomer structure, containing one or more ``site_marker`` atoms.
    fragments : ase.Atoms or sequence of ase.Atoms
        One fragment (applied to every site), or several fragment types to
        mix across the sites.
    ratios : sequence of float, optional
        Relative proportion of active sites to give each entry in
        ``fragments`` (only meaningful when ``fragments`` has more than one
        entry). Need not sum to 1 - normalised internally. Converted to
        exact integer counts via the largest-remainder method, so they
        always sum to the number of active sites. Defaults to an even split.
    site_marker : str, default 'At'
        Element symbol marking an active site in ``host_system``.
    fragment_marker : str, default 'X'
        Element symbol marking each fragment's own bonding atom.
    bond_length : float, default 1.5
        Desired bond length (Angstrom) at each new host-fragment bond.
    seed : int, optional
        Seed for the random assignment of fragment types to sites, for
        reproducible output. ``None`` (default) draws a fresh assignment
        each call.
    host_bond_matrix : np.ndarray, shape (len(host_system), len(host_system)), optional
        The host's own bond-order matrix (e.g. from
        :func:`~cage_isomer_builder.utils.read_write.stk_2_ase_atoms`).
        When given, this function also tracks the new bonding introduced
        by attaching every fragment (each fragment's own internal bonds,
        via distance-based perception, plus the new anchor bond at each
        site) and returns a :class:`FunctionalisedIsomer` instead of a
        bare ``ase.Atoms`` - see Returns. This is what a subsequent
        UFF4MOF re-optimisation of just the new fragments (host frozen)
        needs to build its bond/angle/torsion/vdW terms.

    Returns
    -------
    ase.Atoms or FunctionalisedIsomer
        If ``host_bond_matrix`` is ``None`` (default): the host with a
        fragment attached at every active site, same as before.
        If ``host_bond_matrix`` is given: a ``FunctionalisedIsomer`` named
        tuple with fields ``atoms`` (as above), ``bond_matrix`` (the merged
        bond-order matrix for the whole decorated structure),
        ``free_atom_indices`` (indices of every newly-attached fragment
        atom - everything else is original host, unmoved by attachment),
        and ``anchor_pairs`` (one ``(host_atom_index, fragment_atom_index)``
        pair per site - the new bond formed at each attachment point).

    Raises
    ------
    ValueError
        If ``host_system`` has no ``site_marker`` atoms, or ``ratios`` is
        given with a length that doesn't match ``fragments``.
    """
    if isinstance(fragments, Atoms):
        fragments = [fragments]

    site_indices = [atom.index for atom in host_system if atom.symbol == site_marker]
    if not site_indices:
        raise ValueError(
            f"host_system has no {site_marker!r} sites to functionalise."
        )

    if ratios is None:
        ratios = [1.0] * len(fragments)
    elif len(ratios) != len(fragments):
        raise ValueError(
            f"ratios has {len(ratios)} entries but fragments has {len(fragments)}."
        )

    counts = _allocate_counts(ratios, len(site_indices))
    assignment = [i for i, count in enumerate(counts) for _ in range(count)]
    random.Random(seed).shuffle(assignment)

    # Process sites highest-index-first: functionalise_host-style attachment
    # removes the site atom and appends the fragment at the end, so working
    # top-down never disturbs the not-yet-processed (lower-index) sites.
    site_and_fragment = sorted(zip(site_indices, assignment), reverse=True)

    track_bonds = host_bond_matrix is not None
    # provenance[k] is the *original* host_system index of atom k in
    # `functionalised`, or a unique negative id for a newly-attached
    # fragment atom - a running receipt that survives the index churn from
    # every subsequent removal/append, so bonds can be resolved to final
    # indices in one pass after the whole loop finishes (position-matching
    # would also work, but this is exact rather than tolerance-based).
    if track_bonds:
        provenance = list(range(len(host_system)))
        next_new_id = -1
        site_records = []

    functionalised = host_system.copy()
    for site_index, frag_idx in site_and_fragment:
        fragment_template = fragments[frag_idx]

        # host_x (site_index) is typically an "At" atom - its unusually
        # large covalent radius makes find_connected_atoms's distance-based
        # search prone to picking up some other nearby atom instead of the
        # true bonded one in a compact structure. When the true bond graph
        # is known, use it directly instead of guessing from distance.
        host_neigh_index = (
            int(np.nonzero(host_bond_matrix[site_index])[0][0]) if track_bonds else None
        )

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
            host_neigh_orig = host_neigh_index

            n_new = len(new_fragment)
            new_ids = list(range(next_new_id, next_new_id - n_new, -1))
            next_new_id -= n_new
            # keep_local[k] is fragment_template's own atom index that ended
            # up at new_fragment's local position k - use that to find where
            # sub_neigh (the atom bonded to the anchor) landed.
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


def full_functionalisation(cage, linker_indices):
    """
    A function that Identifies all aromatic C-H bonds in the
    cage and replace the H atoms
    with "At" (Astatine) atoms to mark functional group attachment points.

    Each aromatic H atom is converted into an At atom (the FG anchor),
    with the C-At bond length extended to 1.47 Angstroms to approximate
    a C-N bond length. This avoids ambiguity with real N atoms already
    present in the ligand (e.g. amine groups in diaminobenzene).

    Schematically:
        Before:  Aromatic-C — H        (C-H bond, 1.3 Ang cutoff)
        After:   Aromatic-C — At       (C-At bond, 1.47 Ang)

    Parameters
    ----------
    cage : ase.Atoms
        The cage structure as read from a PDB/XYZ file. Must contain
        aromatic C-H bonds that will be converted to C-At anchor bonds.
        The structure is modified in place.

    Returns
    -------
    fg_anchors : ase.Atoms
        Subset of the cage containing only the At anchor atoms,
        one per aromatic C-H bond found.
    fg_anchor_indices : list of int
        Indices of the At anchor atoms in the full cage Atoms object.
        Used to map between the FG anchor subset and the full cage,
        e.g. in fg2fg_distance_count and generate_isomer_structure_file.
    fg_anchor_h_positions : np.ndarray, shape (len(fg_anchor_indices), 3)
        Each anchor's original (pre-extension) hydrogen position, aligned
        with ``fg_anchor_indices``. Lets a caller restore a non-selected
        anchor slot to a real H atom at its original bond length (see
        :func:`~cage_isomer_builder.utils.read_write.generate_isomer_structure_file`)
        instead of leaving it as an "X" placeholder.
    """

    linker_set = set(linker_indices)
    carbon_indices = neighbor_list('i', cage, {('C', 'H'): 1.3})
    hydrogen_indices = neighbor_list('j', cage, {('C', 'H'): 1.3})

    fg_anchor_indices = []
    fg_anchor_h_positions = []
    for i_carbon, i_hydrogen in zip(carbon_indices, hydrogen_indices):
        if cage[i_carbon].symbol == 'C' and i_carbon in linker_set:

            fg_anchor_h_positions.append(cage[i_hydrogen].position.copy())

            # Extend bond to C-N length (1.47 Ang), moving H atom (fix=0)
            cage.set_distance(i_carbon, i_hydrogen, 1.47, fix=0)

            # Replace H with At as FG attachment point marker
            cage[i_hydrogen].symbol = 'At'

            # Round position for numerical consistency in symmetry operations
            cage[i_hydrogen].position = [
                round(p, 3) for p in cage[i_hydrogen].position
            ]

            fg_anchor_indices.append(i_hydrogen)

    fg_anchors = cage[fg_anchor_indices]

    return fg_anchors, fg_anchor_indices, np.array(fg_anchor_h_positions)
