#!/usr/bin/env python3
from __future__ import print_function
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

import itertools
import numpy as np
from sklearn.neighbors import KernelDensity


def _cycle_decomposition(permutation):
    """Disjoint cycles of a permutation given as perm[i] = image of i."""
    n = len(permutation)
    seen = [False] * n
    cycles = []
    for start in range(n):
        if seen[start]:
            continue
        cycle = []
        j = start
        while not seen[j]:
            seen[j] = True
            cycle.append(j)
            j = permutation[j]
        cycles.append(cycle)
    return cycles


def _fixed_isomer_count(permutation, n_linkers, fg_per_linker):
    """
    Number of "exactly one active slot per linker" descriptors left
    unchanged by a single slot permutation - |Fix(g)| for one symmetry
    operation g, used by count_unique_isomers.

    A descriptor's active-slot set is invariant under a permutation only
    if it's a union of whole cycles (a permuted subset is invariant iff
    it's a union of the permutation's orbits). A cycle can only ever be
    part of such a union if it touches every linker it meets *at most
    once* - otherwise including it would activate more than one slot on
    some linker, which isn't a valid descriptor. Among the cycles that
    pass this test, we need exactly one to cover each linker: an exact
    cover counting problem, solved with the standard bitmask DP (fix the
    lowest not-yet-covered linker at each step; cycles are disjoint, so
    this can't double-count a partition).

    This replaces an earlier version that only special-cased 2-cycles
    landing on the same linker and otherwise assumed every cycle was
    "linker-simple" (touches each linker it meets exactly once) -
    true often enough for small fg_per_linker to go unnoticed, but wrong
    in general: e.g. for a linker with 8 equivalent aromatic positions
    (a biphenyl-type core) some real symmetry operations do produce
    cycles that revisit one linker more than once, which the old
    formula silently mishandled.

    Returns
    -------
    int
    """
    pieces_by_lowest_linker: dict = {}
    touched_linkers: set = set()
    for cycle in _cycle_decomposition(permutation):
        touch: dict = {}
        for slot in cycle:
            linker = slot // fg_per_linker
            touch[linker] = touch.get(linker, 0) + 1
        if any(count > 1 for count in touch.values()):
            continue  # would over-activate a linker - can never be included
        mask = 0
        for linker in touch:
            mask |= 1 << linker
        # Indexed by the piece's own lowest linker (not every linker it
        # touches) - the DP below only ever looks a piece up when that
        # lowest linker is the current mask's lowest unset bit, at which
        # point the piece's other (necessarily higher) linkers are still
        # guaranteed present in mask if the piece is to be usable at all.
        pieces_by_lowest_linker.setdefault(min(touch), []).append(mask)
        touched_linkers.update(touch)

    # Quick reject: every linker needs to be touched by at least one usable
    # piece (anywhere in that piece, not necessarily as its lowest linker),
    # or no valid descriptor can be invariant under this operation.
    if len(touched_linkers) < n_linkers:
        return 0

    full_mask = (1 << n_linkers) - 1
    ways = np.zeros(1 << n_linkers, dtype=np.int64)
    ways[0] = 1
    for mask in range(1, 1 << n_linkers):
        lowest_linker = (mask & (-mask)).bit_length() - 1
        total = 0
        for piece in pieces_by_lowest_linker.get(lowest_linker, ()):
            if (piece & mask) == piece:
                total += ways[mask ^ piece]
        ways[mask] = total
    return int(ways[full_mask])


def count_unique_isomers(transformation_library,
                         n_linkers,
                         n_fg_slots,
                         fg_per_linker=4
                         ):
    """
    A function to count the number of unique isomers by summing
    the number of colorings fixed by each symmetry operation in the
    transformation library.

    Burnside's lemma states:
        number of unique isomers = (1/|G|) * sum_{g in G} |Fix(g)|
    where |Fix(g)| is the number of colorings fixed by symmetry operation g,
    and |G| is the total number of symmetry operations (including identity).
    |Fix(g)| itself is computed exactly via exact-cover counting over g's
    cycle decomposition - see _fixed_isomer_count.

    Parameters
    ----------
    transformation_library : list of list of int
        List of permutations representing symmetry operations of the cage,
        as returned by build_transformation_library. Each permutation is a
        list of length n_fg_slots where permutation[i] is the index that
        slot i maps to under the symmetry operation.
    n_linkers : int
        Number of linkers in the cage. Determines the raw combinatorial
        space (fg_per_linker**n_linkers isomers before symmetry reduction)
        and the size of the 2**n_linkers bitmask DP table used internally.
    n_fg_slots : int
        Total number of FG anchor slots in the cage (n_linkers * fg_per_linker).
        Used to iterate over all slots when finding cycles.
    fg_per_linker : int, default 4
        Functional-group slots per linker (4 for a simple benzene-based
        linker, but this is read from the cage's actual anchor count -
        see CageBuilder.functionalise - so e.g. 8 for a biphenyl-type
        linker works the same way).

    Returns
    -------
    n_unique_isomers : float
        Number of symmetry-unique isomers, computed as the Burnside average.
        Should be a whole number in practice; returned as float due to
        the division by the number of symmetry operations.

    Raises
    ------
    ValueError
        If n_linkers is large enough that the 2**n_linkers bitmask DP
        table becomes impractical (limit: 24, i.e. ~16.7M entries).
    """
    if n_linkers > 24:
        raise ValueError(
            f"count_unique_isomers: {n_linkers} linkers needs a "
            f"2**{n_linkers} = {2 ** n_linkers:,} entry bitmask table per "
            "symmetry operation, which is impractical (limit: 24). For a "
            "periodic structure, consider reducing to its asymmetric unit."
        )

    fixed_colorings = [fg_per_linker ** n_linkers]  # identity always fixes everything
    for permutation in transformation_library:
        if len(set(permutation)) != n_fg_slots:
            # Not a genuine permutation of the slots - shouldn't normally
            # happen, but can't contribute a valid fixed count if it does.
            fixed_colorings.append(0)
            continue
        fixed_colorings.append(
            _fixed_isomer_count(list(permutation), n_linkers, fg_per_linker)
        )

    n_symmetry_ops = len(transformation_library) + 1  # +1 for identity
    return sum(fixed_colorings) / n_symmetry_ops


def fg2fg_distance_count(ase_atom,
                         fg_anchor_indices,
                         fg_per_linker=4
                         ):
    """
    Parameters
    ----------
    ase_atom : ase.Atoms
        The cage structure containing all functional group slots as At atoms.
    fg_anchor_indices : list of int
        Indices of the active "At" atoms in the cage, one per linker, marking
        the attachment point of the functional group on each linker.
        Each index corresponds to an "At" atom that replaced an aromatic H
        during functionalisation (see full_functionalisation function).

    Returns
    -------
    distance_keys : list of float
        Sorted unique inter-linker FG-to-FG distances in Angstroms.
    total_count : list of int
        Total number of active FG pairs at each distance.
    inner_inner_count : list of int
        Pairs where both anchors point inward to the cage cavity (endo-endo).
    inner_outer_count : list of int
        Pairs where one anchor is inner and one is outer (endo-exo).
    outer_outer_count : list of int
        Pairs where both anchors point outward from the cage cavity (exo-exo).
    """

    paired_combinations = [
        paired
        for paired in itertools.combinations(range(len(ase_atom)), 2)
        if int(paired[0] / fg_per_linker) != int(paired[1] / fg_per_linker)
    ]

    endo_anchor_indices = []
    for linker in range(int(len(ase_atom) / fg_per_linker)):
        distances = []
        for slot in range(fg_per_linker):
            dis = round(
                np.linalg.norm(ase_atom[linker * fg_per_linker + slot].position), 1)
            distances += [dis]
        distances = np.array(distances)
        min_dist_indices = np.where(distances == distances.min())[0]
        endo_anchor_indices += [linker * fg_per_linker + slot for slot in min_dist_indices]

    distance_keys = []
    for paired in list(itertools.combinations(range(len(ase_atom)), 2)):
        if int(paired[0] / fg_per_linker) == int(paired[1] / fg_per_linker):
            continue  # skip pairs on the same linker
        distance_keys.append(
            round(ase_atom.get_distance(paired[0], paired[1]), 2)
            )
    distance_keys = sorted(list(set(distance_keys)))

    pairs_at_distance = dict((d, []) for d in distance_keys)
    for paired in paired_combinations:
        k = round(ase_atom.get_distance(paired[0], paired[1]), 2)
        pairs_at_distance[k].append(paired)

    count_total = dict((d, 0) for d in distance_keys)
    pair_weight = dict((paired, 1) for paired in paired_combinations)
    for d in distance_keys:
        for pair in pairs_at_distance[d]:
            count_total[d] += pair_weight[pair]

    count3ds = dict((d, [0, 0, 0]) for d in distance_keys)
    for d in distance_keys:
        for pair in pairs_at_distance[d]:
            is_exo_0 = int(pair[0] not in endo_anchor_indices)  # 0=endo, 1=exo
            is_exo_1 = int(pair[1] not in endo_anchor_indices)  # 0=endo, 1=exo
            count3ds[d][is_exo_0 + is_exo_1] += pair_weight[pair]

    total_count = [count_total[d] for d in distance_keys]
    inner_inner_count = [count3ds[d][0] for d in distance_keys]
    inner_outer_count = [count3ds[d][1] for d in distance_keys]
    outer_outer_count = [count3ds[d][2] for d in distance_keys]
    return (
        distance_keys,
        total_count,
        inner_inner_count,
        inner_outer_count,
        outer_outer_count
    )


def kernel_density_estimation(distance_values,
                              pair_counts,
                              n_bins):
    """
    A function to estimate a smoothed probability density from a distance
    histogram using Gaussian Kernel Density Estimation (KDE).

    The histogram is first expanded into a flat list of raw data points
    where each distance repeated by its count, then fitted with a
    Gaussian KDE. The resulting density is rescaled to preserve the
    total count, making it directly comparable to the original histogram.

    Parameters
    ----------
    distance_values : list of float
        Sorted unique inter-linker FG-to-FG distances in Angstroms.
        Corresponds to distance_keys returned by fg2fg_distance_count.
    pair_counts : list of int
        Number of FG pairs at each distance in distance_values.
        Corresponds to count_total, count_both_inner, count_mixed, or
        count_both_outer returned by fg2fg_distance_count.
    n_bins : int
        Number of points at which to evaluate the KDE curve, evenly
        spaced from 0 to max(distance_values) * 1.05.

    Returns
    -------
    x_eval : numpy.ndarray of shape (n_bins, 1)
        Evenly spaced distances at which the KDE is evaluated, in Angstroms.
    y_density : numpy.ndarray of shape (n_bins,)
        KDE density values at each point in x_eval, rescaled so that
        the sum equals the total number of FG pairs.

    Notes
    -----
    - Gaussian kernel with bandwidth=0.1 Angstroms is used.
    - The density is rescaled by len(data)/sum(y_density) to match
      the original pair count scale rather than integrating to 1.
    """

    raw_data = []
    for distance, count in zip(distance_values, pair_counts):
        raw_data += [distance] * count

    x_train = np.array(raw_data)[:, np.newaxis]

    # Evaluation grid from 0 to 5% beyond max distance
    x_eval = np.linspace(0, max(distance_values) * 1.05, n_bins)[:, np.newaxis]

    # Fit Gaussian KDE and evaluate
    kde_model = KernelDensity(kernel='gaussian', bandwidth=0.1)
    kde_model.fit(x_train)
    log_density = kde_model.score_samples(x_eval)

    # Convert from log density and rescale to match total pair count
    y_density = np.exp(log_density)
    y_density = y_density * len(raw_data) / sum(y_density)

    return x_eval, y_density


def convert_base4_to_base10(isomer_descriptor,
                            n_linkers,
                            fg_per_linker=4):
    """
    A function to convert a base-4 isomer descriptor
    into a base-10 integer index.

    Each element of the isomer descriptor encodes the rotational state
    (0-3) of a functional group anchor on a given linker. The decimal
    index is computed as a base-4 number where each digit corresponds
    to the rotational state of one linker.

    Schematically for n_linkers = 4:
        isomer_descriptor = [slot_0, slot_1, slot_2, slot_3]
        decimal_index = r0 * 4^3 + r1 * 4^2 + r2 * 4^1 + r3 * 4^0
        where r_i = isomer_descriptor[i] % 4  (rotational state of linker i)

    This is the inverse of convert_base10_to_base4.

    Parameters
    ----------
    isomer_descriptor : list of int
        Base-4 isomer descriptor of length n_linkers, where each element
        is the global FG anchor slot index active on that linker.
        The rotational state of linker i is recovered as
        isomer_descriptor[i] % 4, giving a value in range 0-3.
        e.g. [0, 5, 9, 14] for a cage with 4 linkers.
    n_linkers : int
        Number of linkers in the cage. Determines the number of base-4
        digits and the positional weight of each digit.

    Returns
    -------
    decimal_index : int
        Unique base-10 integer index of the isomer in range
        0 to 4**n_linkers - 1. Used to index into the unique flag
        list in the main enumeration loop.
        e.g. [0, 1, 0, 2] -> 0*64 + 1*16 + 0*4 + 2*1 = 18
    """

    decimal_index = 0
    for i in range(n_linkers):
        rotational_state = isomer_descriptor[i] % fg_per_linker
        if rotational_state != 0:
            decimal_index += rotational_state * (fg_per_linker ** (n_linkers - 1 - i))

    return decimal_index


def convert_base10_to_base4(decimal_index, n_linkers, fg_per_linker=4):
    """
    Convert a base-10 (decimal) integer index into a base-4 isomer descriptor.

    The decimal index is decomposed into base-4 digits by successive division,
    where each digit represents the rotational state (0-3) of one linker.
    The digits are then converted into global FG anchor slot indices by
    adding the linker offset (linker_index * 4).

    Schematically for n_linkers = 4:
        decimal_index = 18
        base-4 digits = [0, 1, 0, 2]  (most to least significant)
        isomer_descriptor = [0*4+0, 1*4+1, 2*4+0, 3*4+2]
                          = [0, 5, 8, 14]

    This is the inverse of convert_base4_to_base10.

    Parameters
    ----------
    decimal_index : int
        Unique base-10 integer index of the isomer in range
        0 to 4**n_linkers - 1. Used to index into the uniqueflag
        list in the main enumeration loop.
    n_linkers : int
        Number of linkers in the cage. Determines the number of base-4
        digits and the global slot index offset for each linker.

    Returns
    -------
    isomer_descriptor : list of int
        Base-4 isomer descriptor of length n_linkers, where each element
        is the global FG anchor slot index active on that linker.
        The rotational state of linker i is recovered as
        isomer_descriptor[i] % 4, giving a value in range 0-3.
        e.g. decimal_index=18, n_linkers=4 -> [0, 5, 8, 14]
    """

    isomer_descriptor = []
    base_digits = []

    while decimal_index > 0:
        base_digits.append(decimal_index % fg_per_linker)
        decimal_index = decimal_index // fg_per_linker

    base_digits = base_digits + [0] * (n_linkers - len(base_digits))
    for linker_index in range(len(base_digits)):
        rotational_state = base_digits[n_linkers - 1 - linker_index]
        isomer_descriptor += [linker_index * fg_per_linker + rotational_state]

    return isomer_descriptor


def _is_canonical_isomer(i_isomer,
                         descriptor,
                         tl,
                         n_linkers,
                         fg_per_linker):
    """
    True iff i_isomer is the smallest decimal index in its own symmetry
    orbit - i.e. the representative enumerate_isomers()/iter_unique_isomers
    would keep for this descriptor's equivalence class.

    Checked directly against every image in the orbit rather than via a
    shared "already visited" array, so it needs no memory proportional to
    fg_per_linker**n_linkers - see iter_unique_isomers, which this backs.
    """
    eq_isomers = [tl[i] for i in descriptor]
    eq_isomers = list(zip(*eq_isomers))
    for ei in eq_isomers:
        ei_index = convert_base4_to_base10(sorted(ei), n_linkers, fg_per_linker)
        if ei_index < i_isomer:
            return False
    return True


def iter_unique_isomers(tl, n_linkers, fg_per_linker=4, limit=None):
    """
    Generator over symmetry-unique isomers, yielding (decimal_index,
    descriptor) pairs in ascending index order - the memory-safe
    counterpart to the array-based loop in CageBuilder.enumerate_isomers.

    enumerate_isomers's original approach allocates a flag array of size
    fg_per_linker**n_linkers up front (one entry per raw combination) so it
    can mark equivalents as visited while walking forward - this is what
    raises MemoryError on a real periodic MOF cell with dozens of linkers
    (4**24 ~= 2.8e14 entries). This generator gets the identical result
    (i_isomer is kept iff it's the smallest index in its own symmetry
    orbit - see _is_canonical_isomer) by checking each candidate directly
    against its orbit instead of consulting a shared array, so memory use
    is O(1) regardless of n_linkers.

    This does not change the time complexity - it still walks candidate
    indices in order and is only practical as far as that walk itself is
    fast enough. What it buys you: (1) no more crash on large periodic
    cells, just increasingly long runtime, and (2) a `limit`, so you can
    ask for "the first K unique isomers" without needing to finish the
    walk - practical even on an astronomically large space, since
    canonical representatives occur roughly 1-in-|G| along the way (|G|
    being the symmetry group order), not 1-in-fg_per_linker**n_linkers.

    Parameters
    ----------
    tl : list of tuple
        Transformation library in the "zipped"/slot-major format returned
        by CageBuilder._get_transformation_library - tl[slot] gives that
        slot's image under every non-identity symmetry operation.
    n_linkers : int
    fg_per_linker : int, default 4
    limit : int, optional
        Stop after yielding this many unique isomers. None (default)
        walks the full fg_per_linker**n_linkers space.

    Yields
    ------
    i_isomer : int
        Decimal index of the isomer (see convert_base10_to_base4).
    descriptor : list of int
        Base-fg_per_linker isomer descriptor.
    """
    total = fg_per_linker ** n_linkers
    n_yielded = 0
    i_isomer = 0
    while i_isomer < total:
        descriptor = convert_base10_to_base4(i_isomer, n_linkers, fg_per_linker)
        if _is_canonical_isomer(i_isomer, descriptor, tl, n_linkers, fg_per_linker):
            yield i_isomer, descriptor
            n_yielded += 1
            if limit is not None and n_yielded >= limit:
                return
        i_isomer += 1


def is_higher_greater(isomer_descriptor_1,
                      isomer_descriptor_2
                      ):
    """
    A function to compare two isomer descriptors
    lexicographically, which returns True if
    the first is greater than the second.

    Parameters
    ----------
    isomer_descriptor_1 : list of int
        First base-4 isomer descriptor of length n_linkers, where each
        element is the global FG anchor slot index active on that linker.
        e.g. [0, 5, 9, 14]
    isomer_descriptor_2 : list of int
        Second base-4 isomer descriptor of length n_linkers to compare
        against. Must be the same length as isomer_descriptor_1.
        e.g. [0, 5, 9, 10]

    Returns
    -------
    bool
        True if isomer_descriptor_1 is lexicographically greater than
        isomer_descriptor_2, False otherwise (including when equal).

    """

    for slot_1, slot_2 in zip(isomer_descriptor_1, isomer_descriptor_2):
        if slot_1 > slot_2:
            return True
        elif slot_1 < slot_2:
            return False
    return False
