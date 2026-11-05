#!/usr/bin/env python3
"""
Isomer counting with one or k groups per linker (Burnside's lemma),
enumeration of the single-group isomers, and distance-histogram helpers.
"""
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

import itertools
import math
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
    Number of descriptors with exactly one active slot per linker that a single
    slot permutation leaves unchanged, |Fix(g)| (used by count_unique_isomers).

    A set of active slots is invariant under a permutation if and only if it is
    a union of whole cycles. A cycle can be part of such a union only if it
    touches every linker it meets at most once, and exactly one usable cycle
    must cover each linker: an exact-cover count, solved with a bitmask dynamic
    programme over the linkers (always extending the lowest uncovered linker,
    so no partition is counted twice). Cycles may revisit a linker, as happens
    for linkers with eight equivalent positions such as biphenylene.

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
            continue  # would activate two slots of one linker
        mask = 0
        for linker in touch:
            mask |= 1 << linker
        # Pieces are indexed by their lowest linker: the DP below looks a
        # piece up only when that linker is the lowest bit of the mask.
        pieces_by_lowest_linker.setdefault(min(touch), []).append(mask)
        touched_linkers.update(touch)

    # Every linker must be touched by some usable piece, or no descriptor
    # is invariant under this operation.
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


def _fixed_isomer_count_k(permutation, n_linkers, fg_per_linker, k):
    """
    Number of descriptors with exactly k active slots per linker that a single
    slot permutation leaves unchanged, |Fix(g)| (used by
    count_unique_isomers_k); the k = 1 case is _fixed_isomer_count.

    A set S of active slots is invariant under g if and only if it is a union of
    whole cycles of g, so

        |Fix(g)| = #{ C subset of cycles(g) :
                      sum over c in C of touch_c(L) = k for every linker L }

    where touch_c(L) is the number of slots of cycle c on linker L. A cycle with
    touch_c(L) > k for some L can never be selected and is discarded.

    The count is a sparse dynamic programme over per-linker budgets: each cycle
    is added or not, states that exceed k on a linker are dropped, and the
    answer is the count of the state with every linker at k. Only reachable
    states are stored, which stays small because the cycles of a symmetry
    operation touch few linkers each.

    Returns
    -------
    int
    """
    usable_cycles = []
    touched_linkers: set = set()
    for cycle in _cycle_decomposition(permutation):
        touch: dict = {}
        for slot in cycle:
            linker = slot // fg_per_linker
            touch[linker] = touch.get(linker, 0) + 1
        if any(count > k for count in touch.values()):
            continue  # would exceed k slots on a linker
        usable_cycles.append(touch)
        touched_linkers.update(touch)

    # Every linker needs a usable cycle touching it to reach k > 0 slots.
    if k > 0 and len(touched_linkers) < n_linkers:
        return 0

    states = {frozenset(): 1}
    for touch in usable_cycles:
        next_states: dict = {}
        for state, ways in states.items():
            # cycle stays out
            next_states[state] = next_states.get(state, 0) + ways
            # cycle goes in, if no linker's budget is exceeded
            budgets = dict(state)
            over_budget = False
            for linker, t in touch.items():
                new_count = budgets.get(linker, 0) + t
                if new_count > k:
                    over_budget = True
                    break
                budgets[linker] = new_count
            if not over_budget:
                new_state = frozenset(budgets.items())
                next_states[new_state] = next_states.get(new_state, 0) + ways
        states = next_states

    # For k=0 every cycle was discarded above, so the empty selection
    # (the empty descriptor, always invariant) is the single valid state.
    target = frozenset((linker, k) for linker in range(n_linkers)) if k > 0 else frozenset()
    return states.get(target, 0)


def count_unique_isomers_k(transformation_library,
                           n_linkers,
                           n_fg_slots,
                           fg_per_linker=4,
                           active_per_linker=1,
                           ):
    """
    Number of symmetry-unique isomers with exactly ``active_per_linker`` active
    slots on every linker, from Burnside's lemma (count_unique_isomers is the
    case ``active_per_linker=1``).

    Burnside's lemma: the number of orbits of a group G acting on a set is

        n_unique = (1/|G|) * sum over g in G of |Fix(g)|

    The set here is every choice of k active slots per linker, so the identity
    fixes ``C(fg_per_linker, k) ** n_linkers`` descriptors; every other |Fix(g)|
    comes from g's cycles (see _fixed_isomer_count_k). The average is a whole
    number for a group action, and the division is checked.

    Checks for a result: k = 0 and k = fg_per_linker give 1; k and
    fg_per_linker - k give the same count; k = 1 equals count_unique_isomers.

    Parameters
    ----------
    transformation_library : list of list of int
        Symmetry operations as slot permutations (operation-major), as for
        count_unique_isomers.
    n_linkers : int
        Number of linkers.
    n_fg_slots : int
        Total number of slots (n_linkers * fg_per_linker).
    fg_per_linker : int, default 4
        Slots per linker.
    active_per_linker : int, default 1
        Active slots on every linker, between 0 and fg_per_linker.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If active_per_linker is out of range, or the Burnside average is not a
        whole number (the operations are not a group on these slots).

    Examples
    --------
    >>> from cage_isomer_builder.utils.util import count_unique_isomers_k
    >>> swap = [[2, 3, 0, 1]]                 # two 2-slot linkers swapped by symmetry
    >>> count_unique_isomers_k(swap, n_linkers=2, n_fg_slots=4, fg_per_linker=2, active_per_linker=1)
    3
    """
    k = active_per_linker
    if not 0 <= k <= fg_per_linker:
        raise ValueError(
            f"active_per_linker={k} must be between 0 and fg_per_linker={fg_per_linker}."
        )

    fixed_colorings = [math.comb(fg_per_linker, k) ** n_linkers]  # identity
    for permutation in transformation_library:
        if len(set(permutation)) != n_fg_slots:
            fixed_colorings.append(0)
            continue
        fixed_colorings.append(
            _fixed_isomer_count_k(list(permutation), n_linkers, fg_per_linker, k)
        )

    n_symmetry_ops = len(transformation_library) + 1  # +1 for identity
    total = sum(fixed_colorings)
    n_unique, remainder = divmod(total, n_symmetry_ops)
    if remainder != 0:
        raise ValueError(
            f"Burnside average is not a whole number ({total}/{n_symmetry_ops}) "
            "- transformation_library does not form a closed symmetry group "
            "over these FG slots."
        )
    return n_unique


def count_unique_isomers(transformation_library,
                         n_linkers,
                         n_fg_slots,
                         fg_per_linker=4
                         ):
    """
    Number of symmetry-unique isomers with one active slot per linker, from
    Burnside's lemma:

        n_unique = (1/|G|) * sum over g in G of |Fix(g)|

    where |Fix(g)| is the number of descriptors left unchanged by operation g
    (computed exactly from g's cycles, see _fixed_isomer_count) and |G| counts
    the identity.

    Parameters
    ----------
    transformation_library : list of list of int
        Symmetry operations as slot permutations: ``permutation[i]`` is the
        slot that slot i maps to.
    n_linkers : int
        Number of linkers; there are ``fg_per_linker ** n_linkers`` raw
        descriptors and the internal table has ``2 ** n_linkers`` entries.
    n_fg_slots : int
        Total number of slots (n_linkers * fg_per_linker).
    fg_per_linker : int, default 4
        Slots per linker (4 for a phenylene, 8 for a biphenylene, ...).

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If n_linkers is above 24 (the table becomes impractical; use
        ``rgroup.count_rgroup_isomers``), or the Burnside average is not a whole
        number (the operations are not a group on these slots).

    Examples
    --------
    >>> from cage_isomer_builder.utils.util import count_unique_isomers
    >>> count_unique_isomers([[2, 3, 0, 1]], n_linkers=2, n_fg_slots=4, fg_per_linker=2)
    3
    """
    if n_linkers > 24:
        raise ValueError(
            f"count_unique_isomers: {n_linkers} linkers needs a "
            f"2**{n_linkers} = {2 ** n_linkers:,} entry bitmask table per "
            "symmetry operation, which is impractical (limit: 24). For a "
            "periodic structure, consider reducing to its asymmetric unit."
        )

    fixed_colorings = [fg_per_linker ** n_linkers]  # the identity fixes everything
    for permutation in transformation_library:
        if len(set(permutation)) != n_fg_slots:
            # Not a permutation of the slots: it fixes no descriptor.
            fixed_colorings.append(0)
            continue
        fixed_colorings.append(
            _fixed_isomer_count(list(permutation), n_linkers, fg_per_linker)
        )

    n_symmetry_ops = len(transformation_library) + 1  # +1 for identity
    total = sum(fixed_colorings)
    n_unique, remainder = divmod(total, n_symmetry_ops)
    if remainder != 0:
        raise ValueError(
            f"Burnside average is not a whole number ({total}/{n_symmetry_ops}) "
            "- transformation_library does not form a closed symmetry group "
            "over these FG slots."
        )
    return n_unique


def fg2fg_distance_count(ase_atom,
                         fg_anchor_indices,
                         fg_per_linker=4
                         ):
    """
    Legacy FG-FG pair counts, kept for compatibility. ``CageBuilder.get_statistics``
    does not use it: its endo/exo rule (the slots of each linker closest to the
    origin are endo) assumes the structure is centred at the origin and depends
    on sub-Angstrom differences when a ring lies face-on to the pore, and its
    distances are not minimum-image for periodic cells. Use
    ``CageBuilder.get_statistics`` or
    :func:`cage_isomer_builder.utils.distributions.pair_descriptor` instead.

    Parameters
    ----------
    ase_atom : ase.Atoms
        The site atoms, in slot order.
    fg_anchor_indices : list of int
        Slot indices (one entry per site).
    fg_per_linker : int, default 4

    Returns
    -------
    distance_keys : list of float
        Sorted distinct distances between slots on different linkers.
    total_count : list of int
        Number of slot pairs at each distance.
    inner_inner_count, inner_outer_count, outer_outer_count : list of int
        The same, split by the legacy endo/exo rule.

    Examples
    --------
    >>> from ase import Atoms
    >>> from cage_isomer_builder.utils.util import fg2fg_distance_count
    >>> slots = Atoms("X4", positions=[[0, 0, 0], [1, 0, 0], [5, 0, 0], [6, 0, 0]])
    >>> keys, total, *_ = fg2fg_distance_count(slots, [0, 1, 2, 3], fg_per_linker=2)
    >>> [float(k) for k in keys], total                # pairs on different linkers only
    ([4.0, 5.0, 6.0], [1, 2, 1])
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
    Smoothed version of a distance histogram (Gaussian kernel density
    estimate).

    Each distance is repeated by its count, a Gaussian KDE (bandwidth 0.1
    Angstrom) is fitted, and the curve is scaled so that its values add up to
    the total count, so it can be drawn on the same axis as the histogram.

    Parameters
    ----------
    distance_values : list of float
        Sorted distinct distances (Angstrom).
    pair_counts : list of int
        Number of pairs at each distance.
    n_bins : int
        Number of evaluation points, from 0 to 1.05 * max(distance_values).

    Returns
    -------
    x_eval : numpy.ndarray, shape (n_bins, 1)
        Evaluation grid (Angstrom).
    y_density : numpy.ndarray, shape (n_bins,)
        Curve values, adding up to the total pair count.

    Examples
    --------
    >>> from cage_isomer_builder.utils.util import kernel_density_estimation
    >>> x, y = kernel_density_estimation([4.0, 5.0], [1, 2], n_bins=200)
    >>> x.shape, round(float(y.sum()), 6)        # rescaled to the pair count
    ((200, 1), 3.0)
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
    Integer index of an isomer descriptor (one active slot per linker).

    The local slot of linker i, ``r_i = isomer_descriptor[i] % fg_per_linker``,
    is digit i of a base-``fg_per_linker`` number, most significant first.
    For four linkers with four slots each:

        decimal_index = r0 * 4^3 + r1 * 4^2 + r2 * 4^1 + r3 * 4^0

    This is the inverse of :func:`convert_base10_to_base4`.

    Parameters
    ----------
    isomer_descriptor : list of int
        The active slot of each linker, as a global slot index, e.g.
        ``[0, 5, 9, 14]`` for four linkers.
    n_linkers : int
        Number of linkers (number of digits).
    fg_per_linker : int, default 4
        Slots per linker (the base).

    Returns
    -------
    decimal_index : int
        Index of the descriptor, from 0 to fg_per_linker**n_linkers - 1,
        e.g. [0, 1, 0, 2] -> 0*64 + 1*16 + 0*4 + 2*1 = 18.

    Examples
    --------
    >>> from cage_isomer_builder.utils.util import convert_base4_to_base10
    >>> convert_base4_to_base10([0, 5, 8, 14], n_linkers=4)    # local slots 0, 1, 0, 2
    18
    """

    decimal_index = 0
    for i in range(n_linkers):
        rotational_state = isomer_descriptor[i] % fg_per_linker
        if rotational_state != 0:
            decimal_index += rotational_state * (fg_per_linker ** (n_linkers - 1 - i))

    return decimal_index


def convert_base10_to_base4(decimal_index, n_linkers, fg_per_linker=4):
    """
    Isomer descriptor of an integer index (the inverse of
    :func:`convert_base4_to_base10`).

    The index is split into base-``fg_per_linker`` digits, most significant
    first, and digit i becomes the global slot ``i * fg_per_linker + digit``.
    For four linkers with four slots each, 18 has the digits 0, 1, 0, 2 and
    gives ``[0, 5, 8, 14]``.

    Parameters
    ----------
    decimal_index : int
        Index of the descriptor, from 0 to ``fg_per_linker ** n_linkers - 1``.
    n_linkers : int
        Number of linkers (number of digits).
    fg_per_linker : int, default 4
        Slots per linker (the base).

    Returns
    -------
    isomer_descriptor : list of int
        The active slot of each linker, as a global slot index.

    Examples
    --------
    >>> from cage_isomer_builder.utils.util import convert_base10_to_base4
    >>> convert_base10_to_base4(18, n_linkers=4)
    [0, 5, 8, 14]
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
    True if ``i_isomer`` is the smallest decimal index in its symmetry orbit,
    i.e. the representative that iter_unique_isomers keeps. Checked against the
    orbit directly, so no array over all raw descriptors is needed.
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
    Yield the symmetry-unique isomers with one active slot per linker as
    (decimal_index, descriptor) pairs, in ascending index order.

    A candidate is kept if it is the smallest index of its orbit (see
    _is_canonical_isomer). Each candidate is checked against its own orbit, so
    memory use does not grow with the number of linkers; the run time grows with
    ``fg_per_linker ** n_linkers`` unless ``limit`` stops early. Canonical
    representatives occur about once per |G| indices, so a limit is reached
    quickly even when the full walk is not feasible.

    Parameters
    ----------
    tl : list of tuple
        Slot-major transformation library: ``tl[slot]`` gives the slot's image
        under every non-identity operation.
    n_linkers : int
    fg_per_linker : int, default 4
    limit : int, optional
        Stop after this many isomers. Default: walk every raw descriptor.

    Yields
    ------
    i_isomer : int
        Decimal index of the isomer (see convert_base10_to_base4).
    descriptor : list of int
        The active slot of each linker.

    Examples
    --------
    >>> from cage_isomer_builder.utils.util import iter_unique_isomers
    >>> library = [(2,), (3,), (0,), (1,)]          # slot-major: one swap of two linkers
    >>> [d for _, d in iter_unique_isomers(library, n_linkers=2, fg_per_linker=2)]
    [[0, 2], [0, 3], [1, 3]]
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
    True if the first isomer descriptor is lexicographically greater than
    the second.

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

    Examples
    --------
    >>> from cage_isomer_builder.utils.util import is_higher_greater
    >>> is_higher_greater([0, 5, 9, 14], [0, 5, 9, 10]), is_higher_greater([0, 4], [0, 4])
    (True, False)
    """

    for slot_1, slot_2 in zip(isomer_descriptor_1, isomer_descriptor_2):
        if slot_1 > slot_2:
            return True
        elif slot_1 < slot_2:
            return False
    return False
