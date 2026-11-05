#!/usr/bin/env python3
"""
Fast unit tests for the Burnside isomer-counting machinery in
cage_isomer_builder.utils.util, including the active_per_linker=k
generalisation (count_unique_isomers_k).

Run directly (python3 tests/test_isomer_counting.py) or under pytest.
Uses small synthetic symmetry groups (no cage build needed), checked
against brute-force orbit counting.
"""

import itertools
import math
import random

import pytest

from cage_isomer_builder.utils.util import (
    count_unique_isomers,
    count_unique_isomers_k,
)


def _brute_force_orbit_count(transformation_library, n_linkers, fg_per_linker, k):
    """Ground truth: enumerate every k-per-linker descriptor, apply the
    group, count orbits."""
    n = n_linkers * fg_per_linker
    full_group = [list(range(n))] + [list(p) for p in transformation_library]

    per_linker_choices = []
    for linker in range(n_linkers):
        base = linker * fg_per_linker
        slots = range(base, base + fg_per_linker)
        per_linker_choices.append(list(itertools.combinations(slots, k)))

    seen = set()
    n_orbits = 0
    for combo in itertools.product(*per_linker_choices):
        descriptor = frozenset(itertools.chain.from_iterable(combo))
        if descriptor in seen:
            continue
        n_orbits += 1
        for g in full_group:
            seen.add(frozenset(g[i] for i in descriptor))
    return n_orbits


def _cyclic_group(n_linkers, fg_per_linker, seed):
    """A closed cyclic permutation group that maps whole linkers to whole
    linkers (identity excluded, as count_unique_isomers expects)."""
    rng = random.Random(seed)
    n = n_linkers * fg_per_linker
    linker_perm = list(range(n_linkers))
    rng.shuffle(linker_perm)
    perm = [0] * n
    for linker in range(n_linkers):
        local = list(range(fg_per_linker))
        rng.shuffle(local)
        for i in range(fg_per_linker):
            perm[linker * fg_per_linker + i] = linker_perm[linker] * fg_per_linker + local[i]

    group = []
    cur = perm
    identity = list(range(n))
    while cur != identity:
        group.append(cur)
        cur = [perm[x] for x in cur]
    return group


def test_k_matches_brute_force_on_random_groups() -> None:
    for trial in range(10):
        rng = random.Random(trial)
        n_linkers = rng.randint(2, 4)
        fg_per_linker = rng.randint(2, 4)
        n_fg_slots = n_linkers * fg_per_linker
        group = _cyclic_group(n_linkers, fg_per_linker, seed=trial)
        for k in range(fg_per_linker + 1):
            fast = count_unique_isomers_k(
                group, n_linkers, n_fg_slots, fg_per_linker, active_per_linker=k
            )
            brute = _brute_force_orbit_count(group, n_linkers, fg_per_linker, k)
            assert fast == brute, (
                f"trial={trial} n_linkers={n_linkers} fg_per_linker={fg_per_linker} "
                f"k={k}: burnside={fast} brute={brute}"
            )


def test_k1_reproduces_count_unique_isomers() -> None:
    for trial in range(10):
        rng = random.Random(100 + trial)
        n_linkers = rng.randint(2, 5)
        fg_per_linker = rng.randint(2, 4)
        n_fg_slots = n_linkers * fg_per_linker
        group = _cyclic_group(n_linkers, fg_per_linker, seed=100 + trial)
        k1 = count_unique_isomers_k(
            group, n_linkers, n_fg_slots, fg_per_linker, active_per_linker=1
        )
        legacy = count_unique_isomers(group, n_linkers, n_fg_slots, fg_per_linker)
        assert k1 == legacy, f"trial={trial}: k=1 gave {k1}, legacy gave {legacy}"


def test_complement_and_edge_identities() -> None:
    """count(k) == count(fg_per_linker - k); k=0 and k=fg_per_linker give 1."""
    for trial in range(10):
        rng = random.Random(200 + trial)
        n_linkers = rng.randint(2, 5)
        fg_per_linker = rng.randint(2, 4)
        n_fg_slots = n_linkers * fg_per_linker
        group = _cyclic_group(n_linkers, fg_per_linker, seed=200 + trial)

        counts = {
            k: count_unique_isomers_k(
                group, n_linkers, n_fg_slots, fg_per_linker, active_per_linker=k
            )
            for k in range(fg_per_linker + 1)
        }
        assert counts[0] == 1
        assert counts[fg_per_linker] == 1
        for k in counts:
            assert counts[k] == counts[fg_per_linker - k], (
                f"trial={trial} k={k}: {counts[k]} != {counts[fg_per_linker - k]}"
            )


def test_identity_only_group_gives_binomial_power() -> None:
    """With no symmetry beyond identity, the count is the raw
    C(fg_per_linker, k) ** n_linkers."""
    n_linkers, fg_per_linker = 3, 4
    n_fg_slots = n_linkers * fg_per_linker
    for k in range(fg_per_linker + 1):
        n = count_unique_isomers_k(
            [], n_linkers, n_fg_slots, fg_per_linker, active_per_linker=k
        )
        assert n == math.comb(fg_per_linker, k) ** n_linkers


def test_invalid_k_raises() -> None:
    with pytest.raises(ValueError):
        count_unique_isomers_k([], 2, 8, 4, active_per_linker=5)
    with pytest.raises(ValueError):
        count_unique_isomers_k([], 2, 8, 4, active_per_linker=-1)


if __name__ == "__main__":
    test_k_matches_brute_force_on_random_groups()
    test_k1_reproduces_count_unique_isomers()
    test_complement_and_edge_identities()
    test_identity_only_group_gives_binomial_power()
    test_invalid_k_raises()
    print("\nAll isomer-counting tests passed.")
