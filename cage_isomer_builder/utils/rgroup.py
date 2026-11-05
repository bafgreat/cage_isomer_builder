#!/usr/bin/env python3
"""
Symmetry-unique isomers with several *distinct* functional groups placed at
once (R1, R2, ..., written here as letters A, B, C, ...).

Two ways of specifying what goes on the linkers:

* ``groups`` (default mode): every linker carries the same set of groups.
  ``groups="AB"`` puts exactly one A and one B on every linker,
  ``groups="AAB"`` two A and one B, ``groups="A"`` reproduces the classic
  one-group-per-linker isomers.
* ``linker_patterns`` (mixed mode): linkers carry different sets, with a
  fixed number of linkers of each kind: ``{"A": 3, "B": 3}`` puts one A on
  three linkers and one B on the other three, ``{"AB": 2, "": 4}`` leaves
  four linkers bare. The values must sum to the number of linkers.

Mathematical basis
------------------
Let X be the FG anchor slots, partitioned into linkers of ``F`` slots each,
and G the symmetry group acting on X as permutations. A configuration is a
labelling f: X -> {0 (H), 1 (A), 2 (B), ...}. G acts by
(g.f)(g(x)) = f(x), and two configurations are the same isomer iff they lie
in the same G-orbit. Burnside's lemma:

    n_unique = (1/|G|) * sum_{g in G} |Fix(g)|

f is fixed by g iff f is constant on every cycle of g. Every symmetry maps
whole linkers onto whole linkers (checked, see :func:`prepare_group`), so g
also permutes the linkers, and a fixed f must
give every linker in one *linker cycle* the same set of groups. A site cycle
inside a linker cycle of length l visits each of its l linkers the same
number of times w = |cycle| / l. Hence

    |Fix(g)| = [ prod_P y_P^{N_P} ]  prod_{linker cycles Lam}
                   ( sum_P  W(Lam, P) * y_P^{l(Lam)} )

where P runs over the allowed linker patterns, N_P is how many linkers must
carry P, and W(Lam, P) is the number of ways to give each site cycle of Lam
one label so that each linker receives exactly the composition of P:

    W(Lam, P) = #{ labels for the cycles : sum of w over cycles labelled c
                   = n_c(P) for every group c }.

Both coefficients are extracted with small exact dynamic programmes over
Python integers (no overflow, no floating point). For the identity this
reduces to the multinomial count

    |Fix(e)| = (n_linkers! / prod_P N_P!) * prod_P ( F! / ((F-|P|)! prod_c n_c(P)!) )^{N_P}

Two assumptions are built in. Groups with different letters are never
interchanged by symmetry (A and B are chemically different), and the
substituents are achiral (improper operations such as mirrors count as
symmetries, exactly as for the single-group counts).

Enumeration keeps a configuration iff it is the lexicographically smallest
member of its orbit, compared first on the per-linker pattern sequence and
then on the per-slot labels. Because of that ordering, a whole linker-level
pattern assignment can be skipped when it is not itself minimal under the
induced linker permutations. A complete enumeration is always checked
against the Burnside count.
"""
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

import math
import os
from dataclasses import dataclass

import numpy as np
from numba import njit


# Slot layout

@dataclass(frozen=True)
class SlotLayout:
    """
    How the FG slots are grouped into linkers.

    Slots are numbered linker by linker: linker ``L`` owns the
    ``sizes[L]`` consecutive slots starting at ``offsets[L]``. Linkers of
    different chemical types (e.g. BDC and BPDC in one MOF) may have
    different numbers of slots; ``types[L]`` is the type index of linker
    ``L`` and ``type_names[t]`` its name. A symmetry operation may only send
    a linker onto a linker of the same type.

    Attributes
    ----------
    sizes : tuple of int
    types : tuple of int
    type_names : tuple of str

    Examples
    --------
    >>> from cage_isomer_builder.utils.rgroup import SlotLayout
    >>> layout = SlotLayout(sizes=(4, 4, 2), types=(0, 0, 1), type_names=("bdc", "pyz"))
    >>> layout.n_slots, layout.offsets, layout.slot_linker.tolist()
    (10, (0, 4, 8), [0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    """
    sizes: tuple
    types: tuple
    type_names: tuple

    @classmethod
    def uniform(cls, n_linkers, fg_per_linker, name="L1"):
        """``n_linkers`` linkers of one type with ``fg_per_linker`` slots.

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import SlotLayout
        >>> SlotLayout.uniform(6, 4).sizes
        (4, 4, 4, 4, 4, 4)
        """
        return cls((int(fg_per_linker),) * int(n_linkers), (0,) * int(n_linkers), (name,))

    @property
    def n_linkers(self):
        """
        Number of linkers that carry slots.

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import SlotLayout
        >>> layout = SlotLayout(sizes=(4, 4, 2), types=(0, 0, 1), type_names=("bdc", "pyz"))
        >>> layout.n_linkers
        3
        """
        return len(self.sizes)

    @property
    def n_slots(self):
        """
        Total number of slots.

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import SlotLayout
        >>> layout = SlotLayout(sizes=(4, 4, 2), types=(0, 0, 1), type_names=("bdc", "pyz"))
        >>> layout.n_slots
        10
        """
        return int(sum(self.sizes))

    @property
    def offsets(self):
        """
        Index of the first slot of each linker.

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import SlotLayout
        >>> layout = SlotLayout(sizes=(4, 4, 2), types=(0, 0, 1), type_names=("bdc", "pyz"))
        >>> layout.offsets
        (0, 4, 8)
        """
        return tuple(int(x) for x in np.concatenate([[0], np.cumsum(self.sizes)[:-1]]))

    @property
    def slot_linker(self):
        """Linker index of every slot.

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import SlotLayout
        >>> layout = SlotLayout(sizes=(4, 4, 2), types=(0, 0, 1), type_names=("bdc", "pyz"))
        >>> layout.slot_linker.tolist()
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
        """
        return np.repeat(np.arange(self.n_linkers), self.sizes)

    @property
    def uniform_size(self):
        """The common number of slots per linker, or None if they differ.

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import SlotLayout
        >>> layout = SlotLayout(sizes=(4, 4, 2), types=(0, 0, 1), type_names=("bdc", "pyz"))
        >>> print(layout.uniform_size), SlotLayout.uniform(3, 4).uniform_size
        None
        (None, 4)
        """
        return self.sizes[0] if len(set(self.sizes)) == 1 else None

    @property
    def n_types(self):
        """
        Number of linker types.

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import SlotLayout
        >>> layout = SlotLayout(sizes=(4, 4, 2), types=(0, 0, 1), type_names=("bdc", "pyz"))
        >>> layout.n_types
        2
        """
        return len(self.type_names)

    def linkers_of_type(self, t):
        """
        Indices of the linkers of type ``t``.

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import SlotLayout
        >>> layout = SlotLayout(sizes=(4, 4, 2), types=(0, 0, 1), type_names=("bdc", "pyz"))
        >>> layout.linkers_of_type(0), layout.linkers_of_type(1)
        ([0, 1], [2])
        """
        return [L for L, tt in enumerate(self.types) if tt == t]

    def type_size(self, t):
        """
        Number of slots of each linker of type ``t`` (ValueError if they differ).

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import SlotLayout
        >>> layout = SlotLayout(sizes=(4, 4, 2), types=(0, 0, 1), type_names=("bdc", "pyz"))
        >>> layout.type_size(0), layout.type_size(1)
        (4, 2)
        """
        sizes = {self.sizes[L] for L in self.linkers_of_type(t)}
        if len(sizes) != 1:
            raise ValueError(f"linker type {self.type_names[t]!r} has linkers with "
                             f"different slot counts {sorted(sizes)}.")
        return sizes.pop()


def as_layout(layout, n_slots=None):
    """A :class:`SlotLayout` from a layout or a uniform ``fg_per_linker``.

    Examples
    --------
    >>> from cage_isomer_builder.utils.rgroup import as_layout
    >>> as_layout(4, n_slots=24).sizes                  # six linkers of four slots
    (4, 4, 4, 4, 4, 4)
    """
    if isinstance(layout, SlotLayout):
        if n_slots is not None and layout.n_slots != n_slots:
            raise ValueError(f"layout has {layout.n_slots} slots, expected {n_slots}.")
        return layout
    F = int(layout)
    if n_slots is None:
        raise ValueError("n_slots is needed with a uniform fg_per_linker.")
    if n_slots % F:
        raise ValueError(f"{n_slots} slots is not a whole number of linkers of {F} slots.")
    return SlotLayout.uniform(n_slots // F, F)


# Specification

def _split_groups(pattern):
    """
    'AB' -> ('A', 'B');  ['R1', 'R2'] -> ('R1', 'R2');  '' -> ().

    A string is read one capital letter per group, so it may only contain
    capital letters: ``"NH2"`` would otherwise mean three groups N, H and
    2. Named groups are given as a list or tuple (``["NH2"]``,
    ``("NH2", "OH")``).
    """
    if isinstance(pattern, str):
        if pattern and not (pattern.isascii() and pattern.isalpha() and pattern.isupper()):
            raise ValueError(
                f"pattern {pattern!r}: a string is read one capital letter per group "
                f"(e.g. 'AB' is one A and one B). Give named groups as a list or tuple, "
                f"e.g. [{pattern!r}] or ({pattern!r},)."
            )
        return tuple(pattern)
    return tuple(str(p) for p in pattern)


@dataclass(frozen=True)
class RGroupSpec:
    """
    Normalised description of what goes on the linkers.

    Attributes
    ----------
    labels : tuple of str
        Group names; label integer ``i + 1`` means ``labels[i]`` (0 is H).
    patterns : tuple of tuple of int
        Per-pattern composition: ``patterns[p][c]`` slots carry label c+1.
    pattern_names : tuple of str
    pattern_counts : tuple of int
        Number of linkers that must carry each pattern.
    pattern_types : tuple of int
        Linker type each pattern belongs to.
    layout : SlotLayout

    Examples
    --------
    >>> from cage_isomer_builder.utils.rgroup import make_spec
    >>> spec = make_spec(6, 4, linker_patterns={"A": 3, "B": 3})
    >>> spec.labels, spec.patterns, spec.pattern_counts
    (('A', 'B'), ((1, 0), (0, 1)), (3, 3))
    """
    labels: tuple
    patterns: tuple
    pattern_names: tuple
    pattern_counts: tuple
    pattern_types: tuple
    layout: SlotLayout

    @property
    def n_linkers(self):
        """
        Number of linkers.

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import make_spec
        >>> make_spec(6, 4, groups="A").n_linkers
        6
        """
        return self.layout.n_linkers

    @property
    def fg_per_linker(self):
        """
        Slots per linker (ValueError if linkers differ; use ``layout``).

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import make_spec
        >>> make_spec(6, 4, groups="A").fg_per_linker
        4
        """
        size = self.layout.uniform_size
        if size is None:
            raise ValueError("linkers have different numbers of slots; use spec.layout.")
        return size

    @property
    def is_uniform(self):
        """
        True if every linker carries the same pattern.

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import make_spec
        >>> make_spec(6, 4, groups="AB").is_uniform, make_spec(6, 4, linker_patterns={"A": 3, "": 3}).is_uniform
        (True, False)
        """
        return len(self.patterns) == 1


def _per_type(value, layout, what):
    """Spread a per-type argument: a dict keyed by type name, or one value
    for every type."""
    names = layout.type_names
    if isinstance(value, dict) and value and set(value) <= set(names) and all(
            isinstance(v, (dict, str, list, tuple)) for v in value.values()):
        missing = set(names) - set(value)
        if missing:
            raise ValueError(f"{what} gives no entry for linker type(s) {sorted(missing)}.")
        return [value[n] for n in names]
    return [value] * len(names)


def make_spec(layout, fg_per_linker=None, groups="AB", linker_patterns=None):
    """
    Build an :class:`RGroupSpec`.

    Parameters
    ----------
    layout : SlotLayout or int
        The slot layout, or (single linker type) the number of linkers, with
        ``fg_per_linker`` slots each.
    groups : str, list, or dict
        Groups carried by every linker (default mode). With several linker
        types, a dict ``{type name: groups}`` gives each type its own.
    linker_patterns : dict, optional
        Mixed mode (takes precedence): ``{pattern: number of linkers}``. With
        several linker types, ``{type name: {pattern: number}}``.

    Raises
    ------
    ValueError
        If a pattern needs more slots than its linker has, two patterns of
        one type are the same set of groups written differently, or the
        linker counts of a type don't add up to its number of linkers.

    Examples
    --------
    >>> from cage_isomer_builder.utils.rgroup import SlotLayout
    >>> layout = SlotLayout(sizes=(4, 4, 2), types=(0, 0, 1), type_names=("bdc", "pyz"))
    >>> from cage_isomer_builder.utils.rgroup import make_spec
    >>> make_spec(6, 4, groups="AAB").patterns          # two A and one B on every linker
    ((2, 1),)
    >>> spec = make_spec(layout, groups={"bdc": "AB", "pyz": ""})   # per linker type
    >>> spec.pattern_names, spec.pattern_types
    (('AB', ''), (0, 1))
    """
    if not isinstance(layout, SlotLayout):
        if fg_per_linker is None:
            raise ValueError("give a SlotLayout, or n_linkers and fg_per_linker.")
        layout = SlotLayout.uniform(int(layout), int(fg_per_linker))
    raw = []                                  # (type, pattern tuple, count)
    if linker_patterns is None:
        for t, g in enumerate(_per_type(groups, layout, "groups")):
            raw.append((t, _split_groups(g), len(layout.linkers_of_type(t))))
    else:
        if layout.n_types > 1:
            if not (isinstance(linker_patterns, dict)
                    and set(linker_patterns) == set(layout.type_names)):
                raise ValueError(
                    f"with several linker types {list(layout.type_names)}, give "
                    "linker_patterns as {type name: {pattern: number of linkers}}.")
            per_type = [linker_patterns[n] for n in layout.type_names]
        elif (isinstance(linker_patterns, dict) and set(linker_patterns) == {layout.type_names[0]}
              and isinstance(linker_patterns[layout.type_names[0]], dict)):
            per_type = [linker_patterns[layout.type_names[0]]]
        else:
            per_type = [linker_patterns]
        for t, patterns in enumerate(per_type):
            entries = [(_split_groups(p), int(n)) for p, n in dict(patterns).items()]
            if any(n < 0 for _, n in entries):
                raise ValueError("linker_patterns counts must be non-negative.")
            n_t = len(layout.linkers_of_type(t))
            total = sum(n for _, n in entries)
            if total != n_t:
                where = f" of type {layout.type_names[t]!r}" if layout.n_types > 1 else ""
                raise ValueError(
                    f"linker_patterns counts sum to {total}, but the structure has "
                    f"{n_t} linkers{where}.")
            raw += [(t, p, n) for p, n in entries if n > 0]

    labels = []
    for _, pattern, _ in raw:
        for name in pattern:
            if name not in labels:
                labels.append(name)
    labels = tuple(labels)

    patterns, names, counts, types, seen = [], [], [], [], {}
    for t, pattern, n in raw:
        size = layout.type_size(t)
        if len(pattern) > size:
            raise ValueError(
                f"pattern {''.join(pattern)!r} needs {len(pattern)} slots but each "
                f"linker only has fg_per_linker={size}.")
        comp = tuple(pattern.count(name) for name in labels)
        if (t, comp) in seen:
            raise ValueError(
                f"patterns {seen[(t, comp)]!r} and {''.join(pattern)!r} are the same "
                "set of groups; merge their counts.")
        seen[(t, comp)] = ''.join(pattern)
        patterns.append(comp)
        names.append(''.join(pattern))
        counts.append(n)
        types.append(t)
    # compositions were built while labels grew: pad them to full length
    patterns = [tuple(p) + (0,) * (len(labels) - len(p)) for p in patterns]
    return RGroupSpec(labels, tuple(patterns), tuple(names), tuple(counts), tuple(types), layout)


# Symmetry group

@dataclass(frozen=True)
class SlotGroup:
    """
    A validated symmetry group acting on FG slots.

    Attributes
    ----------
    perms : np.ndarray, shape (|G|, n_slots)
        ``perms[g, x]`` is the image of slot x under g. Row 0 is identity.
    inverse : np.ndarray, shape (|G|, n_slots)
    linker_perms : np.ndarray, shape (|G|, n_linkers)
        Induced permutation of whole linkers.
    linker_inverse : np.ndarray, shape (|G|, n_linkers)
    layout : SlotLayout

    Examples
    --------
    >>> from cage_isomer_builder.utils import rgroup
    >>> # two linkers with two slots each (slots 0,1 | 2,3); one symmetry swaps the linkers
    >>> group = rgroup.prepare_group([[2, 3, 0, 1]], n_slots=4, fg_per_linker=2)
    >>> group.order, group.linker_perms.tolist()
    (2, [[0, 1], [1, 0]])
    """
    perms: np.ndarray
    inverse: np.ndarray
    linker_perms: np.ndarray
    linker_inverse: np.ndarray
    layout: SlotLayout = None

    @property
    def order(self):
        """
        Number of symmetry operations, including the identity.

        Examples
        --------
        >>> from cage_isomer_builder.utils import rgroup
        >>> # two linkers with two slots each (slots 0,1 | 2,3); one symmetry swaps the linkers
        >>> group = rgroup.prepare_group([[2, 3, 0, 1]], n_slots=4, fg_per_linker=2)
        >>> group.order
        2
        """
        return len(self.perms)


def prepare_group(transformation_library, n_slots, fg_per_linker):
    """
    Turn a transformation library into a validated :class:`SlotGroup`.

    Takes the operation-major layout (a list of permutations, as taken by
    ``count_unique_isomers``). The slot-major output of
    ``CageBuilder._get_transformation_library`` must be transposed first
    with ``list(zip(*tl))``; the two layouts cannot be told apart when the
    number of operations equals the number of slots. The identity
    is added and duplicates removed.

    Every condition Burnside's lemma relies on is checked rather than
    assumed: each operation is a bijection of the slots, maps whole linkers
    onto whole linkers of the same type, and the set is closed under
    composition.

    Parameters
    ----------
    fg_per_linker : int or SlotLayout
        Slots per linker, or the layout of linkers with different numbers of
        slots (several linker types).

    Raises
    ------
    ValueError
        If any of those checks fails. This always means the symmetry
        operations don't match the structure (e.g. a mis-oriented cage or a
        too-loose tolerance), and any count derived from them would be wrong.

    Examples
    --------
    >>> from cage_isomer_builder.utils import rgroup
    >>> # two linkers with two slots each (slots 0,1 | 2,3); one symmetry swaps the linkers
    >>> group = rgroup.prepare_group([[2, 3, 0, 1]], n_slots=4, fg_per_linker=2)
    >>> group.perms.tolist()                       # identity added
    [[0, 1, 2, 3], [2, 3, 0, 1]]
    >>> rgroup.prepare_group([[1, 2, 3, 0]], 4, 2)   # moves slot 1 onto the other linker
    Traceback (most recent call last):
    ...
    ValueError: a symmetry operation splits the slots of one linker across several linkers; per-linker compositions are then not symmetry invariant.
    """
    layout = as_layout(fg_per_linker, n_slots)
    ops = [list(op) for op in transformation_library]

    identity = tuple(range(n_slots))
    unique = {identity: None}
    for op in ops:
        if len(op) != n_slots:
            raise ValueError(f"operation has {len(op)} entries, expected {n_slots}.")
        unique.setdefault(tuple(int(i) for i in op), None)
    perms = np.array(list(unique), dtype=np.int64)

    if not all(len(np.unique(row)) == n_slots for row in perms):
        raise ValueError("a symmetry operation is not a bijection of the FG slots.")

    slot_linker = layout.slot_linker
    images = slot_linker[perms]                          # (G, n_slots)
    offsets = np.array(layout.offsets)
    starts = images[:, offsets]
    if not np.all(images == np.repeat(starts, layout.sizes, axis=1)):
        raise ValueError(
            "a symmetry operation splits the slots of one linker across several "
            "linkers; per-linker compositions are then not symmetry invariant."
        )
    linker_perms = starts
    types = np.array(layout.types)
    if not np.all(types[linker_perms] == types[None, :]):
        raise ValueError("a symmetry operation sends a linker onto a linker of a "
                         "different type.")

    rows = {row.tobytes() for row in perms}
    for a in perms:
        composed = a[perms]          # composed[k, x] = a(perms[k](x))
        for row in composed:
            if row.tobytes() not in rows:
                raise ValueError(
                    "the symmetry operations are not closed under composition, "
                    "so they do not form a group; Burnside counts would be wrong."
                )

    inverse = np.argsort(perms, axis=1)
    linker_inverse = np.argsort(linker_perms, axis=1)
    return SlotGroup(perms, inverse, linker_perms, linker_inverse, layout)


def _group_and_layout(transformation_library, n_slots, fg_per_linker):
    if isinstance(transformation_library, SlotGroup):
        group = transformation_library
        layout = group.layout if group.layout is not None else as_layout(fg_per_linker, n_slots)
        return group, layout
    group = prepare_group(transformation_library, n_slots, fg_per_linker)
    return group, group.layout


# Counting

def _linker_cycles(perm, linker_perm, layout):
    """
    For one operation: list of (cycle length l, sorted site-cycle weights w,
    linker type) per linker cycle. The weights of a linker cycle sum to the
    number of slots of its linkers.
    """
    if not isinstance(layout, SlotLayout):
        layout = SlotLayout.uniform(len(linker_perm), layout)
    n_linkers = len(linker_perm)
    offsets, sizes = layout.offsets, layout.sizes
    seen = [False] * n_linkers
    out = []
    for first in range(n_linkers):
        if seen[first]:
            continue
        length, linker = 0, first
        while not seen[linker]:
            seen[linker] = True
            linker = linker_perm[linker]
            length += 1

        lo, hi = offsets[first], offsets[first] + sizes[first]
        visited = set()
        weights = []
        for start in range(lo, hi):
            if start in visited:
                continue
            on_first, j = 0, start
            while True:
                if lo <= j < hi:
                    visited.add(j)
                    on_first += 1
                j = perm[j]
                if j == start:
                    break
            weights.append(on_first)
        out.append((length, tuple(sorted(weights)), layout.types[first]))
    return out


def _cycle_label_ways(weights, comp):
    """W(Lam, P): label the site cycles so the linker gets exactly comp."""
    states = {tuple(0 for _ in comp): 1}
    for w in weights:
        nxt = {}
        for state, ways in states.items():
            nxt[state] = nxt.get(state, 0) + ways          # cycle stays H
            for c, need in enumerate(comp):
                if state[c] + w <= need:
                    new = state[:c] + (state[c] + w,) + state[c + 1:]
                    nxt[new] = nxt.get(new, 0) + ways
        states = nxt
    return states.get(tuple(comp), 0)


def _fixed_count(perm, linker_perm, spec, cache):
    """|Fix(g)| for one operation (see the module docstring)."""
    target = spec.pattern_counts
    states = {tuple(0 for _ in target): 1}
    for length, weights, ltype in _linker_cycles(perm, linker_perm, spec.layout):
        options = []
        for p, comp in enumerate(spec.patterns):
            if spec.pattern_types[p] != ltype:
                continue
            key = (weights, comp)
            if key not in cache:
                cache[key] = _cycle_label_ways(weights, comp)
            if cache[key]:
                options.append((p, cache[key]))
        if not options:
            return 0
        nxt = {}
        for state, ways in states.items():
            for p, w in options:
                if state[p] + length <= target[p]:
                    new = state[:p] + (state[p] + length,) + state[p + 1:]
                    nxt[new] = nxt.get(new, 0) + ways * w
        states = nxt
        if not states:
            return 0
    return states.get(tuple(target), 0)


def identity_fixed_count(spec):
    """|Fix(e)|: the raw number of configurations before symmetry.

    Examples
    --------
    >>> from cage_isomer_builder.utils.rgroup import identity_fixed_count, make_spec
    >>> identity_fixed_count(make_spec(6, 4, groups="A"))        # 4**6 raw placements
    4096
    >>> identity_fixed_count(make_spec(2, 4, groups="AB"))       # (4 * 3)**2
    144
    """
    total = 1
    layout = spec.layout
    for t in range(layout.n_types):
        n_t = len(layout.linkers_of_type(t))
        if n_t == 0:
            continue
        F = layout.type_size(t)
        block = math.factorial(n_t)
        for p, n in enumerate(spec.pattern_counts):
            if spec.pattern_types[p] != t:
                continue
            block //= math.factorial(n)
            local = math.factorial(F) // math.factorial(F - sum(spec.patterns[p]))
            for k in spec.patterns[p]:
                local //= math.factorial(k)
            block *= local ** n
        total *= block
    return total


def count_rgroup_isomers(transformation_library, n_slots, fg_per_linker=4,
                         groups="AB", linker_patterns=None):
    """
    Exact number of symmetry-unique isomers with several distinct groups,
    via Burnside's lemma (see the module docstring for the formula).

    Parameters
    ----------
    transformation_library : list or SlotGroup
        Symmetry operations on the FG slots, operation-major (see
        :func:`prepare_group`), or an already prepared ``SlotGroup``.
    n_slots : int
        Total number of FG slots.
    fg_per_linker : int or SlotLayout, default 4
        Slots per linker, or the layout for several linker types.
    groups : str, sequence of str, or dict, default "AB"
        Groups carried by every linker (default mode); per linker type with
        a dict (see :func:`make_spec`).
    linker_patterns : dict, optional
        Mixed mode (see :func:`make_spec`). Overrides ``groups``.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        Invalid specification, invalid symmetry group (see
        :func:`prepare_group`), or a non-integer Burnside average.

    Examples
    --------
    >>> from cage_isomer_builder.utils import rgroup
    >>> # two linkers with two slots each (slots 0,1 | 2,3); one symmetry swaps the linkers
    >>> group = rgroup.prepare_group([[2, 3, 0, 1]], n_slots=4, fg_per_linker=2)
    >>> rgroup.count_rgroup_isomers(group, 4, 2, groups="A")     # (2*2 raw + 2 fixed) / 2
    3
    >>> rgroup.count_rgroup_isomers(group, 4, 2, linker_patterns={"A": 1, "": 1})
    2
    """
    group, layout = _group_and_layout(transformation_library, n_slots, fg_per_linker)
    spec = make_spec(layout, None, groups, linker_patterns)

    cache = {}
    total = identity_fixed_count(spec)
    for perm, linker_perm in zip(group.perms[1:], group.linker_perms[1:]):
        total += _fixed_count(perm.tolist(), linker_perm.tolist(), spec, cache)

    n_unique, remainder = divmod(total, group.order)
    if remainder:
        raise ValueError(
            f"Burnside average is not a whole number ({total}/{group.order})."
        )
    return n_unique


# Enumeration

def _local_arrangements(comp, fg_per_linker):
    """All distinct ways to place one pattern on a linker's slots, sorted.
    Generated directly as distinct multiset permutations; listing all F!
    permutations first is infeasible for large linkers (16! is about 2e13)."""
    labels = [0] * (fg_per_linker - sum(comp))
    for c, k in enumerate(comp):
        labels += [c + 1] * k
    return [tuple(p) for p in _multiset_permutations(labels)]


def _scan_py(pattern_of_linker, tables, n_tables, inverse, linker_inverse,
             digits, out, offsets, sizes):
    """
    Walk configurations for one fixed per-linker pattern assignment, in
    odometer order starting at ``digits`` (one local-arrangement index per
    linker, last linker fastest), storing the canonical ones in ``out``.
    Linker ``L``'s slots are ``offsets[L] .. offsets[L] + sizes[L] - 1``.

    Returns (number stored, finished). When ``out`` fills up first,
    ``digits`` is left at the next configuration to examine.

    Compiled with numba as ``_scan``. The plain Python version is kept so
    the tests can check the compiled one against it.
    """
    n_linkers = pattern_of_linker.shape[0]
    n_slots = inverse.shape[1]
    n_ops = inverse.shape[0]
    f = np.empty(n_slots, dtype=np.int8)
    n_found = 0
    while True:
        for linker in range(n_linkers):
            row = tables[pattern_of_linker[linker], digits[linker]]
            for i in range(sizes[linker]):
                f[offsets[linker] + i] = row[i]

        canonical = True
        for g in range(n_ops):
            cmp = 0
            for linker in range(n_linkers):
                a = pattern_of_linker[linker_inverse[g, linker]]
                b = pattern_of_linker[linker]
                if a != b:
                    cmp = -1 if a < b else 1
                    break
            if cmp == 0:
                for x in range(n_slots):
                    a = f[inverse[g, x]]
                    b = f[x]
                    if a != b:
                        cmp = -1 if a < b else 1
                        break
            if cmp < 0:
                canonical = False
                break
        if canonical:
            for linker in range(n_linkers):
                out[n_found, linker] = digits[linker]
            n_found += 1

        linker = n_linkers - 1
        while linker >= 0:
            digits[linker] += 1
            if digits[linker] < n_tables[pattern_of_linker[linker]]:
                break
            digits[linker] = 0
            linker -= 1
        if linker < 0:
            return n_found, True
        if n_found == out.shape[0]:
            return n_found, False


_scan = njit(cache=True)(_scan_py)


def _multiset_permutations(items):
    """Distinct permutations of a sorted list, in lexicographic order."""
    a = sorted(items)
    n = len(a)
    while True:
        yield list(a)
        i = n - 2
        while i >= 0 and a[i] >= a[i + 1]:
            i -= 1
        if i < 0:
            return
        j = n - 1
        while a[j] <= a[i]:
            j -= 1
        a[i], a[j] = a[j], a[i]
        a[i + 1:] = reversed(a[i + 1:])


def _pattern_assignments(spec):
    """Every assignment of patterns to linkers respecting linker types and
    pattern counts, as a list of pattern ids per linker."""
    layout = spec.layout
    per_type = []
    for t in range(layout.n_types):
        positions = layout.linkers_of_type(t)
        ids = [p for p, n in enumerate(spec.pattern_counts)
               if spec.pattern_types[p] == t for _ in range(n)]
        per_type.append((positions, ids))

    def rec(t, current):
        if t == len(per_type):
            yield list(current)
            return
        positions, ids = per_type[t]
        if not positions:
            yield from rec(t + 1, current)
            return
        for perm in _multiset_permutations(ids):
            for pos, p in zip(positions, perm):
                current[pos] = p
            yield from rec(t + 1, current)

    yield from rec(0, [0] * layout.n_linkers)


def _is_minimal_linker_assignment(pattern_of_linker, linker_inverse):
    images = pattern_of_linker[linker_inverse]          # (|G|, n_linkers)
    diff = images != pattern_of_linker
    has_diff = diff.any(axis=1)
    first = diff.argmax(axis=1)
    rows = np.nonzero(has_diff)[0]
    return not np.any(images[rows, first[rows]] < pattern_of_linker[first[rows]])


@dataclass(frozen=True)
class RGroupIsomer:
    """
    One symmetry-unique isomer.

    Attributes
    ----------
    site_labels : tuple of int
        Label per FG slot: 0 = H, i = ``labels[i - 1]``.
    labels : tuple of str
    linker_patterns : tuple of str
        Pattern carried by each linker.

    Examples
    --------
    >>> from cage_isomer_builder.utils.rgroup import RGroupIsomer
    >>> iso = RGroupIsomer(site_labels=(1, 0, 2, 0), labels=("A", "B"), linker_patterns=("A", "B"))
    >>> iso.to_dict(), iso.name
    ({'A': [0], 'B': [2]}, 'A-0_B-2')
    """
    site_labels: tuple
    labels: tuple
    linker_patterns: tuple

    def slots(self, label):
        """FG slot indices carrying group ``label`` (e.g. ``"A"``).

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import RGroupIsomer
        >>> RGroupIsomer((1, 1, 0, 2), ("A", "B"), ("AA", "B")).slots("A")
        [0, 1]
        """
        code = self.labels.index(label) + 1
        return [i for i, v in enumerate(self.site_labels) if v == code]

    def to_dict(self):
        """``{"A": [slots], "B": [slots], ...}``

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import RGroupIsomer
        >>> RGroupIsomer((0, 1, 2, 0), ("A", "B"), ("A", "B")).to_dict()
        {'A': [1], 'B': [2]}
        """
        return {label: self.slots(label) for label in self.labels}

    @property
    def name(self):
        """File-name stem, e.g. ``A-0-5-9_B-1-6-10``.

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import RGroupIsomer
        >>> RGroupIsomer((0, 1, 2, 0), ("A", "B"), ("A", "B")).name
        'A-1_B-2'
        """
        parts = []
        for label, slots in self.to_dict().items():
            parts.append('-'.join([label] + [str(s) for s in slots]))
        return '_'.join(parts) if parts else 'bare'


def iter_rgroup_isomers(transformation_library, n_slots, fg_per_linker=4,
                        groups="AB", linker_patterns=None, limit=None,
                        batch_size=4096):
    """
    Generator over symmetry-unique isomers with several distinct groups.

    One representative per orbit: the lexicographically smallest
    configuration, compared first on the pattern sequence over the linkers
    and then on the slot labels. Memory use is independent of the size of
    the configuration space. The inner loop is compiled with numba.

    Parameters
    ----------
    transformation_library, n_slots, fg_per_linker, groups, linker_patterns
        As for :func:`count_rgroup_isomers`.
    limit : int, optional
        Stop after this many isomers.
    batch_size : int, default 4096
        How many isomers the compiled loop collects per call.

    Yields
    ------
    RGroupIsomer

    Examples
    --------
    >>> from cage_isomer_builder.utils import rgroup
    >>> # two linkers with two slots each (slots 0,1 | 2,3); one symmetry swaps the linkers
    >>> group = rgroup.prepare_group([[2, 3, 0, 1]], n_slots=4, fg_per_linker=2)
    >>> # one representative per orbit: (0, 1, 1, 0) also stands for (1, 0, 0, 1)
    >>> [iso.site_labels for iso in rgroup.iter_rgroup_isomers(group, 4, 2, groups="A")]
    [(0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 1, 0)]
    """
    group, layout = _group_and_layout(transformation_library, n_slots, fg_per_linker)
    spec = make_spec(layout, None, groups, linker_patterns)
    n_linkers = layout.n_linkers

    arrangements = [_local_arrangements(comp, layout.type_size(spec.pattern_types[p]))
                    for p, comp in enumerate(spec.patterns)]
    max_k = max(len(a) for a in arrangements)
    max_f = max(layout.sizes)
    tables = np.zeros((len(arrangements), max_k, max_f), dtype=np.int8)
    for p, arr in enumerate(arrangements):
        width = len(arr[0])
        tables[p, :len(arr), :width] = arr
    n_tables = np.array([len(a) for a in arrangements], dtype=np.int64)
    offsets = np.array(layout.offsets, dtype=np.int64)
    sizes = np.array(layout.sizes, dtype=np.int64)

    inverse = np.ascontiguousarray(group.inverse)
    linker_inverse = np.ascontiguousarray(group.linker_inverse)
    out = np.empty((batch_size, n_linkers), dtype=np.int64)

    n_yielded = 0
    for assignment in _pattern_assignments(spec):
        pattern_of_linker = np.array(assignment, dtype=np.int64)
        if not _is_minimal_linker_assignment(pattern_of_linker, linker_inverse):
            continue
        names = tuple(spec.pattern_names[p] for p in assignment)
        digits = np.zeros(n_linkers, dtype=np.int64)
        finished = False
        while not finished:
            n_found, finished = _scan(
                pattern_of_linker, tables, n_tables, inverse, linker_inverse,
                digits, out, offsets, sizes,
            )
            for row in out[:n_found]:
                site_labels = np.concatenate([
                    tables[pattern_of_linker[L], row[L], :sizes[L]] for L in range(n_linkers)
                ])
                yield RGroupIsomer(
                    tuple(int(v) for v in site_labels), spec.labels, names
                )
                n_yielded += 1
                if limit is not None and n_yielded >= limit:
                    return


def enumerate_rgroup_isomers(transformation_library, n_slots, fg_per_linker=4,
                             groups="AB", linker_patterns=None, limit=None):
    """
    List of symmetry-unique isomers (see :func:`iter_rgroup_isomers`).

    A complete enumeration (``limit=None``) is checked against the
    independent Burnside count; a mismatch raises ``RuntimeError``.

    Examples
    --------
    >>> from cage_isomer_builder.utils import rgroup
    >>> # two linkers with two slots each (slots 0,1 | 2,3); one symmetry swaps the linkers
    >>> group = rgroup.prepare_group([[2, 3, 0, 1]], n_slots=4, fg_per_linker=2)
    >>> isomers = rgroup.enumerate_rgroup_isomers(group, 4, 2, groups="AB")
    >>> len(isomers), isomers[0].to_dict()
    (3, {'A': [0, 2], 'B': [1, 3]})
    """
    group, _ = _group_and_layout(transformation_library, n_slots, fg_per_linker)
    isomers = list(iter_rgroup_isomers(
        group, n_slots, fg_per_linker, groups, linker_patterns, limit=limit
    ))
    if limit is None:
        expected = count_rgroup_isomers(group, n_slots, fg_per_linker, groups, linker_patterns)
        if len(isomers) != expected:
            raise RuntimeError(
                f"enumeration found {len(isomers)} isomers but Burnside's lemma "
                f"gives {expected} - please report this as a bug."
            )
    return isomers


# Output

def build_rgroup_isomer_atoms(cage, fg_anchors, isomer, fg_anchor_indices,
                              fg_anchor_h_positions=None):
    """
    The isomer as ``ase.Atoms``. Every active slot is an R site, an ``X``
    atom whose per-atom ``"rgroup"`` entry is its group number (1 = R1 =
    first label, 2 = R2, ...; 0 for every other atom), see
    :mod:`cage_isomer_builder.utils.sites`. Inactive slots go back to H at
    their original position (or 1.09 Angstrom along the C-site direction if
    that is unknown). ``atoms.info["rgroup_labels"]`` records the group
    names, e.g. ``"A B"``.

    Examples
    --------
    >>> import numpy as np
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.sites import mark_site
    >>> benzene = molecule("C6H6")                    # C 0-5, H 6-11 (H i+6 on C i)
    >>> h_positions = benzene.positions[6:].copy()
    >>> for h in range(6, 12):
    ...     mark_site(benzene, h, 1)                  # every H a slot, as functionalise() does
    >>> slots = list(range(6, 12))
    >>> anchors = benzene[slots]
    >>> from cage_isomer_builder.utils.rgroup import RGroupIsomer, build_rgroup_isomer_atoms
    >>> iso = RGroupIsomer((1, 0, 0, 2, 0, 0), ("NH2", "OH"), ("NH2", "OH"))
    >>> atoms = build_rgroup_isomer_atoms(benzene, anchors, iso, slots, h_positions)
    >>> atoms.get_chemical_symbols()[6:], atoms.info["rgroup_labels"]
    (['X', 'H', 'H', 'X', 'H', 'H'], 'NH2 OH')
    """
    from cage_isomer_builder.utils.sites import clear_site, mark_site, restore_hydrogen

    atoms = cage.copy()
    atoms.set_array('rgroup', np.zeros(len(atoms), dtype=int))
    for slot, i_atom in enumerate(fg_anchor_indices):
        label = isomer.site_labels[slot]
        if label:
            mark_site(atoms, i_atom, int(label), fg_anchors[slot].position)
        elif fg_anchor_h_positions is not None:
            clear_site(atoms, i_atom, 'H', fg_anchor_h_positions[slot])
        else:
            restore_hydrogen(atoms, i_atom)
    atoms.info['rgroup_labels'] = ' '.join(isomer.labels)
    atoms.info['isomer'] = isomer.name
    return atoms


def write_rgroup_isomer_file(cage, fg_anchors, isomer, fg_anchor_indices,
                             output_path, fg_anchor_h_positions=None):
    """
    Write the isomer to ``<output_path>/<isomer.name>.xyz`` in extended XYZ
    format, which keeps the ``rgroup`` labels and, for MOFs, the unit cell.
    Read back with ``ase.io.read``; ``atoms.arrays["rgroup"]`` has the labels.

    Returns
    -------
    str
        Path of the written file.

    Examples
    --------
    >>> import numpy as np
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.sites import mark_site
    >>> benzene = molecule("C6H6")                    # C 0-5, H 6-11 (H i+6 on C i)
    >>> h_positions = benzene.positions[6:].copy()
    >>> for h in range(6, 12):
    ...     mark_site(benzene, h, 1)                  # every H a slot, as functionalise() does
    >>> slots = list(range(6, 12))
    >>> anchors = benzene[slots]
    >>> import tempfile
    >>> from cage_isomer_builder.utils.rgroup import RGroupIsomer, write_rgroup_isomer_file
    >>> from cage_isomer_builder.utils.sites import read_isomer
    >>> iso = RGroupIsomer((1, 0, 0, 2, 0, 0), ("NH2", "OH"), ("NH2", "OH"))
    >>> path = write_rgroup_isomer_file(benzene, anchors, iso, slots, tempfile.mkdtemp(), h_positions)
    >>> path.endswith("NH2-0_OH-3.xyz"), read_isomer(path).info["r_sites"]
    (True, {6: 'R1', 9: 'R2'})
    """
    from cage_isomer_builder.utils.sites import write_isomer

    atoms = build_rgroup_isomer_atoms(
        cage, fg_anchors, isomer, fg_anchor_indices, fg_anchor_h_positions
    )
    return write_isomer(os.path.join(output_path, f"{isomer.name}.xyz"), atoms)


_RDKIT_BOND = {1: 'SINGLE', 2: 'DOUBLE', 3: 'TRIPLE', 4: 'QUADRUPLE'}


def rgroup_atoms_to_rdkit(atoms, bond_matrix):
    """
    RDKit molecule for a finite isomer, with every active site as an RDKit
    R-group dummy atom (``R#`` in a MOL/SDF file, ``[1*:1]`` in SMILES), the
    R1, R2, ... notation RDKit uses for R-group decomposition.

    Bonds come from ``bond_matrix`` (e.g. ``CageBuilder.get_bond_matrix``),
    not from distance perception, which is unreliable on metal clusters.
    The molecule is not sanitised. Periodic structures should stay in
    extended XYZ, since SDF cannot store a unit cell.

    Parameters
    ----------
    atoms : ase.Atoms
        Output of :func:`build_rgroup_isomer_atoms`.
    bond_matrix : np.ndarray, shape (len(atoms), len(atoms))
        Bond orders. 1.5 is written as an aromatic bond; other orders
        outside 1-4 (e.g. fractional metal-ligand codes) as single bonds.

    Examples
    --------
    >>> import numpy as np
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.sites import mark_site
    >>> benzene = molecule("C6H6")                    # C 0-5, H 6-11 (H i+6 on C i)
    >>> h_positions = benzene.positions[6:].copy()
    >>> for h in range(6, 12):
    ...     mark_site(benzene, h, 1)                  # every H a slot, as functionalise() does
    >>> slots = list(range(6, 12))
    >>> anchors = benzene[slots]
    >>> from rdkit import Chem
    >>> from cage_isomer_builder.utils.rgroup import RGroupIsomer, build_rgroup_isomer_atoms, rgroup_atoms_to_rdkit
    >>> iso = RGroupIsomer((1, 0, 0, 2, 0, 0), ("NH2", "OH"), ("NH2", "OH"))
    >>> atoms = build_rgroup_isomer_atoms(benzene, anchors, iso, slots, h_positions)
    >>> from cage_isomer_builder.utils.features import bond_graph
    >>> bonds = np.zeros((12, 12))
    >>> for i, nbrs in bond_graph(atoms).items():
    ...     for j in nbrs:
    ...         bonds[i, j] = 1.5 if max(i, j) < 6 else 1.0
    >>> block = Chem.MolToMolBlock(rgroup_atoms_to_rdkit(atoms, bonds))
    >>> [line for line in block.splitlines() if line.startswith("M  RGP")]
    ['M  RGP  2   7   1  10   2']
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    rgroup = atoms.arrays.get('rgroup', np.zeros(len(atoms), dtype=int))
    mol = Chem.RWMol()
    for atom, label in zip(atoms, rgroup):
        if label:
            rd_atom = Chem.Atom(0)
            rd_atom.SetIsotope(int(label))
            rd_atom.SetAtomMapNum(int(label))
            rd_atom.SetIntProp('_MolFileRLabel', int(label))
            rd_atom.SetProp('dummyLabel', f'R{int(label)}')
            rd_atom.SetProp('atomLabel', f'_R{int(label)}')
        else:
            rd_atom = Chem.Atom(int(atom.number))
        rd_atom.SetNoImplicit(True)
        mol.AddAtom(rd_atom)

    i_idx, j_idx = np.nonzero(np.triu(np.asarray(bond_matrix), k=1))
    for i, j in zip(i_idx.tolist(), j_idx.tolist()):
        value = float(bond_matrix[i, j])
        if abs(value - 1.5) < 1e-6:
            # Aromatic order (as written by bond-order perception): rounding
            # it would turn every ring bond into a double bond.
            mol.AddBond(i, j, Chem.BondType.AROMATIC)
            bond = mol.GetBondBetweenAtoms(i, j)
            bond.SetIsAromatic(True)
            mol.GetAtomWithIdx(i).SetIsAromatic(True)
            mol.GetAtomWithIdx(j).SetIsAromatic(True)
            continue
        order = int(round(value))
        bond_type = getattr(Chem.BondType, _RDKIT_BOND.get(order, 'SINGLE'))
        mol.AddBond(i, j, bond_type)

    conformer = Chem.Conformer(len(atoms))
    for k, pos in enumerate(atoms.positions):
        conformer.SetAtomPosition(k, Point3D(*map(float, pos)))
    mol = mol.GetMol()
    mol.AddConformer(conformer, assignId=True)
    mol.UpdatePropertyCache(strict=False)
    return mol


def rgroup_smiles(atoms, bond_matrix):
    """
    CXSMILES of a finite isomer in which every site is an R-group atom.

    Each site is written as a dummy atom ``*`` carrying an atom label, so the
    string ends in e.g. ``|$;;_R1;;_R2$|``, the CXSMILES atom labels marking
    R1 and R2. The same molecule written to SDF/MOL (via
    :func:`rgroup_atoms_to_rdkit` and RDKit's ``MolToMolBlock``) shows
    ``R#`` atoms with ``M  RGP`` records.

    Examples
    --------
    >>> import numpy as np
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.sites import mark_site
    >>> benzene = molecule("C6H6")                    # C 0-5, H 6-11 (H i+6 on C i)
    >>> h_positions = benzene.positions[6:].copy()
    >>> for h in range(6, 12):
    ...     mark_site(benzene, h, 1)                  # every H a slot, as functionalise() does
    >>> slots = list(range(6, 12))
    >>> anchors = benzene[slots]
    >>> from cage_isomer_builder.utils.rgroup import RGroupIsomer, build_rgroup_isomer_atoms, rgroup_smiles
    >>> iso = RGroupIsomer((1, 0, 0, 2, 0, 0), ("NH2", "OH"), ("NH2", "OH"))
    >>> atoms = build_rgroup_isomer_atoms(benzene, anchors, iso, slots, h_positions)
    >>> from cage_isomer_builder.utils.features import bond_graph
    >>> bonds = np.zeros((12, 12))
    >>> for i, nbrs in bond_graph(atoms).items():
    ...     for j in nbrs:
    ...         bonds[i, j] = 1.5 if max(i, j) < 6 else 1.0
    >>> smiles = rgroup_smiles(atoms, bonds)
    >>> smiles.split(" ")[0]
    '[H]c1c([H])c([2*:2])c([H])c([H])c1[1*:1]'
    >>> "_R1" in smiles and "_R2" in smiles
    True
    """
    from rdkit import Chem

    return Chem.MolToCXSmiles(rgroup_atoms_to_rdkit(atoms, bond_matrix))
