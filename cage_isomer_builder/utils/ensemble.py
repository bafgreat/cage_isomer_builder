#!/usr/bin/env python3
"""
Functional-group pair distributions averaged over an isomer ensemble -
multivariate MFMs, partial functionalisation, defects, ratios and linker
flexibility (Objective 2).

The question answered here: "over all ways of placing these groups, how
many NH2-NH2 / NH2-CH3 / CH3-CH3 pairs (endo-endo, endo-exo, ...) are there at
each distance, per structure?"

Exact average over configurations
---------------------------------
Every raw configuration (every placement of the groups on the slots, before
symmetry) counts once. Because a symmetry-unique isomer stands for exactly
``|orbit|`` raw configurations, this is the same as averaging over the
unique isomers weighted by orbit size, which describes a sample with no
energetic preference. That average has a closed form, so no
enumeration is needed:

* linkers carry patterns (``"AB"``, ``"A"``, ``""``, a vacancy, ...); with
  fixed numbers ``N_P`` of linkers per pattern (``mode="fixed"``, a real cage
  or cell), two different linkers carry patterns P and Q with probability

      pi(P, Q) = N_P (N_Q - [P = Q]) / (n (n - 1));

  with independent linkers (``mode="independent"``, a large crystal with
  fraction f_P of each pattern) it is f_P f_Q;
* within a linker carrying P, every arrangement is equally likely, so a slot
  carries label a with probability c_P(a) / F, and two slots of the same
  linker carry a and b with probability c_P(a) (c_P(b) - [a = b]) / (F (F - 1)).

So two slots on different linkers carry (a, b) with probability

    P(a, b) = sum_{P,Q} pi(P, Q) c_P(a) c_Q(b) / F^2,

the same for every such pair of slots. Each slot pair contributes its
distance with weight P(a, b) to pair type (a, b). The total weight is the
expected number of pairs per structure. ``tests/test_ensemble.py`` checks the
formula against explicit orbit-weighted enumeration.

Other weightings (each unique isomer once, or a Boltzmann weight from an
energy per isomer) use :func:`isomer_ensemble_descriptor`.

Defects
-------
A missing linker is a linker pattern whose every slot carries the reserved
label :data:`VACANCY`. Its slots then contribute no pairs, and the symmetry
counting in :mod:`cage_isomer_builder.utils.rgroup` treats it exactly like any
other pattern (one arrangement, never mixed with real groups).

Periodic structures
-------------------
For a MOF the configuration of the unit cell repeats in every cell (as for
the enumerated MOF isomers), and pairs are counted per cell over every image
up to ``max_distance``. A slot and its own image then always carry the same
group, and a slot and the image of another slot of the same linker are
correlated like two slots of one linker; both cases are exact.
``mode="independent"`` (an infinite random crystal) is for finite pore
models only.

Flexibility
-----------
Slot positions may carry samples (e.g. ring rotation angles, see
:mod:`cage_isomer_builder.utils.flexibility`). Slots on the same rotor move
together (paired samples); slots on different rotors move independently, so
every combination of their samples contributes with the product of the
sample weights.
"""
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

import numpy as np

from cage_isomer_builder.utils.distributions import Descriptor, PairSet, pair_angles, pair_key
from cage_isomer_builder.utils.rgroup import SlotLayout, as_layout, make_spec

#: Reserved group name of a missing linker's slots.
VACANCY = "_vac"


def vacancy_pattern(fg_per_linker):
    """Linker pattern of a missing linker: every slot is a vacancy.

    Examples
    --------
    >>> from cage_isomer_builder.utils.ensemble import vacancy_pattern
    >>> vacancy_pattern(4)
    ('_vac', '_vac', '_vac', '_vac')
    """
    return tuple([VACANCY] * fg_per_linker)


# Label-pair probabilities

def _composition_matrix(spec):
    """C[P, a] = number of slots of label a+1 in pattern P."""
    return np.array(spec.patterns, dtype=float).reshape(len(spec.patterns), len(spec.labels))


class LabelProbabilities:
    """
    Exact label probabilities of slots, for every linker type.

    For a linker of type t carrying pattern P with probability pi_t(P)
    (``N_P / n_t`` with fixed counts, or the given fraction), one slot carries
    label a with probability ``single(t)[a] = sum_P pi_t(P) c_P(a) / F_t``.
    Two slots of *different* linkers of the same type with fixed counts are
    correlated (``N_P (N_Q - [P = Q]) / (n_t (n_t - 1))``); slots of linkers of
    different types, or in ``"independent"`` mode, are not.

    Parameters
    ----------
    spec : RGroupSpec
    layout : SlotLayout
        The real layout (slot counts per type).
    mode : {"fixed", "independent"}
    fractions : array-like, optional
        ``"independent"`` only: probability of each pattern of ``spec``
        (summing to 1 within each linker type).

    Examples
    --------
    >>> from cage_isomer_builder.utils.rgroup import make_spec
    >>> from cage_isomer_builder.utils.ensemble import LabelProbabilities
    >>> spec = make_spec(4, 2, linker_patterns={"A": 2, "B": 2})   # 4 linkers, 2 slots each
    >>> probs = LabelProbabilities(spec, spec.layout)
    >>> probs.single(0).tolist()            # a slot carries A or B with 1/4 each
    [0.25, 0.25]
    """

    def __init__(self, spec, layout, mode="fixed", fractions=None):
        if mode not in ("fixed", "independent"):
            raise ValueError(f"mode={mode!r}; use 'fixed' or 'independent'.")
        self.mode = mode
        self.C = _composition_matrix(spec)
        self.types = np.array(spec.pattern_types)
        counts = np.array(spec.pattern_counts, dtype=float)
        self.counts = counts
        self.F = {t: layout.type_size(t) for t in range(layout.n_types)
                  if layout.linkers_of_type(t)}
        self.n = {t: counts[self.types == t].sum() for t in self.F}
        if mode == "fixed":
            self.pi = np.array([counts[p] / self.n[self.types[p]] for p in range(len(counts))])
        else:
            f = (np.asarray(fractions, dtype=float) if fractions is not None
                 else np.array([counts[p] / self.n[self.types[p]] for p in range(len(counts))]))
            for t in self.F:
                total = f[self.types == t].sum()
                if not np.isclose(total, 1.0):
                    raise ValueError(f"fractions of a linker type must sum to 1, got {total}.")
            self.pi = f
        self._cache = {}

    def _patterns(self, t):
        return np.nonzero(self.types == t)[0]

    def single(self, t):
        """Probability that one slot of a type-t linker carries each label.

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import make_spec
        >>> from cage_isomer_builder.utils.ensemble import LabelProbabilities
        >>> spec = make_spec(2, 4, groups="AB")
        >>> LabelProbabilities(spec, spec.layout).single(0).tolist()
        [0.25, 0.25]
        """
        ps = self._patterns(t)
        if len(ps) == 0:
            return np.zeros(self.C.shape[1])
        return self.pi[ps] @ self.C[ps] / self.F[t]

    def across(self, t, u):
        """Two slots on different linkers of types t and u.

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import make_spec
        >>> from cage_isomer_builder.utils.ensemble import LabelProbabilities
        >>> spec = make_spec(4, 1, linker_patterns={"A": 2, "B": 2})   # one slot per linker
        >>> probs = LabelProbabilities(spec, spec.layout)
        >>> probs.across(0, 0).round(4).tolist()   # P(A,A) = 2*1/(4*3): fixed counts are correlated
        [[0.1667, 0.3333], [0.3333, 0.1667]]
        """
        key = ("a", t, u)
        if key not in self._cache:
            if t != u or self.mode == "independent":
                m = np.outer(self.single(t), self.single(u))
            else:
                ps = self._patterns(t)
                n = self.n[t]
                if n < 2:
                    m = np.zeros((self.C.shape[1],) * 2)
                else:
                    N = self.counts[ps]
                    pair = (np.outer(N, N) - np.diag(N)) / (n * (n - 1))
                    m = self.C[ps].T @ pair @ self.C[ps] / self.F[t] ** 2
            self._cache[key] = m
        return self._cache[key]

    def within(self, t):
        """Two different slots on one linker of type t.

        Examples
        --------
        >>> from cage_isomer_builder.utils.rgroup import make_spec
        >>> from cage_isomer_builder.utils.ensemble import LabelProbabilities
        >>> spec = make_spec(1, 2, groups="AB")            # one linker, two slots, A and B
        >>> LabelProbabilities(spec, spec.layout).within(0).tolist()
        [[0.0, 0.5], [0.5, 0.0]]
        """
        key = ("w", t)
        if key not in self._cache:
            F = self.F[t]
            m = np.zeros((self.C.shape[1],) * 2)
            if F > 1:
                for p in self._patterns(t):
                    m += self.pi[p] * (np.outer(self.C[p], self.C[p]) - np.diag(self.C[p]))
                m /= F * (F - 1)
            self._cache[key] = m
        return self._cache[key]


def pattern_pair_probabilities(spec, mode="fixed", fractions=None):
    """
    pi(P, Q) for two *different* linkers and pi(P) for one linker (single
    linker type).

    Parameters
    ----------
    spec : RGroupSpec
    mode : {"fixed", "independent"}
        "fixed": exactly ``spec.pattern_counts`` linkers per pattern.
        "independent": each linker draws pattern P with probability
        ``fractions[P]`` (default ``N_P / n``).

    Examples
    --------
    >>> from cage_isomer_builder.utils.rgroup import make_spec
    >>> from cage_isomer_builder.utils.ensemble import pattern_pair_probabilities
    >>> spec = make_spec(4, 1, linker_patterns={"A": 2, "B": 2})
    >>> pair, single = pattern_pair_probabilities(spec)
    >>> single.tolist(), pair.round(4).tolist()
    ([0.5, 0.5], [[0.1667, 0.3333], [0.3333, 0.1667]])
    """
    counts = np.array(spec.pattern_counts, dtype=float)
    n = counts.sum()
    if mode == "fixed":
        single = counts / n
        if n < 2:
            return np.zeros((len(counts), len(counts))), single
        pair = (np.outer(counts, counts) - np.diag(counts)) / (n * (n - 1))
        return pair, single
    if mode == "independent":
        f = single = (np.asarray(fractions, dtype=float) if fractions is not None
                      else counts / n)
        if not np.isclose(f.sum(), 1.0):
            raise ValueError(f"fractions must sum to 1, got {f.sum()}.")
        return np.outer(f, f), single
    raise ValueError(f"mode={mode!r}; use 'fixed' or 'independent'.")


def label_pair_probabilities(spec, mode="fixed", fractions=None):
    """
    Probabilities that two slots carry labels (a, b), labels numbered
    1..len(spec.labels) as in ``spec.labels`` (0 = H is left out), for a
    single linker type (see :class:`LabelProbabilities` for several).

    Returns
    -------
    across : np.ndarray, shape (n_labels, n_labels)
        Two slots on different linkers.
    within : np.ndarray, shape (n_labels, n_labels)
        Two different slots on the same linker.

    Examples
    --------
    >>> from cage_isomer_builder.utils.rgroup import make_spec
    >>> from cage_isomer_builder.utils.ensemble import label_pair_probabilities
    >>> spec = make_spec(3, 2, groups="A")             # one A on each 2-slot linker
    >>> across, within = label_pair_probabilities(spec)
    >>> across.tolist(), within.tolist()
    ([[0.25]], [[0.0]])
    """
    model = LabelProbabilities(spec, spec.layout, mode, fractions)
    return model.across(0, 0), model.within(0)


def single_label_probabilities(spec, mode="fixed", fractions=None):
    """Probability that one slot carries each label (single linker type).

    Examples
    --------
    >>> from cage_isomer_builder.utils.rgroup import make_spec
    >>> from cage_isomer_builder.utils.ensemble import single_label_probabilities
    >>> single_label_probabilities(make_spec(3, 4, groups="A")).tolist()
    [0.25]
    """
    return LabelProbabilities(spec, spec.layout, mode, fractions).single(0)


# Descriptors

def _slot_samples(slot_features, positions, vectors, orientations, sample_weights):
    n = len(slot_features)
    if positions is None:
        positions = slot_features.positions[:, None, :]
        vectors = slot_features.vectors[:, None, :]
        orientations = np.asarray(slot_features.orientation)[:, None]
        sample_weights = np.ones((n, 1))
    positions = np.asarray(positions, dtype=float)
    n_samples = positions.shape[1]
    if vectors is None:
        vectors = np.repeat(slot_features.vectors[:, None, :], n_samples, axis=1)
    if orientations is None:
        orientations = np.repeat(np.asarray(slot_features.orientation)[:, None], n_samples, axis=1)
    if sample_weights is None:
        sample_weights = np.full((n, n_samples), 1.0 / n_samples)
    sample_weights = np.asarray(sample_weights, dtype=float)
    if sample_weights.ndim == 1:
        sample_weights = np.repeat(sample_weights[None, :], n, axis=0)
    sample_weights = sample_weights / sample_weights.sum(axis=1, keepdims=True)
    return positions, np.asarray(vectors, dtype=float), np.asarray(orientations), sample_weights


def _type_name(label, orientation, by):
    if by == "label" or not orientation:
        return f"fg:{label}"
    return f"fg:{label}:{orientation}"


def _add_slot_pair(desc, x, y, probs, names, by, pos, vec, ori, sw, rotor, directed,
                   shift=None, weight=1.0, max_distance=None):
    """Add every labelled (a, b) contribution of slot pair (x, y), with y
    displaced by the lattice vector ``shift`` (periodic images)."""
    if rotor is not None and rotor[x] == rotor[y] and rotor[x] >= 0:
        sx, sy = np.arange(pos.shape[1]), np.arange(pos.shape[1])
        w_s = sw[x].copy()
    else:
        sx, sy = np.meshgrid(np.arange(pos.shape[1]), np.arange(pos.shape[1]), indexing="ij")
        sx, sy = sx.ravel(), sy.ravel()
        w_s = sw[x][sx] * sw[y][sy]
    w_s = w_s * weight
    other = pos[y, sy] if shift is None else pos[y, sy] + shift
    d = np.linalg.norm(other - pos[x, sx], axis=1)
    if max_distance is not None:
        keep = d <= max_distance
        if not keep.any():
            return
        sx, sy, d, w_s = sx[keep], sy[keep], d[keep], w_s[keep]
    ang = pair_angles(vec[x, sx], vec[y, sy], np.full(len(sx), directed), np.full(len(sx), directed))
    ox, oy = ori[x, sx], ori[y, sy]
    for a in range(len(names)):
        for b in range(len(names)):
            p = probs[a, b]
            if p <= 0 or names[a] == VACANCY or names[b] == VACANCY:
                continue
            if by == "label":
                desc.add((f"fg:{names[a]}", f"fg:{names[b]}"), PairSet(d, ang, w_s * p))
                continue
            keys = [pair_key(_type_name(names[a], o1, by), _type_name(names[b], o2, by))
                    for o1, o2 in zip(ox, oy)]
            unique_keys = sorted(set(keys))
            code_of = {key: n for n, key in enumerate(unique_keys)}
            codes = np.array([code_of[k] for k in keys], dtype=int)
            for n, key in enumerate(unique_keys):
                m = codes == n
                desc.add(key, PairSet(d[m], ang[m], w_s[m] * p))


def _periodic_pairs(slot_features, pos, max_distance):
    """
    Every (x, y, lattice shift) within ``max_distance`` of each other in at
    least one sample, each unordered pair listed in both directions (the
    caller weights each direction by 0.5), plus whether it is the nearest
    image of the two slots.
    """
    from ase import Atoms
    from ase.geometry import find_mic
    from ase.neighborlist import neighbor_list

    cell = np.asarray(slot_features.cell, dtype=float)
    static = pos[:, 0, :]
    moved = np.linalg.norm(pos - static[:, None, :], axis=2).max() if pos.shape[1] > 1 else 0.0
    pseudo = Atoms(numbers=np.ones(len(static), dtype=int), positions=static,
                   cell=cell, pbc=slot_features.pbc)
    i, j, d, S = neighbor_list("ijdS", pseudo, max_distance + 2 * moved + 1e-6,
                               self_interaction=False)
    # ASE's shifts refer to the positions as given (wrapped or not):
    # D = static[j] - static[i] + S @ cell.
    shifts = S @ cell
    d_static = np.linalg.norm(static[j] + shifts - static[i], axis=1)
    _, d_mic = find_mic(static[j] - static[i], cell, slot_features.pbc)
    nearest = np.abs(d_static - d_mic) < 1e-6
    return i, j, shifts, nearest


def ensemble_descriptor(slot_features, fg_per_linker, groups="A", linker_patterns=None,
                        mode="fixed", fractions=None, by="orientation",
                        include_same_linker=False, positions=None, vectors=None,
                        orientations=None, sample_weights=None, rotor_of_slot=None,
                        max_distance=None):
    """
    Exact configuration-averaged FG pair descriptor (see the module
    docstring).

    Parameters
    ----------
    slot_features : FeatureSet
        ``fg`` features in slot order (``CageBuilder.slot_features()``).
    fg_per_linker : int or SlotLayout
        Slots per linker, or the layout of linkers of several types
        (``CageBuilder.slot_layout()``).
    groups, linker_patterns
        As in :func:`cage_isomer_builder.utils.rgroup.make_spec` (per linker
        type for several types). Use :func:`vacancy_pattern` as a pattern for
        missing linkers.
    mode : {"fixed", "independent"}
        See :func:`pattern_pair_probabilities`.
    fractions : dict, optional
        With ``mode="independent"``: ``{pattern: fraction of linkers}``,
        e.g. ``{"A": 0.3, "B": 0.7}`` or ``{"A": 0.9, vacancy_pattern(4): 0.1}``;
        ``{type name: {pattern: fraction}}`` for several linker types.
        Describes any composition exactly, not only ones that fit a whole
        number of linkers (a large crystal). Overrides ``groups`` and
        ``linker_patterns``.
    by : {"orientation", "label"}
        Type pairs by group and endo/exo class, or by group only.
    include_same_linker : bool, default False
        Also count pairs of two groups on one linker.
    positions, vectors, orientations : arrays, shape (n_slots, n_samples, ...)
        Sampled slot geometry (flexibility). Default: the static geometry.
    sample_weights : array, shape (n_samples,) or (n_slots, n_samples)
        Weight of each sample (normalised per slot). Default: uniform.
    rotor_of_slot : sequence of int, optional
        Slots with the same rotor id move together (paired samples).
        Default: one rotor per linker when samples are given.
    max_distance : float, optional
        Periodic structures (required there): count pairs over every
        periodic image up to this distance, per unit cell. The
        configuration of the cell repeats in every cell, as for the
        enumerated isomers of a MOF, so a slot and its own image carry the
        same group, and a slot and an image of another slot of the same
        linker are correlated like two slots of one linker.

    Returns
    -------
    Descriptor
        Weights are expected pair counts per structure (per unit cell for
        periodic structures).

    Examples
    --------
    >>> from cage_isomer_builder.utils.features import FeatureSet
    >>> # three linkers with two FG slots each, along a line
    >>> slots = FeatureSet.from_rows([dict(position=[x, 0, 0], kind="fg", owner=x // 10)
    ...                               for x in (0, 1, 10, 11, 20, 21)])
    >>> from cage_isomer_builder.utils.ensemble import ensemble_descriptor, vacancy_pattern
    >>> d = ensemble_descriptor(slots, 2, groups="A", by="label")
    >>> d.total()                      # 3 groups -> 3 pairs per structure
    3.0
    >>> mixed = ensemble_descriptor(slots, 2, linker_patterns={"A": 2, "B": 1}, by="label")
    >>> {k: round(v, 3) for k, (_, v) in mixed.summary().items()}
    {('fg:A', 'fg:A'): 1.0, ('fg:A', 'fg:B'): 2.0}
    >>> defect = ensemble_descriptor(slots, 2, linker_patterns={"A": 2, vacancy_pattern(2): 1},
    ...                              by="label")
    >>> defect.total()                 # one linker missing: one A-A pair left
    1.0
    """
    n_slots = len(slot_features)
    layout = as_layout(fg_per_linker, n_slots)
    if mode == "independent" and fractions is not None:
        # Only the pattern compositions matter; build them on a stand-in
        # layout with one linker per pattern.
        per_type = _fractions_per_type(fractions, layout)
        stand_in = SlotLayout(
            sizes=tuple(layout.type_size(t) for t, f in enumerate(per_type) for _ in f),
            types=tuple(t for t, f in enumerate(per_type) for _ in f),
            type_names=layout.type_names)
        spec = make_spec(stand_in, None, linker_patterns=(
            {layout.type_names[t]: {p: 1 for p in f} for t, f in enumerate(per_type)}
            if layout.n_types > 1 else {p: 1 for p in per_type[0]}))
        fraction_array = np.array([v for f in per_type for v in f.values()], dtype=float)
    else:
        spec = make_spec(layout, None, groups, linker_patterns)
        fraction_array = None
    model = LabelProbabilities(spec, layout, mode, fraction_array)
    pos, vec, ori, sw = _slot_samples(slot_features, positions, vectors, orientations,
                                      sample_weights)
    block = layout.slot_linker
    slot_type = np.array(layout.types)[block]
    if rotor_of_slot is None:
        rotor_of_slot = block
    directed = bool(slot_features.directed[0]) if n_slots else True
    desc = Descriptor(info={"by": by, "mode": mode, "labels": spec.labels,
                            "patterns": dict(zip(spec.pattern_names, spec.pattern_counts))})

    if np.any(slot_features.pbc):
        if max_distance is None:
            raise ValueError("periodic structures need max_distance.")
        if mode == "independent":
            raise ValueError(
                "mode='independent' describes an infinite random crystal, but a "
                "periodic cell repeats its own configuration in every cell; use "
                "mode='fixed' with whole-linker counts for periodic structures.")
        for x, y, shift, nearest in zip(*_periodic_pairs(slot_features, pos, max_distance)):
            if x == y:
                probs = np.diag(model.single(slot_type[x]))   # a slot and its own image
            elif block[x] == block[y]:
                if nearest and not include_same_linker:
                    continue                                   # two slots of one linker
                probs = model.within(slot_type[x])
            else:
                probs = model.across(slot_type[x], slot_type[y])
            _add_slot_pair(desc, x, y, probs, spec.labels, by, pos, vec, ori, sw,
                           rotor_of_slot, directed, shift=shift, weight=0.5,
                           max_distance=max_distance)
        return desc

    for x in range(n_slots):
        for y in range(x + 1, n_slots):
            same = block[x] == block[y]
            if same and not include_same_linker:
                continue
            probs = (model.within(slot_type[x]) if same
                     else model.across(slot_type[x], slot_type[y]))
            _add_slot_pair(desc, x, y, probs, spec.labels, by,
                           pos, vec, ori, sw, rotor_of_slot, directed,
                           max_distance=max_distance)
    return desc


def _fractions_per_type(fractions, layout):
    """``{pattern: f}`` (one linker type) or ``{type name: {pattern: f}}``
    -> list of dicts, one per type."""
    fractions = dict(fractions)
    if layout.n_types > 1 or (set(fractions) == {layout.type_names[0]}
                              and isinstance(fractions[layout.type_names[0]], dict)):
        if set(fractions) != set(layout.type_names):
            raise ValueError(f"with several linker types {list(layout.type_names)}, give "
                             "fractions as {type name: {pattern: fraction}}.")
        return [dict(fractions[n]) for n in layout.type_names]
    return [fractions]


def isomer_ensemble_descriptor(slot_features, isomers, fg_per_linker, weights=None,
                               by="orientation", include_same_linker=False,
                               max_distance=None):
    """
    FG pair descriptor averaged over explicit isomers with given weights.

    Each isomer's active slots are passed to
    :func:`cage_isomer_builder.utils.distributions.pair_descriptor` (so
    periodic images are handled the same way; ``max_distance`` is required
    for periodic structures).

    Parameters
    ----------
    isomers : sequence of RGroupIsomer
    weights : sequence of float, optional
        One per isomer, normalised to sum 1. Default: each isomer equally.
        Use :func:`orbit_sizes` for the configuration average, or
        :func:`boltzmann_weights` for energies.

    Examples
    --------
    >>> from cage_isomer_builder.utils.features import FeatureSet
    >>> # three linkers with two FG slots each, along a line
    >>> slots = FeatureSet.from_rows([dict(position=[x, 0, 0], kind="fg", owner=x // 10)
    ...                               for x in (0, 1, 10, 11, 20, 21)])
    >>> from cage_isomer_builder.utils.rgroup import RGroupIsomer
    >>> from cage_isomer_builder.utils.ensemble import isomer_ensemble_descriptor
    >>> iso = RGroupIsomer((1, 0, 1, 0, 1, 0), ("A",), ("A", "A", "A"))   # first slot of each
    >>> isomer_ensemble_descriptor(slots, [iso], 2, by="label").histogram()[0].tolist()
    [10.0, 20.0]
    """
    from cage_isomer_builder.utils.distributions import pair_descriptor

    if weights is None:
        weights = np.ones(len(isomers))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    if by not in ("orientation", "label"):
        raise ValueError(f"by={by!r}; use 'orientation' or 'label'.")
    desc = Descriptor(info={"by": by, "mode": "isomers"})
    base = slot_features.subset(np.arange(len(slot_features)))
    base.owners = as_layout(fg_per_linker, len(slot_features)).slot_linker
    for iso, w in zip(isomers, weights):
        labels = np.asarray(iso.site_labels)
        names = iso.labels
        active = [x for x in np.nonzero(labels)[0] if names[labels[x] - 1] != VACANCY]
        if len(active) < 2 and not np.any(base.pbc):
            continue
        sub = base.subset(active)
        sub.labels = np.array([names[labels[x] - 1] for x in active], dtype="<U16")
        part = pair_descriptor(sub, by=by, exclude_same_owner=not include_same_linker,
                               max_distance=max_distance)
        desc = desc + part.scaled(w)
    desc.info = {"by": by, "mode": "isomers"}
    return desc


def orbit_sizes(group, isomers):
    """Number of raw configurations each isomer stands for:
    |G| / |stabiliser|, using a prepared ``rgroup.SlotGroup``.

    Examples
    --------
    >>> from cage_isomer_builder.utils import rgroup
    >>> from cage_isomer_builder.utils.ensemble import orbit_sizes
    >>> group = rgroup.prepare_group([[1, 0]], 2, 2)     # one linker, its two slots swap
    >>> isomers = rgroup.enumerate_rgroup_isomers(group, 2, 2, groups="A")
    >>> orbit_sizes(group, isomers).tolist()             # the one isomer stands for 2 placements
    [2]
    """
    sizes = []
    for iso in isomers:
        f = np.asarray(iso.site_labels)
        stab = sum(1 for inv in group.inverse if np.array_equal(f[inv], f))
        sizes.append(group.order // stab)
    return np.array(sizes)


def boltzmann_weights(energies, temperature=298.15, unit="eV"):
    """Normalised Boltzmann weights exp(-(E - E_min) / kT).

    Examples
    --------
    >>> from cage_isomer_builder.utils.ensemble import boltzmann_weights
    >>> boltzmann_weights([0.0, 0.05], temperature=298.15).round(3).tolist()
    [0.875, 0.125]
    """
    k_b = {"eV": 8.617333262e-5, "kJ/mol": 8.314462618e-3, "kcal/mol": 1.987204259e-3}[unit]
    e = np.asarray(energies, dtype=float)
    w = np.exp(-(e - e.min()) / (k_b * temperature))
    return w / w.sum()


# Ratios

def ratio_series(slot_features, fg_per_linker, labels=("A", "B"), by="orientation",
                 include_same_linker=False, max_distance=None, linker_type=None,
                 other_groups=""):
    """
    Exact descriptors for every whole-linker ratio of two groups, one group
    per linker: ``{fraction of linkers with labels[0]: Descriptor}`` for
    k = 0..n linkers. Distances are the same at every ratio; only the
    weights of the label pairs change. ``max_distance`` is required for
    periodic structures (see :func:`ensemble_descriptor`).

    With several linker types, the ratio is varied on ``linker_type`` (a
    type name) and every other type carries ``other_groups`` (bare by
    default).

    Examples
    --------
    >>> from cage_isomer_builder.utils.features import FeatureSet
    >>> # three linkers with two FG slots each, along a line
    >>> slots = FeatureSet.from_rows([dict(position=[x, 0, 0], kind="fg", owner=x // 10)
    ...                               for x in (0, 1, 10, 11, 20, 21)])
    >>> from cage_isomer_builder.utils.ensemble import ratio_series
    >>> series = ratio_series(slots, 2, labels=("A", "B"), by="label")
    >>> [round(f, 3) for f in series]                 # 0, 1, 2 or 3 of the 3 linkers carry A
    [0.0, 0.333, 0.667, 1.0]
    >>> series[1.0].total([("fg:A", "fg:A")])
    3.0
    """
    layout = as_layout(fg_per_linker, len(slot_features))
    if layout.n_types > 1 and linker_type is None:
        raise ValueError(f"choose the linker_type to vary: {list(layout.type_names)}.")
    t = 0 if linker_type is None else layout.type_names.index(linker_type)
    n_linkers = len(layout.linkers_of_type(t))
    a, b = labels
    out = {}
    for k in range(n_linkers + 1):
        patterns = {p: n for p, n in {a: k, b: n_linkers - k}.items() if n}
        if layout.n_types > 1:
            patterns = {name: (patterns if u == t else
                               {other_groups: len(layout.linkers_of_type(u))})
                        for u, name in enumerate(layout.type_names)}
        out[k / n_linkers] = ensemble_descriptor(
            slot_features, layout, linker_patterns=patterns, by=by,
            include_same_linker=include_same_linker, max_distance=max_distance)
    return out


def interpolate_descriptor(series, x):
    """
    Linear interpolation between the two descriptors of ``series`` (a
    ``{fraction: Descriptor}`` dict) that bracket ``x``. Because every
    descriptor holds the same pairs with different weights, this is exactly
    linear interpolation of their KDE curves.

    Note: for a fixed number of linkers the exact weights are quadratic in
    the fraction (e.g. P(A, A) = k (k - 1) / (n (n - 1))), so the
    interpolation is approximate between grid points. For an exact value at
    any fraction in a large crystal use :func:`ensemble_descriptor` with
    ``mode="independent"``.

    Examples
    --------
    >>> from cage_isomer_builder.utils.features import FeatureSet
    >>> # three linkers with two FG slots each, along a line
    >>> slots = FeatureSet.from_rows([dict(position=[x, 0, 0], kind="fg", owner=x // 10)
    ...                               for x in (0, 1, 10, 11, 20, 21)])
    >>> from cage_isomer_builder.utils.ensemble import interpolate_descriptor, ratio_series
    >>> series = ratio_series(slots, 2, labels=("A", "B"), by="label")
    >>> half = interpolate_descriptor(series, 0.5)    # between 1/3 and 2/3
    >>> round(half.total([("fg:A", "fg:A")]), 3)
    0.5
    """
    fracs = np.array(sorted(series))
    if x < fracs[0] - 1e-12 or x > fracs[-1] + 1e-12:
        raise ValueError(f"fraction {x} is outside the series range [{fracs[0]}, {fracs[-1]}].")
    hi = int(np.searchsorted(fracs, x))
    if hi < len(fracs) and np.isclose(fracs[hi], x):
        return series[fracs[hi]]
    lo = hi - 1
    t = (x - fracs[lo]) / (fracs[hi] - fracs[lo])
    out = series[fracs[lo]].scaled(1 - t) + series[fracs[hi]].scaled(t)
    out.info = {"interpolated": float(x), "between": (float(fracs[lo]), float(fracs[hi]))}
    return out
