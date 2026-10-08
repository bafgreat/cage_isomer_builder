#!/usr/bin/env python3
"""
Describe a guest the same way as the host and match the two (Objective 3).

Guest descriptor
----------------
A flexible guest is a set of conformers from a GA/PES scan, MD snapshots or
any conformer generator; this module does not generate them. Each
conformer's feature sites (:func:`cage_isomer_builder.utils.features.detect_features`)
give a pair descriptor; the guest descriptor is their weighted average,
with Boltzmann weights from conformer energies or uniform weights (e.g. for
MD snapshots, already Boltzmann-distributed by the simulation).

Matching
--------
A guest binds where the host offers *complementary* features at the same
separations as the guest's own: a guest donor-acceptor pair at distance d fits
a host acceptor-donor pair at about d. For every guest pair type (a, b) the
host pair types (a', b') with a' complementary to a and b' to b are pooled,
and the two distance distributions are compared after normalising each to unit
area (overlap = integral of min(p, q), 0 to 1). The total score is the
average over guest pair types, weighted by the guest's pair weights, so it is
also between 0 and 1.

Default complementarity (:data:`COMPLEMENTARY`):

* guest donor     -> host acceptor
* guest acceptor  -> host donor, host open metal site
* guest pi        -> host pi
* guest fg        -> host fg (same label), for functional-group fingerprints

This is a screening score, a fast way to rank pores, isomers and guests
before computing host-guest energies; it is not an energy. The distance
compared is between features of the same molecule, so contact offsets
(e.g. the H...O distance) cancel to first order but are not modelled.
"""
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

from dataclasses import dataclass, field

import numpy as np

from cage_isomer_builder.utils.distributions import (
    Descriptor, bhattacharyya, overlap, pair_descriptor,
)
from cage_isomer_builder.utils.ensemble import boltzmann_weights
from cage_isomer_builder.utils.features import detect_features

#: guest feature kind -> host feature kinds it can interact with.
COMPLEMENTARY = {
    "donor": ("acceptor",),
    "acceptor": ("donor", "open_metal"),
    "pi": ("pi",),
    "fg": ("fg",),
}

GUEST_KINDS = ("pi", "donor", "acceptor")


def conformer_descriptor(conformers, energies=None, temperature=298.15, unit="eV",
                         weights=None, kinds=GUEST_KINDS, by="kind"):
    """
    Pair descriptor of a guest averaged over its conformers.

    Parameters
    ----------
    conformers : sequence of ase.Atoms
        All the same molecule (any orientation).
    energies : sequence of float, optional
        Conformer energies, for Boltzmann weights at ``temperature``.
    weights : sequence of float, optional
        Explicit weights instead (normalised). Default: uniform.
    kinds : sequence of str
        Feature kinds to use.
    by : str
        Pair typing (see :func:`cage_isomer_builder.utils.distributions.feature_type`).

    Returns
    -------
    Descriptor
        Expected pairs per molecule. ``info["weights"]`` holds the conformer
        weights used.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.matching import conformer_descriptor
    >>> water = molecule("H2O")
    >>> bent = water.copy()
    >>> bent.positions[1] *= 1.05                      # a second, slightly different conformer
    >>> desc = conformer_descriptor([water, bent], energies=[0.0, 0.02])
    >>> [round(w, 3) for w in desc.info["weights"]]    # Boltzmann weights at 298 K
    [0.685, 0.315]
    >>> desc.total()                                   # 2 acceptor-donor + 1 donor-donor per molecule
    3.0
    """
    conformers = list(conformers)
    if not conformers:
        raise ValueError("need at least one conformer.")
    if weights is None:
        weights = (boltzmann_weights(energies, temperature, unit) if energies is not None
                   else np.full(len(conformers), 1.0 / len(conformers)))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    out = Descriptor(info={"by": by})
    reference = None
    for atoms, w in zip(conformers, weights):
        if reference is None:
            reference = atoms.get_chemical_symbols()
        elif atoms.get_chemical_symbols() != reference:
            raise ValueError("conformers must have the same atoms in the same order.")
        feats = detect_features(atoms, kinds=kinds)
        # a guest is one molecule: no building blocks to exclude
        out = out + pair_descriptor(feats, by=by, exclude_same_owner=False).scaled(w)
    out.info = {"by": by, "weights": weights.tolist(), "n_conformers": len(conformers)}
    return out


@dataclass
class MatchResult:
    """
    Attributes
    ----------
    score : float
        Weighted mean overlap over guest pair types, 0 to 1.
    per_pair : dict
        ``{guest pair type: (overlap, host pair types used, guest weight)}``.
    unmatched : list
        Guest pair types for which the host has no complementary pairs
        (they score 0).
    defined : bool
        False when the guest has no feature pairs at all (e.g. a single
        aromatic ring and nothing else): pair distributions then say nothing
        and ``score`` is 0 by convention, not a measured mismatch.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.matching import conformer_descriptor, match_descriptors
    >>> water = conformer_descriptor([molecule("H2O")])
    >>> result = match_descriptors(water, water)
    >>> result.defined, sorted(result.per_pair)
    (True, [('acceptor', 'donor'), ('donor', 'donor')])
    """
    score: float
    per_pair: dict = field(default_factory=dict)
    unmatched: list = field(default_factory=list)
    defined: bool = True


def _kind(type_name):
    return type_name.split(":")[0]


def _complements(guest_type, host_types, complementarity):
    kind = _kind(guest_type)
    allowed = set(complementarity.get(kind, ()))
    if kind == "fg":
        return [t for t in host_types if t == guest_type]
    return [t for t in host_types if _kind(t) in allowed]


def match_descriptors(host, guest, grid=None, bandwidth=0.25, metric="overlap",
                      complementarity=None):
    """
    Score how well a guest's feature-pair distances fit the host's
    complementary feature-pair distances (see the module docstring).

    Parameters
    ----------
    host, guest : Descriptor
        Typed with the same ``by`` (normally ``"kind"``).
    grid : array-like, optional
        Distance grid (Angstrom). Default 0 to 30 in 0.02 steps.
    bandwidth : float, default 0.25
        KDE width (Angstrom), wider than for host-only statistics because a
        binding pocket tolerates some mismatch.
    metric : {"overlap", "bhattacharyya"}
    complementarity : dict, optional
        Overrides :data:`COMPLEMENTARY`.

    Returns
    -------
    MatchResult

    Examples
    --------
    >>> import numpy as np
    >>> from cage_isomer_builder.utils.distributions import Descriptor, PairSet
    >>> from cage_isomer_builder.utils.matching import match_descriptors
    >>> def pairs(key, d):
    ...     out = Descriptor()
    ...     out.add(key, PairSet(np.array([d]), np.array([np.nan]), np.array([1.0])))
    ...     return out
    >>> guest = pairs(("acceptor", "donor"), 5.0)          # guest donor and acceptor 5 A apart
    >>> round(match_descriptors(pairs(("acceptor", "donor"), 5.0), guest).score, 3)
    1.0
    >>> round(match_descriptors(pairs(("acceptor", "donor"), 7.0), guest).score, 3)
    0.0
    >>> match_descriptors(pairs(("pi", "pi"), 5.0), guest).unmatched   # nothing complementary
    [('acceptor', 'donor')]
    """
    if grid is None:
        grid = np.arange(0.0, 30.0, 0.02)
    grid = np.asarray(grid, dtype=float)
    comp = complementarity or COMPLEMENTARY
    compare = {"overlap": overlap, "bhattacharyya": bhattacharyya}[metric]
    host_types = sorted({t for key in host.keys() for t in key})
    per_pair, unmatched = {}, []
    total_w, total = 0.0, 0.0
    for key in guest.keys():
        w = guest.total([key])
        if w <= 0:
            continue
        ga, gb = key
        ha, hb = _complements(ga, host_types, comp), _complements(gb, host_types, comp)
        host_keys = sorted({tuple(sorted((x, y))) for x in ha for y in hb} & set(host.keys()))
        total_w += w
        if not host_keys:
            unmatched.append(key)
            per_pair[key] = (0.0, [], w)
            continue
        score = compare(host.kde(grid, host_keys, bandwidth), guest.kde(grid, [key], bandwidth), grid)
        per_pair[key] = (score, host_keys, w)
        total += w * score
    return MatchResult(total / total_w if total_w else 0.0, per_pair, unmatched,
                       defined=total_w > 0)


def rank_hosts(hosts, guest, **kwargs):
    """``[(name, MatchResult), ...]`` best first, for ``{name: Descriptor}``
    hosts (e.g. pores, isomers or defect variants).

    Examples
    --------
    >>> import numpy as np
    >>> from cage_isomer_builder.utils.distributions import Descriptor, PairSet
    >>> from cage_isomer_builder.utils.matching import rank_hosts
    >>> def pairs(d):
    ...     out = Descriptor()
    ...     out.add(("acceptor", "donor"), PairSet(np.array([d]), np.array([np.nan]), np.array([1.0])))
    ...     return out
    >>> ranked = rank_hosts({"small pore": pairs(4.0), "matching pore": pairs(5.0)}, pairs(5.0))
    >>> [name for name, _ in ranked]
    ['matching pore', 'small pore']
    """
    results = [(name, match_descriptors(desc, guest, **kwargs)) for name, desc in hosts.items()]
    return sorted(results, key=lambda r: -r[1].score)
