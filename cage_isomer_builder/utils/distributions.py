#!/usr/bin/env python3
"""
Pair distributions of feature sites: the descriptor.

A :class:`Descriptor` stores, for every *pair type* (e.g. ``("acceptor",
"donor")`` or ``("fg:R1:endo", "fg:R2:exo")``), the distance of each pair of
feature sites, the angle between their direction vectors, and a weight. The
weight is 1 for a plain structure; ensembles (isomers, conformers, ring
rotations) use fractional weights so that a descriptor is the *expected*
number of pairs at each distance per structure.

Smoothed curves are weighted Gaussian kernel density estimates,

    rho(r) = sum_k  w_k  N(r; d_k, h),

so ``rho`` has units of pairs per Angstrom and integrates to the total pair
weight. Curves of different ensembles are therefore directly comparable and
add up. The 2-D version (distance x angle) uses a product kernel, with the
angle kernel reflected at the ends of its range so no weight leaks past
0 / 90 / 180 degrees.

Additivity
----------
A descriptor of the union of two feature sets A and B is *exactly*

    D(A + B) = D(A) + D(B) + D(A, B)

where ``D(A, B)`` holds the cross pairs (one site from each). So partial
distributions (e.g. node H-bond sites, linker pi-rings, functional groups)
build the full pore descriptor only when the cross terms are included:
``D(A) + D(B)`` alone is missing every A-B pair. :func:`assemble` does this
sum and :func:`additivity_residual` checks it numerically.

Periodic structures
-------------------
Pairs are counted per unit cell over every periodic image within
``max_distance`` (like a radial distribution function), not just the
nearest image, so a distribution of a MOF does not depend on the choice of
cell.
"""
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

@dataclass
class PairSet:
    """Distances (Angstrom), angles (degrees, NaN if undefined) and weights
    of the pairs of one pair type. ``max_angle`` is 90 when the angles are
    folded because a vector is a sign-less axis (a ring normal), else 180.

    Examples
    --------
    >>> import numpy as np
    >>> from cage_isomer_builder.utils.distributions import PairSet
    >>> ps = PairSet(np.array([2.0, 3.0]), np.array([np.nan, 90.0]), np.array([1.0, 0.5]))
    >>> len(ps), ps.total
    (2, 1.5)
    """
    distances: np.ndarray
    angles: np.ndarray
    weights: np.ndarray
    max_angle: float = 180.0

    @classmethod
    def empty(cls):
        """
        A pair set with no pairs.

        Examples
        --------
        >>> from cage_isomer_builder.utils.distributions import PairSet
        >>> len(PairSet.empty())
        0
        """
        return cls(np.zeros(0), np.zeros(0), np.zeros(0))

    def __add__(self, other):
        if len(self) == 0:
            return other
        if len(other) == 0:
            return self
        max_angle = self.max_angle
        if other.max_angle != self.max_angle:
            max_angle = float("nan")          # mixed folded/unfolded angles
        return PairSet(np.concatenate([self.distances, other.distances]),
                       np.concatenate([self.angles, other.angles]),
                       np.concatenate([self.weights, other.weights]), max_angle)

    def __len__(self):
        return len(self.distances)

    def scaled(self, factor):
        """
        Copy with every weight multiplied by ``factor``.

        Examples
        --------
        >>> import numpy as np
        >>> from cage_isomer_builder.utils.distributions import PairSet
        >>> PairSet(np.array([2.0]), np.array([np.nan]), np.array([1.0])).scaled(0.25).total
        0.25
        """
        return PairSet(self.distances, self.angles, self.weights * factor, self.max_angle)

    @property
    def total(self):
        """
        Total weight of the pairs.

        Examples
        --------
        >>> import numpy as np
        >>> from cage_isomer_builder.utils.distributions import PairSet
        >>> PairSet(np.array([1.0, 2.0]), np.array([np.nan, np.nan]), np.array([0.5, 0.5])).total
        1.0
        """
        return float(self.weights.sum())


def pair_key(type_a, type_b):
    """Order-independent key of a pair type.

    Examples
    --------
    >>> from cage_isomer_builder.utils.distributions import pair_key
    >>> pair_key("pi", "acceptor") == pair_key("acceptor", "pi")
    True
    >>> pair_key("pi", "acceptor")
    ('acceptor', 'pi')
    """
    return tuple(sorted((str(type_a), str(type_b))))


@dataclass
class Descriptor:
    """
    Pair distributions keyed by pair type.

    Attributes
    ----------
    pairs : dict
        ``{(type_a, type_b): PairSet}`` with ``type_a <= type_b``.
    info : dict
        Free-form metadata (e.g. the ensemble it was averaged over).

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.features import detect_features
    >>> from cage_isomer_builder.utils.distributions import pair_descriptor
    >>> desc = pair_descriptor(detect_features(molecule("H2O")))
    >>> desc.summary()                      # {pair type: (number of pairs, weight)}
    {('acceptor', 'donor'): (2, 2.0), ('donor', 'donor'): (1, 1.0)}
    """
    pairs: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def keys(self):
        """
        Pair types present, sorted.

        Examples
        --------
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.features import detect_features
        >>> from cage_isomer_builder.utils.distributions import pair_descriptor
        >>> desc = pair_descriptor(detect_features(molecule("H2O")))
        >>> desc.keys()
        [('acceptor', 'donor'), ('donor', 'donor')]
        """
        return sorted(self.pairs)

    def add(self, key, pairset):
        """
        Add pairs to pair type ``key`` (the order of the two type names does not matter).

        Examples
        --------
        >>> import numpy as np
        >>> from cage_isomer_builder.utils.distributions import Descriptor, PairSet
        >>> desc = Descriptor()
        >>> desc.add(("pi", "acceptor"), PairSet(np.array([4.0]), np.array([np.nan]), np.array([1.0])))
        >>> desc.add(("acceptor", "pi"), PairSet(np.array([5.0]), np.array([np.nan]), np.array([1.0])))
        >>> desc.summary()
        {('acceptor', 'pi'): (2, 2.0)}
        """
        key = pair_key(*key)
        self.pairs[key] = self.pairs[key] + pairset if key in self.pairs else pairset

    def __add__(self, other):
        out = Descriptor({k: v for k, v in self.pairs.items()}, dict(self.info))
        for k, v in other.pairs.items():
            out.add(k, v)
        return out

    def scaled(self, factor):
        """Every weight multiplied by ``factor``.

        Examples
        --------
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.features import detect_features
        >>> from cage_isomer_builder.utils.distributions import pair_descriptor
        >>> desc = pair_descriptor(detect_features(molecule("H2O")))
        >>> desc.scaled(0.5).total()
        1.5
        """
        return Descriptor({k: v.scaled(factor) for k, v in self.pairs.items()}, dict(self.info))

    def merged(self, keys=None):
        """One PairSet of the given keys (default: all). A key may also be a
        single type name, meaning every pair type containing it.

        Examples
        --------
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.features import detect_features
        >>> from cage_isomer_builder.utils.distributions import pair_descriptor
        >>> desc = pair_descriptor(detect_features(molecule("H2O")))
        >>> len(desc.merged()), len(desc.merged("donor")), len(desc.merged([("acceptor", "donor")]))
        (3, 3, 2)
        """
        chosen = self._resolve(keys)
        out = PairSet.empty()
        for k in chosen:
            out = out + self.pairs[k]
        return out

    def _resolve(self, keys):
        if keys is None:
            return self.keys()
        if isinstance(keys, tuple) and len(keys) == 2 and all(isinstance(k, str) for k in keys):
            keys = [keys]
        if isinstance(keys, str):
            keys = [keys]
        out = []
        for k in keys:
            if isinstance(k, str):
                out += [key for key in self.pairs if k in key]
            else:
                k = pair_key(*k)
                if k in self.pairs:
                    out.append(k)
        return sorted(set(out))

    def total(self, keys=None):
        """Total pair weight of the given keys (default: all).

        Examples
        --------
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.features import detect_features
        >>> from cage_isomer_builder.utils.distributions import pair_descriptor
        >>> desc = pair_descriptor(detect_features(molecule("H2O")))
        >>> desc.total(), desc.total([("donor", "donor")])
        (3.0, 1.0)
        """
        return self.merged(keys).total

    def kde(self, grid, keys=None, bandwidth=0.1):
        """Weighted Gaussian KDE (pairs per Angstrom) on ``grid``.

        Examples
        --------
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.features import detect_features
        >>> from cage_isomer_builder.utils.distributions import pair_descriptor
        >>> desc = pair_descriptor(detect_features(molecule("H2O")))
        >>> import numpy as np
        >>> r = np.linspace(0, 5, 501)
        >>> round(float(np.trapezoid(desc.kde(r), r)), 3)        # integrates to the pair count
        3.0
        """
        return kde_1d(self.merged(keys), grid, bandwidth)

    def kde2d(self, distance_grid, angle_grid, keys=None, bandwidth=0.1,
              angle_bandwidth=5.0):
        """
        2-D KDE (pairs per Angstrom per degree) over distance x angle, using
        only pairs whose angle is defined. The angle range is [0, 90] for
        pair types involving a sign-less axis (ring normals) and [0, 180]
        otherwise; keys mixing the two raise ValueError.

        Examples
        --------
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.features import detect_features
        >>> from cage_isomer_builder.utils.distributions import pair_descriptor
        >>> desc = pair_descriptor(detect_features(molecule("H2O")))
        >>> import numpy as np
        >>> d, a = np.linspace(0, 4, 201), np.linspace(0, 180, 181)
        >>> z = desc.kde2d(d, a, [("acceptor", "donor")], bandwidth=0.2, angle_bandwidth=10)
        >>> z.shape, round(float(np.trapezoid(np.trapezoid(z, a, axis=1), d)), 2)
        ((201, 181), 2.0)
        """
        ps = self.merged(keys)
        if np.isnan(ps.max_angle):
            raise ValueError("these pair types mix folded (0-90) and unfolded (0-180) "
                             "angles; take the 2-D KDE of each separately.")
        return kde_2d(ps, distance_grid, angle_grid, bandwidth, angle_bandwidth,
                      (0.0, ps.max_angle))

    def histogram(self, keys=None, decimals=2):
        """Exact table: sorted unique distances (rounded) and total weight
        at each.

        Examples
        --------
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.features import detect_features
        >>> from cage_isomer_builder.utils.distributions import pair_descriptor
        >>> desc = pair_descriptor(detect_features(molecule("H2O")))
        >>> distances, weights = desc.histogram([("donor", "donor")])
        >>> distances.round(2).tolist(), weights.tolist()       # the H...H distance of water
        ([1.53], [1.0])
        """
        ps = self.merged(keys)
        if len(ps) == 0:
            return np.zeros(0), np.zeros(0)
        d = np.round(ps.distances, decimals)
        values, inverse = np.unique(d, return_inverse=True)
        return values, np.bincount(inverse, weights=ps.weights)

    def summary(self):
        """``{key: (number of pairs, total weight)}``.

        Examples
        --------
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.features import detect_features
        >>> from cage_isomer_builder.utils.distributions import pair_descriptor
        >>> desc = pair_descriptor(detect_features(molecule("H2O")))
        >>> desc.summary()[("donor", "donor")]
        (1, 1.0)
        """
        return {k: (len(v), v.total) for k, v in sorted(self.pairs.items())}


# ---------------------------------------------------------------------------
# Building descriptors
# ---------------------------------------------------------------------------

def feature_type(features, i, by="kind"):
    """
    Type name of feature ``i``:

    * ``"kind"``: ``"donor"``
    * ``"label"``: ``"donor:O-H"``
    * ``"orientation"``: ``"fg:R1:endo"`` (label plus endo/exo/tangential
      when the feature has one)

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.features import detect_features
    >>> from cage_isomer_builder.utils.distributions import feature_type
    >>> feats = detect_features(molecule("H2O")).select(kinds="donor")
    >>> feature_type(feats, 0, "kind"), feature_type(feats, 0, "label")
    ('donor', 'donor:O-H')
    """
    kind = str(features.kinds[i])
    if by == "kind":
        return kind
    label = f"{kind}:{features.labels[i]}"
    if by == "label":
        return label
    if by == "orientation":
        o = str(features.orientation[i])
        return f"{label}:{o}" if o else label
    raise ValueError(f"by={by!r}; use 'kind', 'label' or 'orientation'.")


def pair_angles(vec_a, vec_b, directed_a, directed_b):
    """Angle (degrees) between paired unit vectors; folded into [0, 90]
    when either is a sign-less axis; NaN when either vector is zero.

    Examples
    --------
    >>> import numpy as np
    >>> from cage_isomer_builder.utils.distributions import pair_angles
    >>> z, minus_z = np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]])
    >>> pair_angles(z, minus_z, [True], [True]).tolist()     # directed vectors
    [180.0]
    >>> pair_angles(z, minus_z, [False], [False]).tolist()   # ring normals (sign-less)
    [0.0]
    """
    na = np.linalg.norm(vec_a, axis=-1)
    nb = np.linalg.norm(vec_b, axis=-1)
    cos = np.einsum("ij,ij->i", vec_a, vec_b) / np.where(na * nb > 0, na * nb, 1.0)
    axis = ~(np.asarray(directed_a) & np.asarray(directed_b))
    cos = np.where(axis, np.abs(cos), cos)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.where((na > 0) & (nb > 0), ang, np.nan)


def _pair_list(features, other, max_distance):
    """(i, j, distance, same_cell) for every pair; j indexes ``other`` when
    given. Periodic: all images within max_distance, each unordered
    within-set pair once (weight 0.5 per direction is applied by caller)."""
    periodic = bool(np.any(features.pbc))
    if not periodic:
        if other is None:
            i, j = np.triu_indices(len(features), k=1)
            d = np.linalg.norm(features.positions[i] - features.positions[j], axis=1)
        else:
            i, j = np.meshgrid(np.arange(len(features)), np.arange(len(other)), indexing="ij")
            i, j = i.ravel(), j.ravel()
            d = np.linalg.norm(features.positions[i] - other.positions[j], axis=1)
        keep = d <= max_distance if max_distance is not None else np.ones(len(d), bool)
        return i[keep], j[keep], d[keep], np.ones(keep.sum(), bool), np.ones(keep.sum())
    if max_distance is None:
        raise ValueError("periodic structures need max_distance (pairs are counted "
                         "over all images within it).")
    n_a = len(features)
    pos = features.positions if other is None else np.vstack([features.positions, other.positions])
    pseudo = Atoms(numbers=np.ones(len(pos), dtype=int), positions=pos,
                   cell=features.cell, pbc=features.pbc)
    i, j, d, S = neighbor_list("ijdS", pseudo, max_distance, self_interaction=False)
    if other is None:
        # every unordered pair appears once in each direction
        weight = np.full(len(i), 0.5)
    else:
        keep = (i < n_a) & (j >= n_a)
        i, j, d, S = i[keep], j[keep] - n_a, d[keep], S[keep]
        weight = np.ones(len(i))
    # "Same building block" = the minimum-image pair of two features (a
    # linker split by the cell boundary still has its own two sites as
    # each other's nearest images, whatever cell shift that takes).
    from ase.geometry import find_mic
    other_pos = features.positions if other is None else other.positions
    _, d_mic = find_mic(other_pos[j] - features.positions[i], features.cell, features.pbc)
    nearest_image = np.abs(d - d_mic) < 1e-6
    return i, j, d, nearest_image, weight


def pair_descriptor(features, other=None, by="kind", exclude_same_owner=True,
                    feature_weights=None, other_weights=None, max_distance=None):
    """
    Descriptor of every pair of features (``other=None``) or of every cross
    pair between two feature sets.

    Parameters
    ----------
    features, other : FeatureSet
    by : str
        How pairs are typed (see :func:`feature_type`).
    exclude_same_owner : bool, default True
        Skip pairs inside one building block (e.g. two sites on one ring),
        as in the original FG-FG statistics. Periodic images of one
        building block are different building blocks and are kept.
    feature_weights, other_weights : array-like, optional
        Occupation probability of each feature; a pair gets the product.
    max_distance : float, optional
        Only pairs up to this distance. Required for periodic structures.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.utils.features import detect_features
    >>> from cage_isomer_builder.utils.distributions import pair_descriptor
    >>> feats = detect_features(example_data.read("ibuprofen"))
    >>> sorted(pair_descriptor(feats).keys())
    [('acceptor', 'acceptor'), ('acceptor', 'donor'), ('acceptor', 'pi'), ('donor', 'pi')]
    >>> # cross pairs between two feature sets:
    >>> cross = pair_descriptor(feats.select(kinds="pi"), feats.select(kinds="acceptor"))
    >>> cross.total()
    2.0
    """
    i, j, d, same_cell, w = _pair_list(features, other, max_distance)
    b = features if other is None else other
    if exclude_same_owner:
        oa, ob = features.owners[i], b.owners[j]
        keep = ~((oa == ob) & (oa >= 0) & same_cell)   # same_cell: nearest image
        i, j, d, w = i[keep], j[keep], d[keep], w[keep]
    if feature_weights is not None:
        w = w * np.asarray(feature_weights, dtype=float)[i]
    if other_weights is not None or (other is None and feature_weights is not None):
        wb = other_weights if other is not None else feature_weights
        w = w * np.asarray(wb, dtype=float)[j]
    ang = pair_angles(features.vectors[i], b.vectors[j], features.directed[i], b.directed[j])
    types_a = np.array([feature_type(features, k, by) for k in range(len(features))])
    types_b = types_a if other is None else np.array([feature_type(b, k, by) for k in range(len(b))])
    desc = Descriptor(info={"by": by})
    keys = [pair_key(ta, tb) for ta, tb in zip(types_a[i], types_b[j])]
    axis_pair = ~(features.directed[i] & b.directed[j])
    for key in sorted(set(keys)):
        m = np.array([k == key for k in keys])
        folded = axis_pair[m]
        if folded.any() and not folded.all():
            raise ValueError(f"pair type {key} mixes directed and sign-less vectors.")
        desc.add(key, PairSet(d[m], ang[m], w[m], 90.0 if folded.all() else 180.0))
    return desc


def assemble(parts, by="kind", exclude_same_owner=True, max_distance=None):
    """
    Full descriptor of several feature sets taken together: every partial
    descriptor plus every cross term (see the module docstring).

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.utils.features import detect_features
    >>> from cage_isomer_builder.utils.distributions import assemble, pair_descriptor
    >>> feats = detect_features(example_data.read("ibuprofen"))
    >>> parts = [feats.select(kinds="pi"), feats.select(kinds=["donor", "acceptor"])]
    >>> assemble(parts).total() == pair_descriptor(feats).total()   # partials + cross terms
    True
    """
    parts = list(parts)
    out = Descriptor(info={"by": by})
    for a in range(len(parts)):
        out = out + pair_descriptor(parts[a], by=by, exclude_same_owner=exclude_same_owner,
                                    max_distance=max_distance)
        for b in range(a + 1, len(parts)):
            out = out + pair_descriptor(parts[a], parts[b], by=by,
                                        exclude_same_owner=exclude_same_owner,
                                        max_distance=max_distance)
    return out


def additivity_residual(parts, grid, by="kind", bandwidth=0.1, max_distance=None):
    """
    Largest absolute difference between the KDE of the union of ``parts``
    and the KDE of :func:`assemble` (partials + cross terms), and the
    weight that the partials alone miss. The first number should be at
    floating-point noise level.

    Returns
    -------
    residual : float
    missing_cross_weight : float
        Total weight of the cross pairs, which summing the partial
        distributions alone would leave out.

    Examples
    --------
    >>> import numpy as np
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.utils.features import detect_features
    >>> from cage_isomer_builder.utils.distributions import additivity_residual
    >>> feats = detect_features(example_data.read("ibuprofen"))
    >>> parts = [feats.select(kinds="pi"), feats.select(kinds=["donor", "acceptor"])]
    >>> residual, missing = additivity_residual(parts, np.linspace(0, 15, 301))
    >>> residual < 1e-12, missing            # 2 pi-acceptor + 1 pi-donor cross pairs
    (True, 3.0)
    """
    union = parts[0]
    for p in parts[1:]:
        union = union + p
    full = pair_descriptor(union, by=by, max_distance=max_distance)
    summed = assemble(parts, by=by, max_distance=max_distance)
    partial_only = Descriptor()
    for p in parts:
        partial_only = partial_only + pair_descriptor(p, by=by, max_distance=max_distance)
    residual = float(np.abs(full.kde(grid, bandwidth=bandwidth)
                            - summed.kde(grid, bandwidth=bandwidth)).max())
    return residual, full.total() - partial_only.total()


# ---------------------------------------------------------------------------
# Smoothing and comparison
# ---------------------------------------------------------------------------

def kde_1d(pairset, grid, bandwidth=0.1):
    """sum_k w_k N(grid; d_k, bandwidth), in pairs per Angstrom.

    Examples
    --------
    >>> import numpy as np
    >>> from cage_isomer_builder.utils.distributions import PairSet, kde_1d
    >>> ps = PairSet(np.array([3.0]), np.array([np.nan]), np.array([2.0]))
    >>> r = np.linspace(0, 6, 601)
    >>> round(float(np.trapezoid(kde_1d(ps, r, bandwidth=0.2), r)), 3)
    2.0
    """
    grid = np.asarray(grid, dtype=float).ravel()
    if len(pairset) == 0:
        return np.zeros_like(grid)
    z = (grid[:, None] - pairset.distances[None, :]) / bandwidth
    return (np.exp(-0.5 * z * z) @ pairset.weights) / (bandwidth * np.sqrt(2 * np.pi))


def _reflected_angle_kernel(grid, angles, bandwidth, lo, hi):
    def g(x):
        z = (grid[:, None] - x[None, :]) / bandwidth
        return np.exp(-0.5 * z * z) / (bandwidth * np.sqrt(2 * np.pi))
    return g(angles) + g(2 * lo - angles) + g(2 * hi - angles)


def kde_2d(pairset, distance_grid, angle_grid, bandwidth=0.1, angle_bandwidth=5.0,
           angle_range=(0.0, 180.0)):
    """2-D product-kernel KDE, shape (len(distance_grid), len(angle_grid)).

    Examples
    --------
    >>> import numpy as np
    >>> from cage_isomer_builder.utils.distributions import PairSet, kde_2d
    >>> ps = PairSet(np.array([3.0]), np.array([0.0]), np.array([1.0]))
    >>> d, a = np.linspace(0, 6, 301), np.linspace(0, 90, 181)
    >>> z = kde_2d(ps, d, a, bandwidth=0.2, angle_bandwidth=5.0, angle_range=(0.0, 90.0))
    >>> round(float(np.trapezoid(np.trapezoid(z, a, axis=1), d)), 3)   # reflected at 0 degrees
    1.0
    """
    dg = np.asarray(distance_grid, dtype=float).ravel()
    ag = np.asarray(angle_grid, dtype=float).ravel()
    ok = ~np.isnan(pairset.angles)
    if not ok.any():
        return np.zeros((len(dg), len(ag)))
    z = (dg[:, None] - pairset.distances[ok][None, :]) / bandwidth
    kd = np.exp(-0.5 * z * z) / (bandwidth * np.sqrt(2 * np.pi))
    ka = _reflected_angle_kernel(ag, pairset.angles[ok], angle_bandwidth, *angle_range)
    return (kd * pairset.weights[ok][None, :]) @ ka.T


def _normalised(y, x):
    area = np.trapezoid(y, x)
    return y / area if area > 0 else np.zeros_like(y)


def overlap(y1, y2, x):
    """Overlap of two curves after normalising each to unit area:
    integral of min(p, q), from 0 (disjoint) to 1 (identical shape).

    Examples
    --------
    >>> import numpy as np
    >>> from cage_isomer_builder.utils.distributions import overlap
    >>> x = np.linspace(0, 10, 1001)
    >>> p, q = np.exp(-(x - 4) ** 2), np.exp(-(x - 6) ** 2)
    >>> round(overlap(p, p, x), 3), round(overlap(p, q, x), 3)
    (1.0, 0.157)
    """
    return float(np.trapezoid(np.minimum(_normalised(y1, x), _normalised(y2, x)), x))


def bhattacharyya(y1, y2, x):
    """Bhattacharyya coefficient of the two normalised curves (0 to 1).

    Examples
    --------
    >>> import numpy as np
    >>> from cage_isomer_builder.utils.distributions import bhattacharyya
    >>> x = np.linspace(0, 10, 1001)
    >>> p, q = np.exp(-(x - 4) ** 2), np.exp(-(x - 6) ** 2)
    >>> round(bhattacharyya(p, p, x), 3), round(bhattacharyya(p, q, x), 3)
    (1.0, 0.368)
    """
    return float(np.trapezoid(np.sqrt(_normalised(y1, x) * _normalised(y2, x)), x))
