#!/usr/bin/env python3
"""
Chemical feature sites of a host (cage, MOF, COF) or a guest molecule.

A *feature* is a point with a type and, where it has one, a direction:

==============  =================================  ==================================
kind            position                           vector
==============  =================================  ==================================
``fg``          functional-group site (R site)     ring carbon -> site, unit (directed)
``pi``          aromatic ring centroid             ring normal (axis: sign-less)
``donor``       the H of an O-H / N-H              heavy atom -> H, unit (directed)
``acceptor``    an O or N with a free lone pair    lone-pair bisector, unit (directed)
``open_metal``  a metal below its full             direction of the empty
                coordination                       coordination site (directed)
==============  =================================  ==================================

The same detector runs on hosts and guests, so their descriptors can be
compared directly (see :mod:`cage_isomer_builder.utils.matching`).

Rules (kept simple, so a result can be checked by hand):

* **Bonds** are perceived from covalent radii (``natural_cutoffs`` x 1.2), except
  metal-ligand contacts, which use x 1.15 so a carboxylate C next to a metal
  is not counted as a ligand atom. R-site markers (``X``) are bonded to their
  nearest heavy atom.
* **Aromatic rings** are 5- and 6-membered cycles of C/N/O/S/B atoms, each
  with at most three neighbours, lying in a plane (RMS deviation below
  0.15 Angstrom).
* **Donors** are H atoms bonded to O or N. A donor whose O/N is bonded to two
  or more metals is labelled ``mu<n>-OH`` (e.g. the mu3-OH of a Zr6 node).
* **Acceptors** are O atoms, and N atoms that are not three-coordinate and
  planar (amide/aromatic-substituted N). O and N bonded to a metal *and* to
  a non-metal (e.g. a metal-bound carboxylate O) are excluded by default,
  being coordinatively saturated; metal-only O (oxo, mu3-O) are kept and
  labelled ``mu<n>-O``.
* **Open metal sites**: by default mofstructure's geometric detector
  (Chung et al., the CoRE MOF method): a metal is open when its coordination
  sphere is an incomplete polyhedron (an octahedron, square antiprism, ...
  with a site missing), so every Cu of a bare paddlewheel is open and a
  defect-free UiO-66 has none. The alternative ``method="coordination"``
  compares coordination numbers: a metal is open below ``expected_cn`` for
  its element, or below the most highly coordinated metal of that element in
  the structure. The direction is minus the sum of the unit vectors to the
  ligands.
* **FG orientation** (endo/exo): the cosine between the ring-C -> site bond and
  the direction from that carbon to the pore centre. ``endo`` if above
  ``threshold`` (default 0.5, i.e. within 60 degrees), ``exo`` if below ``-threshold``, else ``tangential`` (e.g.
  every H of a ring lying face-on to the pore). Periodic structures have no
  single pore centre unless one is given.
"""
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

from dataclasses import dataclass, field

import numpy as np
from ase.data import covalent_radii
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list

from cage_isomer_builder.utils.sites import site_array, site_indices

KINDS = ("fg", "pi", "donor", "acceptor", "open_metal")

#: Atomic numbers that are *not* metals: dummy (0), non-metals, noble gases
#: and the metalloids B, Si, Ge, As, Sb, Te.
_NON_METALS = frozenset({0, 1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 32, 33, 34,
                         35, 36, 51, 52, 53, 54, 85, 86})
_RING_ELEMENTS = frozenset({"C", "N", "O", "S", "B"})


def is_metal(z):
    """True for metals (incl. alkali/alkaline earth, Al, Ga, In, Sn, ...).

    Examples
    --------
    >>> from cage_isomer_builder.utils.features import is_metal
    >>> is_metal(40), is_metal(6), is_metal(0)          # Zr, C, dummy X
    (True, False, False)
    """
    return int(z) not in _NON_METALS


# Container

@dataclass
class FeatureSet:
    """
    Feature sites, as parallel arrays (one row per feature).

    Attributes
    ----------
    positions : np.ndarray, shape (n, 3)
    vectors : np.ndarray, shape (n, 3)
        Unit direction of each feature, or zeros if it has none.
    directed : np.ndarray of bool, shape (n,)
        False for a sign-less axis (a ring normal), so angles to it are
        folded into [0, 90] degrees.
    kinds : np.ndarray of str
        One of :data:`KINDS`.
    labels : np.ndarray of str
        Finer chemical label: R-group name for ``fg``, ``"O-H"``,
        ``"mu3-O"``, ``"Zr(CN=7)"``, ``"C6"``, ...
    orientation : np.ndarray of str
        ``"endo"``, ``"exo"``, ``"tangential"`` for ``fg`` features in a
        finite structure; ``""`` otherwise.
    owners : np.ndarray of int
        Building block (linker/node) each feature belongs to, -1 if unknown.
        Pairs inside one building block can be excluded from descriptors.
    atom_indices : np.ndarray of int
        A representative atom of the feature in the source structure (site,
        H, acceptor, metal, first ring atom).
    cell : np.ndarray, shape (3, 3)
    pbc : np.ndarray of bool, shape (3,)

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.features import detect_features
    >>> feats = detect_features(molecule("H2O"))
    >>> len(feats), sorted(feats.kinds.tolist())
    (3, ['acceptor', 'donor', 'donor'])
    >>> feats.counts()
    {'acceptor': 1, 'donor': 2}
    """
    positions: np.ndarray
    vectors: np.ndarray
    directed: np.ndarray
    kinds: np.ndarray
    labels: np.ndarray
    orientation: np.ndarray
    owners: np.ndarray
    atom_indices: np.ndarray
    cell: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    pbc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=bool))

    def __len__(self):
        return len(self.positions)

    @classmethod
    def empty(cls, cell=None, pbc=None):
        """
        A feature set with no features (optionally carrying a cell).

        Examples
        --------
        >>> from cage_isomer_builder.utils.features import FeatureSet
        >>> len(FeatureSet.empty())
        0
        """
        return cls(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, dtype=bool),
                   np.zeros(0, dtype="<U16"), np.zeros(0, dtype="<U16"),
                   np.zeros(0, dtype="<U16"), np.zeros(0, dtype=int),
                   np.zeros(0, dtype=int),
                   np.zeros((3, 3)) if cell is None else np.asarray(cell, dtype=float),
                   np.zeros(3, dtype=bool) if pbc is None else np.asarray(pbc, dtype=bool))

    @classmethod
    def from_rows(cls, rows, cell=None, pbc=None):
        """Build from dicts with keys position, vector, directed, kind,
        label, orientation, owner, atom.

        Examples
        --------
        >>> from cage_isomer_builder.utils.features import FeatureSet
        >>> fs = FeatureSet.from_rows([
        ...     dict(position=[0, 0, 0], vector=[0, 0, 1], directed=False, kind="pi", label="C6"),
        ...     dict(position=[3.5, 0, 0], kind="acceptor", label="O"),
        ... ])
        >>> fs.kinds.tolist(), fs.directed.tolist()
        (['pi', 'acceptor'], [False, True])
        """
        if not rows:
            return cls.empty(cell, pbc)
        return cls(
            np.array([r["position"] for r in rows], dtype=float).reshape(-1, 3),
            np.array([r.get("vector", np.zeros(3)) for r in rows], dtype=float).reshape(-1, 3),
            np.array([r.get("directed", True) for r in rows], dtype=bool),
            np.array([r["kind"] for r in rows], dtype="<U16"),
            np.array([r.get("label", r["kind"]) for r in rows], dtype="<U16"),
            np.array([r.get("orientation", "") for r in rows], dtype="<U16"),
            np.array([r.get("owner", -1) for r in rows], dtype=int),
            np.array([r.get("atom", -1) for r in rows], dtype=int),
            np.zeros((3, 3)) if cell is None else np.asarray(cell, dtype=float),
            np.zeros(3, dtype=bool) if pbc is None else np.asarray(pbc, dtype=bool),
        )

    def subset(self, mask):
        """Features where ``mask`` (bool array or index list) is selected.

        Examples
        --------
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.features import detect_features
        >>> feats = detect_features(molecule("H2O"))
        >>> len(feats.subset([0, 1]))
        2
        """
        idx = np.asarray(mask)
        if idx.dtype == bool:
            idx = np.nonzero(idx)[0]
        return FeatureSet(self.positions[idx], self.vectors[idx], self.directed[idx],
                          self.kinds[idx], self.labels[idx], self.orientation[idx],
                          self.owners[idx], self.atom_indices[idx], self.cell, self.pbc)

    def select(self, kinds=None, labels=None):
        """Features of the given kind(s) and/or label(s).

        Examples
        --------
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.features import detect_features
        >>> feats = detect_features(molecule("C5H5N"))           # pyridine
        >>> feats.select(kinds="acceptor").labels.tolist()
        ['N']
        """
        mask = np.ones(len(self), dtype=bool)
        if kinds is not None:
            mask &= np.isin(self.kinds, [kinds] if isinstance(kinds, str) else list(kinds))
        if labels is not None:
            mask &= np.isin(self.labels, [labels] if isinstance(labels, str) else list(labels))
        return self.subset(mask)

    def __add__(self, other):
        if len(self) == 0:
            return other
        if len(other) == 0:
            return self
        return FeatureSet(
            np.vstack([self.positions, other.positions]),
            np.vstack([self.vectors, other.vectors]),
            np.concatenate([self.directed, other.directed]),
            np.concatenate([self.kinds, other.kinds]).astype("<U16"),
            np.concatenate([self.labels, other.labels]).astype("<U16"),
            np.concatenate([self.orientation, other.orientation]).astype("<U16"),
            np.concatenate([self.owners, other.owners]),
            np.concatenate([self.atom_indices, other.atom_indices]),
            self.cell, self.pbc,
        )

    def counts(self):
        """``{kind: number of features}``.

        Examples
        --------
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.features import detect_features
        >>> detect_features(molecule("NH3")).counts()
        {'acceptor': 1, 'donor': 3}
        """
        kinds, n = np.unique(self.kinds, return_counts=True)
        return dict(zip(kinds.tolist(), n.tolist()))


# Bond graph

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros(3)


def _vec(atoms, i, j):
    """Vector i -> j, minimum image if periodic."""
    d = atoms.positions[j] - atoms.positions[i]
    if atoms.pbc.any():
        d, _ = find_mic(d, atoms.cell, atoms.pbc)
    return d


def bond_graph(atoms, mult=1.2, metal_mult=1.15):
    """
    Adjacency lists ``{i: [j, ...]}`` from covalent radii.

    Metal-ligand contacts use the tighter ``metal_mult``; metal-metal
    contacts are left out (they are not coordination bonds). Each R-site
    marker (``X``) is bonded to its nearest heavy atom only.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.features import bond_graph
    >>> graph = bond_graph(molecule("H2O"))               # atoms: O, H, H
    >>> sorted(graph[0])
    [1, 2]
    """
    numbers = atoms.numbers
    radii = covalent_radii[numbers].copy()
    sites = set(site_indices(atoms))
    radii[numbers == 0] = 0.0
    cutoff = float(2 * radii.max() * mult) + 0.01
    i_list, j_list, d_list = neighbor_list("ijd", atoms, cutoff)
    adjacency = {i: [] for i in range(len(atoms))}
    for i, j, d in zip(i_list, j_list, d_list):
        if i >= j or numbers[i] == 0 or numbers[j] == 0:
            continue
        mi, mj = is_metal(numbers[i]), is_metal(numbers[j])
        if mi and mj:
            continue
        factor = metal_mult if (mi or mj) else mult
        if d <= factor * (radii[i] + radii[j]) and int(j) not in adjacency[i]:
            # (a small periodic cell can list the same pair via two images)
            adjacency[i].append(int(j))
            adjacency[j].append(int(i))
    heavy = np.array([k for k in range(len(atoms))
                      if numbers[k] > 1 and k not in sites], dtype=int)
    for s in sites:
        if len(heavy) == 0:
            break
        d = atoms.positions[heavy] - atoms.positions[s]
        if atoms.pbc.any():
            d, _ = find_mic(d, atoms.cell, atoms.pbc)
        j = int(heavy[np.argmin(np.linalg.norm(d, axis=1))])
        adjacency[s].append(j)
        adjacency[j].append(s)
    return adjacency


def adjacency_from_bond_matrix(atoms, bond_matrix):
    """
    Adjacency lists from a known bond-order matrix (e.g. STK's), leaving
    out metal-metal entries. Use this instead of :func:`bond_graph`
    whenever the true bonds are known: a built but unoptimised structure
    can have bonds far longer than any distance cutoff.

    Examples
    --------
    >>> import numpy as np
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.features import adjacency_from_bond_matrix
    >>> water = molecule("H2O")
    >>> bonds = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    >>> adjacency_from_bond_matrix(water, bonds)
    {0: [1, 2], 1: [0], 2: [0]}
    """
    numbers = atoms.numbers
    adjacency = {i: [] for i in range(len(atoms))}
    for i, j in zip(*np.nonzero(np.triu(np.asarray(bond_matrix), k=1))):
        if is_metal(numbers[i]) and is_metal(numbers[j]):
            continue
        adjacency[int(i)].append(int(j))
        adjacency[int(j)].append(int(i))
    return adjacency


# Rings

def find_rings(atoms, adjacency=None, sizes=(5, 6)):
    """
    Every simple cycle of the given sizes among ring-forming atoms
    (C/N/O/S/B, at most three neighbours), each as a tuple of atom
    indices in ring order, starting at its lowest index.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.features import find_rings
    >>> find_rings(molecule("C6H6"))
    [(0, 1, 2, 3, 4, 5)]
    """
    if adjacency is None:
        adjacency = bond_graph(atoms)
    symbols = atoms.get_chemical_symbols()
    ok = {i for i, s in enumerate(symbols)
          if s in _RING_ELEMENTS
          and len([j for j in adjacency[i] if atoms.numbers[j] != 0]) <= 3}
    max_size = max(sizes)
    rings = set()
    for start in sorted(ok):
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for nb in adjacency[node]:
                if nb not in ok or nb < start:
                    continue
                if nb == start and len(path) in sizes:
                    # each cycle is found twice (both directions): keep one
                    if path[1] < path[-1]:
                        rings.add(tuple(path))
                elif nb not in path and len(path) < max_size:
                    stack.append((nb, path + [nb]))
    # drop cycles that are not chordless (e.g. a 6-cycle around a fused 5/5 pair)
    out = []
    for ring in rings:
        members = set(ring)
        chords = sum(1 for a in ring for b in adjacency[a] if b in members)
        if chords == 2 * len(ring):
            out.append(ring)
    return sorted(out)


def ring_geometry(atoms, ring):
    """Centroid, unit normal and RMS out-of-plane deviation of a ring
    (unwrapped across periodic boundaries).

    Examples
    --------
    >>> import numpy as np
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.features import find_rings, ring_geometry
    >>> benzene = molecule("C6H6")
    >>> centroid, normal, rms = ring_geometry(benzene, find_rings(benzene)[0])
    >>> np.round(np.abs(normal), 3).tolist(), rms < 1e-6       # flat ring in the xy plane
    ([0.0, 0.0, 1.0], True)
    """
    base = atoms.positions[ring[0]]
    pts = np.array([base + _vec(atoms, ring[0], k) for k in ring])
    centroid = pts.mean(axis=0)
    _, s, vt = np.linalg.svd(pts - centroid)
    rms = s[2] / np.sqrt(len(ring))
    return centroid, vt[2], float(rms)


def aromatic_rings(atoms, adjacency=None, planarity=0.15):
    """Rings from :func:`find_rings` that are planar within ``planarity``
    (RMS, Angstrom).

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.features import aromatic_rings
    >>> len(aromatic_rings(molecule("C6H6"))), len(aromatic_rings(molecule("C2H4")))
    (1, 0)
    """
    if adjacency is None:
        adjacency = bond_graph(atoms)
    return [r for r in find_rings(atoms, adjacency)
            if ring_geometry(atoms, r)[2] < planarity]


# Detectors

def _owner_lookup(owners):
    if owners is None:
        return lambda i: -1
    table = {int(a): k for k, atoms in enumerate(owners) for a in atoms}
    return lambda i: table.get(int(i), -1)


def pi_features(atoms, adjacency=None, owners=None, planarity=0.15):
    """One ``pi`` feature per aromatic ring: centroid + ring normal.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.features import pi_features
    >>> rings = pi_features(molecule("C5H5N"))
    >>> rings.labels.tolist(), bool(rings.directed[0])
    (['CN6'], False)
    """
    if adjacency is None:
        adjacency = bond_graph(atoms)
    owner = _owner_lookup(owners)
    rows = []
    symbols = atoms.get_chemical_symbols()
    for ring in aromatic_rings(atoms, adjacency, planarity):
        centroid, normal, _ = ring_geometry(atoms, ring)
        composition = "".join(sorted({symbols[k] for k in ring}))
        rows.append(dict(position=centroid, vector=normal, directed=False, kind="pi",
                         label=f"{composition}{len(ring)}", owner=owner(ring[0]),
                         atom=ring[0]))
    return FeatureSet.from_rows(rows, atoms.cell[:], atoms.pbc)


def hbond_features(atoms, adjacency=None, owners=None, include_metal_bound=False):
    """``donor`` and ``acceptor`` features (see the module rules).

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.features import hbond_features
    >>> hbond_features(molecule("CH3CONH2")).counts()    # the planar amide N is no acceptor
    {'acceptor': 1, 'donor': 2}
    """
    if adjacency is None:
        adjacency = bond_graph(atoms)
    owner = _owner_lookup(owners)
    numbers = atoms.numbers
    symbols = atoms.get_chemical_symbols()
    rows = []
    for i, s in enumerate(symbols):
        nbrs = [j for j in adjacency[i] if numbers[j] != 0]
        metals = [j for j in nbrs if is_metal(numbers[j])]
        nonmetals = [j for j in nbrs if not is_metal(numbers[j])]
        if s == "H" and len(nonmetals) == 1 and symbols[nonmetals[0]] in ("O", "N"):
            d = nonmetals[0]
            n_metal = sum(1 for j in adjacency[d] if is_metal(numbers[j]))
            label = f"mu{n_metal}-{symbols[d]}H" if n_metal >= 2 else f"{symbols[d]}-H"
            rows.append(dict(position=atoms.positions[i], vector=_unit(_vec(atoms, d, i)),
                             kind="donor", label=label, owner=owner(i), atom=i))
        if s not in ("O", "N"):
            continue
        if metals and nonmetals and not include_metal_bound:
            continue                       # metal-bound carboxylate O, amine N, ...
        if s == "N" and not metals and len(nbrs) >= 3:
            if len(nbrs) >= 4:
                continue                   # ammonium
            u = [_unit(_vec(atoms, i, j)) for j in nbrs]
            angle_sum = sum(np.degrees(np.arccos(np.clip(u[a] @ u[b], -1, 1)))
                            for a, b in ((0, 1), (0, 2), (1, 2)))
            if angle_sum > 350.0:
                continue                   # planar: lone pair in the pi system
        lone = -sum((_unit(_vec(atoms, i, j)) for j in nbrs), np.zeros(3))
        if metals and not nonmetals:
            label = f"mu{len(metals)}-{s}" if len(metals) >= 2 else f"{s}-M"
        else:
            label = s
        rows.append(dict(position=atoms.positions[i], vector=_unit(lone), kind="acceptor",
                         label=label, owner=owner(i), atom=i))
    return FeatureSet.from_rows(rows, atoms.cell[:], atoms.pbc)


def coordination_numbers(atoms, adjacency=None):
    """``{metal index: number of bonded non-metal, non-site atoms}``.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> sbu = example_data.read("uio66_sbu")
    >>> sbu = sbu[[a.index for a in sbu if a.symbol != "X"]]      # drop connection markers
    >>> from cage_isomer_builder.utils.features import coordination_numbers
    >>> sorted(set(coordination_numbers(sbu).values()))   # every Zr is 8-coordinate
    [8]
    """
    if adjacency is None:
        adjacency = bond_graph(atoms)
    numbers = atoms.numbers
    return {i: sum(1 for j in adjacency[i] if numbers[j] > 0 and not is_metal(numbers[j]))
            for i in range(len(atoms)) if is_metal(numbers[i])}


def geometric_open_metals(atoms, padding=15.0):
    """
    Open-metal-site analysis of every metal by mofstructure's detector
    (``omsdetector_forked``, Chung et al., the CoRE MOF open-metal-site
    method): each metal's first coordination sphere is checked against the
    ideal closed polyhedra, so a site is open when part of its coordination
    shell is missing, whether or not other metals of the same element are
    more highly coordinated (e.g. every Cu of a bare paddlewheel is open).

    Finite structures are placed in a box ``padding`` Angstrom larger than
    the structure on every side, so no periodic image is within bonding
    reach. R-site markers (``X``) are left out.

    Returns
    -------
    dict
        ``{metal index: (is_open, site type, number of ligands,
        unit vacancy direction)}``. The vacancy direction is minus the sum
        of the unit vectors to the detector's ligands (zero if they cancel).

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> sbu = example_data.read("uio66_sbu")
    >>> sbu = sbu[[a.index for a in sbu if a.symbol != "X"]]      # drop connection markers
    >>> from cage_isomer_builder.utils.features import geometric_open_metals
    >>> result = geometric_open_metals(sbu)
    >>> any(is_open for is_open, *_ in result.values())
    False
    """
    from omsdetector_forked import mof

    keep = [i for i in range(len(atoms)) if atoms.numbers[i] > 0]
    sub = atoms[keep]
    if sub.pbc.all():
        lattice, coords = np.asarray(sub.cell[:]), sub.positions
    else:
        lo, hi = sub.positions.min(axis=0), sub.positions.max(axis=0)
        lattice = np.diag(hi - lo + 2 * padding)
        coords = sub.positions - lo + padding
    structure = mof.MofStructure(lattice=lattice, species=sub.get_chemical_symbols(),
                                 coords=coords, coords_are_cartesian=True, name="oms")
    out = {}
    for m, sphere in enumerate(structure.metal_coord_spheres):
        sphere.check_if_open()
        cart = np.asarray(sphere.cart_coords)
        vacancy = -sum((_unit(c - cart[0]) for c in cart[1:]), np.zeros(3))
        out[keep[structure.metal_indices[m]]] = (
            bool(sphere.is_open), str(sphere.metal_type), int(sphere.num_linkers), _unit(vacancy),
        )
    return out


def _geometric_detector_available():
    import importlib.util
    return importlib.util.find_spec("omsdetector_forked") is not None


def open_metal_features(atoms, adjacency=None, owners=None, expected_cn=None,
                        method="auto"):
    """
    ``open_metal`` features.

    Parameters
    ----------
    method : {"auto", "geometry", "coordination"}
        ``"geometry"``: mofstructure's detector (:func:`geometric_open_metals`),
        the default when it is installed (``"auto"``). ``"coordination"``:
        a metal is open when it has fewer ligand atoms than ``expected_cn``
        for its element, or than the most highly coordinated metal of that
        element in the structure (see the module rules); used by ``"auto"``
        only when the detector is not installed. ``expected_cn`` is ignored
        by ``"geometry"``.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> sbu = example_data.read("uio66_sbu")
    >>> sbu = sbu[[a.index for a in sbu if a.symbol != "X"]]      # drop connection markers
    >>> from cage_isomer_builder.utils.features import bond_graph, open_metal_features
    >>> graph = bond_graph(sbu)
    >>> carbon = next(a.index for a in sbu if a.symbol == "C")
    >>> gone = [carbon] + [j for j in graph[carbon] if sbu[j].symbol == "O"]
    >>> defect = sbu[[i for i in range(len(sbu)) if i not in gone]]   # remove one carboxylate
    >>> open_metal_features(defect).labels.tolist()
    ['Zr(CN=7)', 'Zr(CN=7)']
    """
    if method == "auto":
        method = "geometry" if _geometric_detector_available() else "coordination"
    owner = _owner_lookup(owners)
    symbols = atoms.get_chemical_symbols()
    if adjacency is None:
        adjacency = bond_graph(atoms)
    cn = coordination_numbers(atoms, adjacency)
    rows = []
    if method == "geometry":
        for i, (is_open, site_type, n, vacancy) in sorted(geometric_open_metals(atoms).items()):
            if not is_open:
                continue
            if np.linalg.norm(vacancy) < 1e-6:
                vacancy = _cluster_vacancy(atoms, i, list(cn))
            rows.append(dict(position=atoms.positions[i], vector=vacancy, kind="open_metal",
                             label=f"{symbols[i]}(CN={n})", owner=owner(i), atom=i))
        return FeatureSet.from_rows(rows, atoms.cell[:], atoms.pbc)
    if method != "coordination":
        raise ValueError(f"method={method!r}; use 'auto', 'geometry' or 'coordination'.")
    reference = {}
    for i, n in cn.items():
        reference[symbols[i]] = max(reference.get(symbols[i], 0), n)
    reference.update(dict(expected_cn or {}))
    for i, n in cn.items():
        if n >= reference[symbols[i]]:
            continue
        ligands = [j for j in adjacency[i] if atoms.numbers[j] > 0 and not is_metal(atoms.numbers[j])]
        vacancy = -sum((_unit(_vec(atoms, i, j)) for j in ligands), np.zeros(3))
        if np.linalg.norm(vacancy) < 0.3:
            vacancy = _cluster_vacancy(atoms, i, list(cn))
        rows.append(dict(position=atoms.positions[i], vector=_unit(vacancy), kind="open_metal",
                         label=f"{symbols[i]}(CN={n})", owner=owner(i), atom=i))
    return FeatureSet.from_rows(rows, atoms.cell[:], atoms.pbc)


def _cluster_vacancy(atoms, i, metals):
    """Ligands too symmetric to point at the gap: the direction away from
    the neighbouring metals of the same cluster."""
    near = [m for m in metals if m != i and np.linalg.norm(_vec(atoms, i, m)) < 4.0]
    return _unit(-sum((_unit(_vec(atoms, i, m)) for m in near), np.zeros(3)))


def orientation_class(cosine, threshold=0.5):
    """``"endo"`` / ``"exo"`` / ``"tangential"`` from the cosine between a
    site bond and the inward direction.

    Examples
    --------
    >>> from cage_isomer_builder.utils.features import orientation_class
    >>> orientation_class(0.9), orientation_class(-0.8), orientation_class(0.1)
    ('endo', 'exo', 'tangential')
    """
    if cosine > threshold:
        return "endo"
    if cosine < -threshold:
        return "exo"
    return "tangential"


def site_orientation(atoms, site_idx, carbon_idx, centre, threshold=0.5):
    """Cosine and class (:func:`orientation_class`) of each site bond with
    respect to ``centre``.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.features import site_orientation
    >>> benzene = molecule("C6H6")                 # H 6 is bonded to C 0
    >>> centre = 3 * benzene.positions[6] - 2 * benzene.positions[0]   # out along C-H
    >>> cosines, classes = site_orientation(benzene, [6], [0], centre)
    >>> classes
    ['endo']
    """
    cosines, classes = [], []
    for s, c in zip(site_idx, carbon_idx):
        bond = _unit(_vec(atoms, c, s))
        inward = _unit(np.asarray(centre) - atoms.positions[c])
        cos = float(bond @ inward)
        cosines.append(cos)
        classes.append(orientation_class(cos, threshold))
    return np.array(cosines), classes


def fg_features(atoms, slot_indices=None, adjacency=None, owners=None, centre=None,
                threshold=0.5, labels=None):
    """
    One ``fg`` feature per R site (slot), in slot order.

    Parameters
    ----------
    slot_indices : sequence of int, optional
        Site atoms in slot order. Default: every site atom of ``atoms``.
    centre : array-like, optional
        Pore centre for endo/exo. Default: for a finite structure, the
        centroid of its non-H, non-site atoms; periodic structures get no
        orientation unless a centre is given.
    labels : sequence of str, optional
        Label per slot (default ``"R<n>"`` from the ``rgroup`` array).

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.sites import mark_site
    >>> from cage_isomer_builder.utils.features import fg_features
    >>> benzene = molecule("C6H6")
    >>> mark_site(benzene, 6, 1)
    >>> mark_site(benzene, 9, 2)
    >>> fg_features(benzene).labels.tolist()
    ['R1', 'R2']
    """
    if slot_indices is None:
        slot_indices = site_indices(atoms)
    slot_indices = [int(i) for i in slot_indices]
    if adjacency is None:
        adjacency = bond_graph(atoms)
    owner = _owner_lookup(owners)
    if centre is None and not atoms.pbc.any():
        slot_set = set(slot_indices)
        heavy = [i for i in range(len(atoms))
                 if atoms.numbers[i] > 1 and i not in slot_set]
        centre = atoms.positions[heavy].mean(axis=0)
    rg = site_array(atoms)
    carbons = []
    for s in slot_indices:
        nb = [j for j in adjacency[s] if atoms.numbers[j] > 1]
        if not nb:
            raise ValueError(f"site atom {s} is not bonded to any heavy atom.")
        carbons.append(nb[0])
    if centre is not None:
        _, classes = site_orientation(atoms, slot_indices, carbons, centre, threshold)
    else:
        classes = [""] * len(slot_indices)
    rows = []
    for k, (s, c) in enumerate(zip(slot_indices, carbons)):
        label = labels[k] if labels is not None else f"R{max(int(rg[s]), 1)}"
        rows.append(dict(position=atoms.positions[s], vector=_unit(_vec(atoms, c, s)),
                         kind="fg", label=label, orientation=classes[k],
                         owner=owner(s), atom=s))
    return FeatureSet.from_rows(rows, atoms.cell[:], atoms.pbc)


def detect_features(atoms, kinds=("pi", "donor", "acceptor", "open_metal"), owners=None,
                    expected_cn=None, include_metal_bound=False, centre=None,
                    threshold=0.5, slot_indices=None, adjacency=None,
                    open_metal_method="auto"):
    """
    All features of the requested kinds (``"fg"`` needs R sites on
    ``atoms``). Works on hosts and guest molecules alike.

    Parameters
    ----------
    atoms : ase.Atoms
    kinds : sequence of str
        Subset of :data:`KINDS`.
    owners : list of list of int, optional
        Atom indices of each building block; sets ``FeatureSet.owners``.
    expected_cn : dict, optional
        ``{element: full coordination number}`` for the ``"coordination"``
        open-metal method.
    open_metal_method : {"auto", "geometry", "coordination"}
        See :func:`open_metal_features`.
    include_metal_bound : bool, default False
        Keep O/N acceptors bonded to both a metal and a non-metal.
    centre, threshold, slot_indices
        For ``fg`` features (see :func:`fg_features`).

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.utils.features import detect_features
    >>> detect_features(example_data.read("ibuprofen")).counts()
    {'acceptor': 2, 'donor': 1, 'pi': 1}
    """
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise ValueError(f"unknown feature kind(s) {sorted(unknown)}; choose from {KINDS}.")
    if adjacency is None:
        adjacency = bond_graph(atoms)
    out = FeatureSet.empty(atoms.cell[:], atoms.pbc)
    if "fg" in kinds and (slot_indices or site_indices(atoms)):
        out = out + fg_features(atoms, slot_indices, adjacency, owners, centre, threshold)
    if "pi" in kinds:
        out = out + pi_features(atoms, adjacency, owners)
    if "donor" in kinds or "acceptor" in kinds:
        hb = hbond_features(atoms, adjacency, owners, include_metal_bound)
        out = out + hb.select(kinds=[k for k in ("donor", "acceptor") if k in kinds])
    if "open_metal" in kinds:
        out = out + open_metal_features(atoms, adjacency, owners, expected_cn,
                                        method=open_metal_method)
    out.cell = np.asarray(atoms.cell[:], dtype=float)
    out.pbc = np.asarray(atoms.pbc, dtype=bool)
    return out
