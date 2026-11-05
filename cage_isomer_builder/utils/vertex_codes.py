#!/usr/bin/env python3
"""
Vertex codes: name a one-group-per-linker isomer by what each vertex (node)
sees of the groups on its linkers.

Every linker joins two vertices and has four sites, numbered as in a
para-phenylene drawn with this vertex at the bottom and seen from the pore
centre: 1 (bottom left), 2 (top left), 3 (top right), 4 (bottom right).
Seen from one vertex, a linker reads

* ``1``: its group is next to this vertex, on the left,
* ``4``: next to this vertex, on the right,
* ``0``: no group, or the group is next to the other vertex (2 and 3 never
  occur, since every vertex reads its linkers with itself at the bottom).

A vertex code lists its linkers clockwise as seen from the pore centre.
Two forms are used:

* the **anchored code** starts at the vertex's lowest-numbered linker. The
  vertex name of an isomer, its anchored codes in vertex order joined by
  ``-`` (e.g. ``000-100-141-401``), is unique and converts back to the
  site indices exactly;
* the **environment label** is the rotation of the code that sorts first
  with 1 < 4 < 0 (e.g. ``114``, ``140``, ``104``). It names the vertex
  environment whatever linker it starts at, so different isomers can share
  the same labels.
"""

from dataclasses import dataclass
from itertools import product
from typing import Sequence, Union

import numpy as np

DIGITS = "014"
_ORDER = {"1": 0, "4": 1, "0": 2}


def environment_label(code: str) -> str:
    """
    The environment label of a vertex code: its rotation that sorts first
    with 1 < 4 < 0.

    Examples
    --------
    >>> from cage_isomer_builder.utils.vertex_codes import environment_label
    >>> environment_label("141"), environment_label("401"), environment_label("041")
    ('114', '140', '104')
    """
    _check_code(code)
    rotations = [code[i:] + code[:i] for i in range(len(code))]
    return min(rotations, key=lambda r: [_ORDER[c] for c in r])


def mirror_label(code: str) -> str:
    """
    Environment label of the mirror image of a vertex: left and right swap
    (1 <-> 4) and the linkers are read the other way round.

    Examples
    --------
    >>> from cage_isomer_builder.utils.vertex_codes import mirror_label
    >>> mirror_label("114"), mirror_label("110"), mirror_label("140")
    ('144', '440', '140')
    """
    _check_code(code)
    swapped = code[::-1].translate(str.maketrans("14", "41"))
    return environment_label(swapped)


def environments(n_linkers: int = 3) -> list:
    """
    Every environment label of a vertex with ``n_linkers`` linkers, sorted
    with 1 < 4 < 0.

    Examples
    --------
    >>> from cage_isomer_builder.utils.vertex_codes import environments
    >>> labels = environments(3)
    >>> len(labels), labels
    (11, ['111', '114', '110', '144', '140', '104', '100', '444', '440', '400', '000'])
    """
    labels = {environment_label("".join(w)) for w in product(DIGITS, repeat=n_linkers)}
    return sorted(labels, key=lambda r: [_ORDER[c] for c in r])


def _check_code(code: str) -> None:
    if not code or set(code) - set(DIGITS):
        raise ValueError(f"vertex code {code!r} must be made of the digits 0, 1 and 4.")


@dataclass(frozen=True)
class VertexScheme:
    """
    How the sites of a cage map onto vertex codes.

    Attributes
    ----------
    vertex_linkers : tuple of tuple of int
        For each vertex, its linkers clockwise as seen from the pore centre,
        starting at the lowest-numbered one.
    linker_sites : tuple of tuple of int
        The site numbers of each linker.
    site_codes : dict
        ``site -> (vertex, digit)``: the vertex the site is next to and its
        digit there (1 left, 4 right).

    Examples
    --------
    Two vertices joined by two linkers, sites 0-3 and 4-7:

    >>> from cage_isomer_builder.utils.vertex_codes import VertexScheme
    >>> scheme = VertexScheme(
    ...     vertex_linkers=((0, 1), (0, 1)),
    ...     linker_sites=((0, 1, 2, 3), (4, 5, 6, 7)),
    ...     site_codes={0: (0, 1), 1: (0, 4), 2: (1, 1), 3: (1, 4),
    ...                 4: (0, 1), 5: (0, 4), 6: (1, 1), 7: (1, 4)})
    >>> scheme.name([1, 6])
    '40-01'
    >>> scheme.isomer("40-01")
    [1, 6]
    >>> scheme.resolve("1-6"), scheme.resolve("40-01")   # either name gives the isomer
    ([1, 6], [1, 6])
    """

    vertex_linkers: tuple
    linker_sites: tuple
    site_codes: dict

    @property
    def n_vertices(self) -> int:
        """
        Number of vertices.

        Examples
        --------
        >>> from cage_isomer_builder.utils.vertex_codes import VertexScheme
        >>> VertexScheme(((0, 1), (0, 1)), ((0, 1, 2, 3), (4, 5, 6, 7)), {}).n_vertices
        2
        """
        return len(self.vertex_linkers)

    def _site_of_linker(self, isomer: Sequence[int]) -> dict:
        linker_of = {s: L for L, sites in enumerate(self.linker_sites) for s in sites}
        chosen = {}
        for s in isomer:
            s = int(s)
            if s not in linker_of:
                raise ValueError(f"site {s} is not a site of this cage.")
            L = linker_of[s]
            if L in chosen:
                raise ValueError(f"sites {chosen[L]} and {s} are both on linker {L}: "
                                 "vertex codes need at most one group per linker.")
            chosen[L] = s
        return chosen

    def codes(self, isomer: Sequence[int]) -> list:
        """
        Anchored code of every vertex, in vertex order.

        Examples
        --------
        >>> from cage_isomer_builder.utils.vertex_codes import VertexScheme
        >>> scheme = VertexScheme(((0, 1), (0, 1)), ((0, 1, 2, 3), (4, 5, 6, 7)),
        ...                       {0: (0, 1), 1: (0, 4), 2: (1, 1), 3: (1, 4),
        ...                        4: (0, 1), 5: (0, 4), 6: (1, 1), 7: (1, 4)})
        >>> scheme.codes([1, 6])
        ['40', '01']
        """
        chosen = self._site_of_linker(isomer)
        digit = {}
        for L, s in chosen.items():
            v, d = self.site_codes[s]
            digit[v, L] = d
        return ["".join(str(digit.get((v, L), 0)) for L in linkers)
                for v, linkers in enumerate(self.vertex_linkers)]

    def name(self, isomer: Sequence[int]) -> str:
        """
        Vertex name: the anchored codes joined by ``-``. Unique for every
        placement, and :meth:`isomer` converts it back.

        Examples
        --------
        >>> from cage_isomer_builder.utils.vertex_codes import VertexScheme
        >>> scheme = VertexScheme(((0, 1), (0, 1)), ((0, 1, 2, 3), (4, 5, 6, 7)),
        ...                       {0: (0, 1), 1: (0, 4), 2: (1, 1), 3: (1, 4),
        ...                        4: (0, 1), 5: (0, 4), 6: (1, 1), 7: (1, 4)})
        >>> scheme.name([0, 4])
        '11-00'
        """
        return "-".join(self.codes(isomer))

    def labels(self, isomer: Sequence[int]) -> list:
        """
        Environment label of every vertex, in vertex order (see
        :func:`environment_label`).

        Examples
        --------
        >>> from cage_isomer_builder.utils.vertex_codes import VertexScheme
        >>> scheme = VertexScheme(((0, 1), (0, 1)), ((0, 1, 2, 3), (4, 5, 6, 7)),
        ...                       {0: (0, 1), 1: (0, 4), 2: (1, 1), 3: (1, 4),
        ...                        4: (0, 1), 5: (0, 4), 6: (1, 1), 7: (1, 4)})
        >>> scheme.labels([1, 6])
        ['40', '10']
        """
        return [environment_label(c) for c in self.codes(isomer)]

    def is_vertex_name(self, name: str) -> bool:
        """
        True if ``name`` has the form of a vertex name of this cage (one code
        per vertex, one digit per linker, digits 0, 1 and 4).

        Examples
        --------
        >>> from cage_isomer_builder.utils.vertex_codes import VertexScheme
        >>> scheme = VertexScheme(((0, 1), (0, 1)), ((0, 1, 2, 3), (4, 5, 6, 7)),
        ...                       {0: (0, 1), 1: (0, 4), 2: (1, 1), 3: (1, 4),
        ...                        4: (0, 1), 5: (0, 4), 6: (1, 1), 7: (1, 4)})
        >>> scheme.is_vertex_name("40-01"), scheme.is_vertex_name("1-6")
        (True, False)
        """
        parts = str(name).split("-")
        return (len(parts) == self.n_vertices
                and all(len(p) == len(ls) and not set(p) - set(DIGITS)
                        for p, ls in zip(parts, self.vertex_linkers)))

    def isomer(self, name: str) -> list:
        """
        Site indices (sorted) of the isomer with this vertex name.

        Raises
        ------
        ValueError
            If ``name`` is not a vertex name of this cage, or a linker has a
            group at both ends.

        Examples
        --------
        >>> from cage_isomer_builder.utils.vertex_codes import VertexScheme
        >>> scheme = VertexScheme(((0, 1), (0, 1)), ((0, 1, 2, 3), (4, 5, 6, 7)),
        ...                       {0: (0, 1), 1: (0, 4), 2: (1, 1), 3: (1, 4),
        ...                        4: (0, 1), 5: (0, 4), 6: (1, 1), 7: (1, 4)})
        >>> scheme.isomer("11-00")
        [0, 4]
        """
        if not self.is_vertex_name(name):
            raise ValueError(
                f"{name!r} is not a vertex name of this cage: expected {self.n_vertices} "
                f"codes of {[len(ls) for ls in self.vertex_linkers]} digits (0, 1, 4).")
        lookup = {(L, v, d): s for L, sites in enumerate(self.linker_sites)
                  for s in sites for v, d in [self.site_codes[s]]}
        chosen = {}
        for v, (code, linkers) in enumerate(zip(name.split("-"), self.vertex_linkers)):
            for c, L in zip(code, linkers):
                if c == "0":
                    continue
                if L in chosen:
                    raise ValueError(f"{name!r} puts two groups on linker {L}.")
                chosen[L] = lookup[L, v, int(c)]
        return sorted(chosen.values())

    def resolve(self, name: Union[str, Sequence[int]]) -> list:
        """
        Site indices of an isomer given by its vertex name
        (``"000-100-141-401"``), its site name (``"0-4-9-13-16-22"``) or its
        list of sites.

        Examples
        --------
        >>> from cage_isomer_builder.utils.vertex_codes import VertexScheme
        >>> scheme = VertexScheme(((0, 1), (0, 1)), ((0, 1, 2, 3), (4, 5, 6, 7)),
        ...                       {0: (0, 1), 1: (0, 4), 2: (1, 1), 3: (1, 4),
        ...                        4: (0, 1), 5: (0, 4), 6: (1, 1), 7: (1, 4)})
        >>> scheme.resolve([6, 1]), scheme.resolve("1-6"), scheme.resolve("40-01")
        ([1, 6], [1, 6], [1, 6])
        """
        if not isinstance(name, str):
            sites = sorted(int(s) for s in name)
        elif self.is_vertex_name(name):
            return self.isomer(name)
        else:
            try:
                sites = sorted(int(s) for s in name.split("-"))
            except ValueError:
                raise ValueError(f"{name!r} is neither a vertex name nor a site name.") from None
        self._site_of_linker(sites)
        return sites

    def to_info(self) -> dict:
        """
        The scheme as text entries for a structure's ``info``, so an isomer
        file carries what it takes to convert between its names
        (:meth:`from_info` reads them back).

        * ``vertex_order``: each vertex's linkers in reading order
          (``"0:0,1,2 1:0,4,3"``: vertex 1 reads linker 0, then 4, then 3),
        * ``linker_sites``: the sites of each linker (``"0:0-3 1:4-7"``),
        * ``site_codes``: ``site:vertex/digit`` for every site
          (``"9:2/1"``: site 9 is next to vertex 2, on the left).

        Examples
        --------
        >>> from cage_isomer_builder.utils.vertex_codes import VertexScheme
        >>> scheme = VertexScheme(((0, 1), (0, 1)), ((0, 1, 2, 3), (4, 5, 6, 7)),
        ...                       {0: (0, 1), 1: (0, 4), 2: (1, 1), 3: (1, 4),
        ...                        4: (0, 1), 5: (0, 4), 6: (1, 1), 7: (1, 4)})
        >>> info = scheme.to_info()
        >>> info["vertex_order"], info["linker_sites"]
        ('0:0,1 1:0,1', '0:0-3 1:4-7')
        >>> info["site_codes"]
        '0:0/1 1:0/4 2:1/1 3:1/4 4:0/1 5:0/4 6:1/1 7:1/4'
        >>> VertexScheme.from_info(info) == scheme
        True
        """
        return {
            "vertex_order": " ".join(f"{v}:{','.join(map(str, ls))}"
                                     for v, ls in enumerate(self.vertex_linkers)),
            "linker_sites": " ".join(f"{L}:{_format_ranges(ss)}"
                                     for L, ss in enumerate(self.linker_sites)),
            "site_codes": " ".join(f"{s}:{v}/{d}"
                                   for s, (v, d) in sorted(self.site_codes.items())),
        }

    @classmethod
    def from_info(cls, info: dict) -> "VertexScheme":
        """
        Read the scheme back from a structure's ``info`` (see
        :meth:`to_info`), e.g. from an isomer file written by
        :meth:`~cage_isomer_builder.cage.base.CageBuilder.enumerate_isomers`.

        Raises
        ------
        ValueError
            If ``info`` has no vertex scheme or it cannot be read.

        Examples
        --------
        >>> from cage_isomer_builder.utils.vertex_codes import VertexScheme
        >>> scheme = VertexScheme.from_info({
        ...     "vertex_order": "0:0,1 1:0,1", "linker_sites": "0:0-3 1:4-7",
        ...     "site_codes": "0:0/1 1:0/4 2:1/1 3:1/4 4:0/1 5:0/4 6:1/1 7:1/4"})
        >>> scheme.name([1, 6])
        '40-01'
        >>> VertexScheme.from_info({"isomer": "1-6"})
        Traceback (most recent call last):
        ...
        ValueError: no vertex scheme (vertex_order, linker_sites, site_codes) in this structure: ...
        """
        missing = [k for k in ("vertex_order", "linker_sites", "site_codes") if k not in info]
        if missing:
            raise ValueError(
                f"no vertex scheme ({', '.join(missing)}) in this structure: write it with "
                "enumerate_isomers() from a cage that has vertex codes.")
        try:
            vertex_linkers = {}
            for item in str(info["vertex_order"]).split():
                v, ls = item.split(":")
                vertex_linkers[int(v)] = tuple(int(L) for L in ls.split(",") if L)
            linker_sites = {}
            for item in str(info["linker_sites"]).split():
                L, ss = item.split(":")
                linker_sites[int(L)] = _parse_ranges(ss)
            site_codes = {}
            for item in str(info["site_codes"]).split():
                s, code = item.split(":")
                v, d = code.split("/")
                site_codes[int(s)] = (int(v), int(d))
            return cls(vertex_linkers=tuple(vertex_linkers[v] for v in range(len(vertex_linkers))),
                       linker_sites=tuple(linker_sites[L] for L in range(len(linker_sites))),
                       site_codes=site_codes)
        except (ValueError, KeyError) as exc:
            raise ValueError(f"cannot read the vertex scheme of this structure: {exc}") from None


def _format_ranges(sites) -> str:
    """``(0, 1, 2, 3, 7)`` -> ``"0-3,7"``."""
    sites = sorted(int(s) for s in sites)
    runs = []
    for s in sites:
        if runs and s == runs[-1][1] + 1:
            runs[-1][1] = s
        else:
            runs.append([s, s])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _parse_ranges(text: str) -> tuple:
    """``"0-3,7"`` -> ``(0, 1, 2, 3, 7)``."""
    sites = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        sites.extend(range(int(a), int(b or a) + 1))
    return tuple(sites)


def scheme_from_geometry(site_positions, linker_sites, linker_vertices, linker_centres,
                         vertex_centres, centre, tol: float = 0.2) -> VertexScheme:
    """
    Work out the vertex scheme of a cage from its geometry.

    Parameters
    ----------
    site_positions : array, shape (n_sites, 3)
    linker_sites : sequence of sequence of int
        The sites of each linker.
    linker_vertices : sequence of (int, int)
        The two vertices each linker joins.
    linker_centres, vertex_centres : array, shape (n, 3)
    centre : array, shape (3,)
        Pore centre: left, right and clockwise are as seen from here.
    tol : float, default 0.2
        Smallest sine of the angle that still tells left from right, or one
        linker from the next around a vertex.

    Raises
    ------
    ValueError
        If a linker does not have one site on each side at each end, or the
        linkers of a vertex cannot be put in order around it.

    Examples
    --------
    One linker along z from vertex 0 (bottom) to vertex 1, its ring facing a
    pore centre at -x. Looking from there with vertex 0 at the bottom, -y
    is on the right; turned round for vertex 1, +y is:

    >>> import numpy as np
    >>> from cage_isomer_builder.utils.vertex_codes import scheme_from_geometry
    >>> sites = np.array([[0, -1, -1], [0, -1, 1], [0, 1, 1], [0, 1, -1]], float)
    >>> scheme = scheme_from_geometry(sites, [[0, 1, 2, 3]], [(0, 1)], [[0, 0, 0]],
    ...                               [[0, 0, -3], [0, 0, 3]], [-5, 0, 0])
    >>> scheme.site_codes
    {0: (0, 4), 1: (1, 1), 2: (1, 4), 3: (0, 1)}
    """
    site_positions = np.asarray(site_positions, float)
    linker_centres = np.asarray(linker_centres, float)
    vertex_centres = np.asarray(vertex_centres, float)
    centre = np.asarray(centre, float)
    n_vertices = len(vertex_centres)

    for L, sites in enumerate(linker_sites):
        if len(sites) != 4:
            raise ValueError(
                f"linker {L} has {len(sites)} sites; vertex codes need four per linker "
                "(one ring, as in a para-phenylene).")

    site_codes = {}
    for L, (sites, (va, vb)) in enumerate(zip(linker_sites, linker_vertices)):
        lc = linker_centres[L]
        out = lc - centre
        seen = set()
        for s in sites:
            p = site_positions[s]
            near, far = ((va, vb) if np.linalg.norm(p - vertex_centres[va])
                         < np.linalg.norm(p - vertex_centres[vb]) else (vb, va))
            up = vertex_centres[far] - vertex_centres[near]
            r = p - lc
            side = np.dot(np.cross(up, r), out) / (
                np.linalg.norm(up) * np.linalg.norm(r) * np.linalg.norm(out))
            if abs(side) < tol:
                raise ValueError(
                    f"site {s} of linker {L} lies in the plane through the linker and the "
                    "pore centre, so it is neither left nor right.")
            digit = 4 if side > 0 else 1
            site_codes[int(s)] = (int(near), digit)
            seen.add((near, digit))
        if len(sites) != 4 or len(seen) != 4:
            raise ValueError(
                f"linker {L} needs four sites, one left and one right at each end, "
                f"for vertex codes; it has {len(sites)} ({sorted(seen)}).")

    vertex_linkers = []
    for v in range(n_vertices):
        linkers = sorted(L for L, pair in enumerate(linker_vertices) if v in pair)
        if not linkers:
            vertex_linkers.append(())
            continue
        axis = vertex_centres[v] - centre
        axis /= np.linalg.norm(axis)
        flat = {}
        for L in linkers:
            w = linker_centres[L] - vertex_centres[v]
            w = w - np.dot(w, axis) * axis
            flat[L] = w / np.linalg.norm(w)
        e1 = flat[linkers[0]]
        e2 = np.cross(axis, e1)
        # Counter-clockwise seen from outside is clockwise seen from the pore centre.
        angle = {L: np.arctan2(flat[L] @ e2, flat[L] @ e1) % (2 * np.pi) for L in linkers}
        ordered = sorted(linkers, key=angle.get)
        # The anchor's angle is 0, or just under 2 pi after rounding: start there.
        start = ordered.index(linkers[0])
        ordered = ordered[start:] + ordered[:start]
        gaps = [np.cross(flat[a], flat[b]) @ axis
                for a, b in zip(ordered, ordered[1:] + ordered[:1])]
        if len(linkers) > 2 and min(abs(g) for g in gaps) < tol:
            raise ValueError(f"the linkers of vertex {v} cannot be put in order around it.")
        vertex_linkers.append(tuple(int(L) for L in ordered))

    return VertexScheme(vertex_linkers=tuple(vertex_linkers),
                        linker_sites=tuple(tuple(int(s) for s in ss) for ss in linker_sites),
                        site_codes=site_codes)
