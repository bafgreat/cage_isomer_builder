#!/usr/bin/env python3
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

"""
Dispersion energies for host-guest placement: DFT-D4 (needs the optional
dftd4 package), or the UFF Lennard-Jones term, which needs nothing extra.

Scoring a guest pose with a full dftd4 call for every candidate is too slow
when thousands of poses are tried. The C6 coefficients of D4 depend on each
atom's coordination number and charge, which barely change when a guest
moves around inside a cage. So they are computed once, by dftd4, for the
host and each guest type placed far apart. A numba kernel then sums the
two-body D4 energy with rational (Becke-Johnson) damping over all
host-guest atom pairs of each pose:

    E = - sum_AB [ s6*C6/(r^6 + R0^6) + s8*C8/(r^8 + R0^8) ]
    C8 = 3*C6*r4r2_A*r4r2_B,   R0 = a1*sqrt(3*r4r2_A*r4r2_B) + a2

The r4r2 values are read from dftd4 itself (see :func:`_r4r2`), so this is
the same two-body energy dftd4 computes.

D4 only describes attraction. On its own it pulls a guest into the host
wall until it hits whatever distance limit is set. So the repulsive half of
the UFF Lennard-Jones term, D*(x/r)^12 with UFF4MOF's per-element x and D,
is added to every pair. Checked against GFN2-xTB on a Tri4Di6 cage: D4 with
only a distance limit put methane 2.0 A from node oxygens, while D4 plus
this repulsion gave 3.1 A contacts and the sites GFN2-xTB rates best. The three-body term and the small
changes in coordination number are left out of this fast score; the final
placements are rescored with a full dftd4 call
(:func:`d4_interaction_energy`).

Without dftd4 (it has no Windows wheels), :func:`pair_model` falls back to
the UFF Lennard-Jones term, E = D*((x/r)^12 - 2*(x/r)^6), with UFF4MOF's
per-element x and D and geometric-mean mixing. It has its own repulsion and
needs no rescoring, since the pair sum is the whole energy.

The caller also supplies a minimum allowed distance for every atom pair,
as a hard safety limit. Closer pairs add a steep penalty, so a search can push a
clashing pose out instead of throwing it away, and the kernel reports how
far inside the limit the worst pair is.
"""

import importlib.util
import warnings
from functools import lru_cache

import numpy as np
from ase import Atoms, units
from numba import njit, prange

#: D4 rational damping parameters per functional (dftd4 parameter file).
D4_PARAMETERS = {
    "pbe": {"s6": 1.0, "s8": 0.95948085, "s9": 1.0, "a1": 0.38574991,
            "a2": 4.80688534, "alp": 16.0},
}

#: Penalty force constant (Hartree/Bohr^2, about 97 eV/A^2) for pairs closer
#: than their minimum distance.
_CLASH_PENALTY = 1.0


def dftd4_available():
    """True if the optional dftd4 package can be imported.

    Examples
    --------
    >>> from cage_isomer_builder.utils.dispersion import dftd4_available
    >>> isinstance(dftd4_available(), bool)
    True
    """
    return importlib.util.find_spec("dftd4") is not None


def _require_dftd4():
    if not dftd4_available():
        raise ImportError(
            "D4 dispersion needs the dftd4 package. Install it with "
            "'pip install cage_isomer_builder[d4]', or use dispersion='uff', "
            "which needs nothing extra."
        )


def _damping(method):
    """Parameter dict for ``method`` (a name in D4_PARAMETERS, or a dict)."""
    if isinstance(method, dict):
        return {"s9": 1.0, "alp": 16.0, **method}
    try:
        return D4_PARAMETERS[method.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown D4 method {method!r}. Use one of {sorted(D4_PARAMETERS)} "
            "or a dict with s6, s8, a1, a2 (and optionally s9)."
        ) from None


@lru_cache(maxsize=None)
def _r4r2(atomic_number):
    """
    The D4 r4r2 value of an element, read from dftd4.

    For a dimer with zero damping radius (a1 = a2 = 0) and only the C8 term,
    dftd4 returns E = -3*C6*r4r2^2 / r^8, and C6 comes from the same model.
    """
    from dftd4.interface import DampingParam, DispersionModel

    distance = 20.0  # Bohr
    numbers = np.array([atomic_number, atomic_number])
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
    model = DispersionModel(numbers, positions)
    c6 = model.get_properties()["c6 coefficients"][0, 1]
    energy = model.get_dispersion(
        DampingParam(s6=0.0, s8=1.0, s9=0.0, a1=0.0, a2=0.0, alp=16.0), grad=False
    )["energy"]
    return float(np.sqrt(-energy * distance ** 8 / (3.0 * c6)))


class D4PairModel:
    """
    Fixed C6 coefficients and r4r2 values for a host and its guest types.

    Atoms are indexed in one list: the host first, then each guest type in
    order. ``offsets[t]`` is where guest type ``t`` starts.

    Parameters
    ----------
    host : ase.Atoms
    guests : sequence of ase.Atoms
    method : str or dict, default "pbe"
        D4 damping parameters, see :data:`D4_PARAMETERS`.

    Examples
    --------
    >>> import numpy as np
    >>> from ase.build import molecule
    >>> a, b = molecule("CH4"), molecule("CH4")
    >>> b.translate([4.0, 0.0, 0.0])                    # methane dimer, C...C = 4 A
    >>> from cage_isomer_builder.utils.dispersion import D4PairModel   # needs dftd4
    >>> model = D4PairModel(a, [b])
    >>> model.name, model.offsets
    ('D4(pbe)+UFF repulsion', [0, 5])
    """

    def __init__(self, host, guests, method="pbe"):
        _require_dftd4()
        from dftd4.interface import DispersionModel

        self.name = f"D4({method})+UFF repulsion" if isinstance(method, str) else "D4+UFF repulsion"
        self.method = method
        self.params = _damping(method)
        # Place each fragment far from the others so their coordination
        # numbers are those of the isolated molecules.
        fragments = [host] + list(guests)
        combined = Atoms()
        self.offsets = []
        for n, fragment in enumerate(fragments):
            self.offsets.append(len(combined))
            moved = fragment.copy()
            moved.translate(-moved.get_center_of_mass() + [1000.0 * n, 0.0, 0.0])
            combined += moved
        model = DispersionModel(combined.numbers, combined.positions / units.Bohr)
        self.c6 = np.ascontiguousarray(model.get_properties()["c6 coefficients"])
        self.r4r2 = np.array([_r4r2(int(z)) for z in combined.numbers])
        self.uff_x, self.uff_d = _uff_element_params(combined.numbers)

    def score(self, source_existing, existing_positions, source_guest,
              guest_positions, min_distance):
        """
        Two-body D4 energy plus UFF repulsion of each guest pose with the
        existing atoms, and the clash penalty.

        Parameters
        ----------
        source_existing : np.ndarray of int, shape (n_e,)
            Index of each existing atom in this model's atom list.
        existing_positions : np.ndarray, shape (n_e, 3)
        source_guest : np.ndarray of int, shape (n_g,)
            Index of each guest atom in this model's atom list.
        guest_positions : np.ndarray, shape (n_poses, n_g, 3)
        min_distance : np.ndarray, shape (n_e, n_g)
            Smallest allowed distance (Angstrom) for each pair.

        Returns
        -------
        energy : np.ndarray, shape (n_poses,)
            D4 plus repulsion energy (eV).
        penalty : np.ndarray, shape (n_poses,)
            Clash penalty (eV), zero when no pair is too close.
        overlap : np.ndarray, shape (n_poses,)
            How far (Angstrom) the worst pair is inside its minimum distance.

        Examples
        --------
        >>> import numpy as np
        >>> from ase.build import molecule
        >>> a, b = molecule("CH4"), molecule("CH4")
        >>> b.translate([4.0, 0.0, 0.0])                    # methane dimer, C...C = 4 A
        >>> from cage_isomer_builder.utils.dispersion import D4PairModel
        >>> model = D4PairModel(a, [b])
        >>> host, guest = np.arange(5), model.offsets[1] + np.arange(5)
        >>> energy, penalty, overlap = model.score(host, a.positions, guest, b.positions[None],
        ...                                        min_distance=np.zeros((5, 5)))
        >>> bool(energy[0] < 0), float(penalty[0])          # attractive, no clash
        (True, 0.0)
        """
        p = self.params
        x = np.sqrt(self.uff_x[source_existing][:, None] * self.uff_x[source_guest][None, :])
        d = np.sqrt(self.uff_d[source_existing][:, None] * self.uff_d[source_guest][None, :])
        energy, penalty, overlap = _score_poses(
            np.ascontiguousarray(existing_positions / units.Bohr),
            np.ascontiguousarray(guest_positions / units.Bohr),
            np.ascontiguousarray(self.c6[np.ix_(source_existing, source_guest)]),
            self.r4r2[source_existing],
            self.r4r2[source_guest],
            np.ascontiguousarray(min_distance / units.Bohr),
            np.ascontiguousarray(x / units.Bohr),
            np.ascontiguousarray(d / units.Hartree),
            p["s6"], p["s8"], p["a1"], p["a2"], _CLASH_PENALTY,
        )
        return energy * units.Hartree, penalty * units.Hartree, overlap * units.Bohr

    def interaction_energy(self, fragments):
        """
        Full D4 interaction energy (see d4_interaction_energy) plus the UFF
        repulsion between the fragments, in eV.

        Examples
        --------
        >>> import numpy as np
        >>> from ase.build import molecule
        >>> a, b = molecule("CH4"), molecule("CH4")
        >>> b.translate([4.0, 0.0, 0.0])                    # methane dimer, C...C = 4 A
        >>> from cage_isomer_builder.utils.dispersion import D4PairModel
        >>> energy = D4PairModel(a, [b]).interaction_energy([a, b])
        >>> bool(-0.05 < energy < 0)                         # eV: a weak dispersion contact
        True
        """
        return (d4_interaction_energy(fragments, self.method)
                + uff_interaction_energy(fragments, repulsion_only=True))


@njit(parallel=True, cache=True)
def _score_poses(existing, poses, c6, r4r2_e, r4r2_g, min_dist, rep_x, rep_d,
                 s6, s8, a1, a2, k_clash):
    n_poses = poses.shape[0]
    n_e = existing.shape[0]
    n_g = poses.shape[1]
    energy = np.zeros(n_poses)
    penalty = np.zeros(n_poses)
    overlap = np.zeros(n_poses)
    for p in prange(n_poses):
        e = 0.0
        pen = 0.0
        worst = 0.0
        for b in range(n_g):
            bx = poses[p, b, 0]
            by = poses[p, b, 1]
            bz = poses[p, b, 2]
            for a in range(n_e):
                dx = existing[a, 0] - bx
                dy = existing[a, 1] - by
                dz = existing[a, 2] - bz
                r2 = dx * dx + dy * dy + dz * dz
                limit = min_dist[a, b]
                if r2 < limit * limit:
                    depth = limit - np.sqrt(r2)
                    pen += k_clash * depth * depth
                    if depth > worst:
                        worst = depth
                q = 3.0 * r4r2_e[a] * r4r2_g[b]
                r0 = a1 * np.sqrt(q) + a2
                r0_2 = r0 * r0
                r0_6 = r0_2 * r0_2 * r0_2
                r6 = r2 * r2 * r2
                e -= c6[a, b] * (s6 / (r6 + r0_6) + s8 * q / (r6 * r2 + r0_6 * r0_2))
                x6 = (rep_x[a, b] * rep_x[a, b] / r2) ** 3
                e += rep_d[a, b] * x6 * x6
        energy[p] = e
        penalty[p] = pen
        overlap[p] = worst
    return energy, penalty, overlap


def d4_interaction_energy(fragments, method="pbe"):
    """
    Full D4 interaction energy (eV), including the three-body term:
    E(all fragments together) - sum of E(each fragment alone).

    Parameters
    ----------
    fragments : sequence of ase.Atoms
        For example the host followed by each placed guest.
    method : str or dict, default "pbe"

    Examples
    --------
    >>> import numpy as np
    >>> from ase.build import molecule
    >>> a, b = molecule("CH4"), molecule("CH4")
    >>> b.translate([4.0, 0.0, 0.0])                    # methane dimer, C...C = 4 A
    >>> from cage_isomer_builder.utils.dispersion import d4_interaction_energy
    >>> far = b.copy()
    >>> far.translate([50.0, 0.0, 0.0])
    >>> bool(d4_interaction_energy([a, b]) < 0), round(float(d4_interaction_energy([a, far])), 6)
    (True, 0.0)
    """
    _require_dftd4()
    from dftd4.interface import DampingParam, DispersionModel

    param = DampingParam(**_damping(method))

    def energy(atoms):
        model = DispersionModel(atoms.numbers, atoms.positions / units.Bohr)
        return model.get_dispersion(param, grad=False)["energy"]

    combined = Atoms()
    for fragment in fragments:
        combined += fragment
    total = energy(combined) - sum(energy(f) for f in fragments)
    return total * units.Hartree


def _uff_element_params(numbers):
    """UFF4MOF van der Waals x (Angstrom) and D (eV) of each atom, by element."""
    from ase.data import chemical_symbols

    from cage_isomer_builder.utils.uff4mof_params import _element_symbol, load_uff4mof_db

    by_element = {}
    for type_name, params in load_uff4mof_db().items():
        by_element.setdefault(_element_symbol(type_name), params)
    x = np.empty(len(numbers))
    d = np.empty(len(numbers))
    for i, z in enumerate(numbers):
        params = by_element.get(chemical_symbols[z])
        if params is None:
            raise ValueError(f"No UFF van der Waals parameters for {chemical_symbols[z]}.")
        x[i] = params.x1
        d[i] = params.D1 * units.kcal / units.mol
    return x, d


class UFFPairModel:
    """
    UFF Lennard-Jones pair energies for a host and its guest types, with the
    same interface as :class:`D4PairModel`. Needs no optional packages.

    Atoms are indexed in one list: the host first, then each guest type in
    order. ``offsets[t]`` is where guest type ``t`` starts.

    Examples
    --------
    >>> import numpy as np
    >>> from ase.build import molecule
    >>> a, b = molecule("CH4"), molecule("CH4")
    >>> b.translate([4.0, 0.0, 0.0])                    # methane dimer, C...C = 4 A
    >>> from cage_isomer_builder.utils.dispersion import UFFPairModel
    >>> model = UFFPairModel(a, [b])
    >>> model.name, [int(o) for o in model.offsets]
    ('UFF', [0, 5])
    """

    name = "UFF"

    def __init__(self, host, guests):
        numbers = np.concatenate([host.numbers] + [g.numbers for g in guests])
        self.offsets = list(np.cumsum([0, len(host)] + [len(g) for g in guests])[:-1])
        self.x, self.d = _uff_element_params(numbers)

    def score(self, source_existing, existing_positions, source_guest,
              guest_positions, min_distance):
        """
        Lennard-Jones energy of each guest pose with the existing atoms, plus
        the clash penalty. Same arguments and returns as
        :meth:`D4PairModel.score`, in eV and Angstrom.

        Examples
        --------
        >>> import numpy as np
        >>> from ase.build import molecule
        >>> a, b = molecule("CH4"), molecule("CH4")
        >>> b.translate([4.0, 0.0, 0.0])                    # methane dimer, C...C = 4 A
        >>> from cage_isomer_builder.utils.dispersion import UFFPairModel, uff_interaction_energy
        >>> model = UFFPairModel(a, [b])
        >>> host, guest = np.arange(5), model.offsets[1] + np.arange(5)
        >>> energy, penalty, overlap = model.score(host, a.positions, guest, b.positions[None],
        ...                                        min_distance=np.zeros((5, 5)))
        >>> bool(np.isclose(energy[0], uff_interaction_energy([a, b]))), float(penalty[0])
        (True, 0.0)
        """
        x = np.sqrt(self.x[source_existing][:, None] * self.x[source_guest][None, :])
        d = np.sqrt(self.d[source_existing][:, None] * self.d[source_guest][None, :])
        k_clash = _CLASH_PENALTY * units.Hartree / units.Bohr ** 2
        return _score_poses_lj(
            np.ascontiguousarray(existing_positions, dtype=float),
            np.ascontiguousarray(guest_positions, dtype=float),
            np.ascontiguousarray(x), np.ascontiguousarray(d),
            np.ascontiguousarray(min_distance, dtype=float), k_clash,
        )

    def interaction_energy(self, fragments):
        """Lennard-Jones energy (eV) between all pairs of fragments.

        Examples
        --------
        >>> import numpy as np
        >>> from ase.build import molecule
        >>> a, b = molecule("CH4"), molecule("CH4")
        >>> b.translate([4.0, 0.0, 0.0])                    # methane dimer, C...C = 4 A
        >>> from cage_isomer_builder.utils.dispersion import UFFPairModel, uff_interaction_energy
        >>> UFFPairModel(a, [b]).interaction_energy([a, b]) == uff_interaction_energy([a, b])
        True
        """
        return uff_interaction_energy(fragments)


@njit(parallel=True, cache=True)
def _score_poses_lj(existing, poses, x, d, min_dist, k_clash):
    n_poses = poses.shape[0]
    n_e = existing.shape[0]
    n_g = poses.shape[1]
    energy = np.zeros(n_poses)
    penalty = np.zeros(n_poses)
    overlap = np.zeros(n_poses)
    for p in prange(n_poses):
        e = 0.0
        pen = 0.0
        worst = 0.0
        for b in range(n_g):
            for a in range(n_e):
                dx = existing[a, 0] - poses[p, b, 0]
                dy = existing[a, 1] - poses[p, b, 1]
                dz = existing[a, 2] - poses[p, b, 2]
                r2 = dx * dx + dy * dy + dz * dz
                limit = min_dist[a, b]
                if r2 < limit * limit:
                    depth = limit - np.sqrt(r2)
                    pen += k_clash * depth * depth
                    if depth > worst:
                        worst = depth
                s6 = (x[a, b] * x[a, b] / r2) ** 3
                e += d[a, b] * (s6 * s6 - 2.0 * s6)
        energy[p] = e
        penalty[p] = pen
        overlap[p] = worst
    return energy, penalty, overlap


def uff_interaction_energy(fragments, repulsion_only=False):
    """
    UFF Lennard-Jones interaction energy (eV): the sum over every atom pair
    that belongs to two different fragments. With ``repulsion_only``, only
    the repulsive D*(x/r)^12 part.

    Examples
    --------
    >>> import numpy as np
    >>> from ase.build import molecule
    >>> a, b = molecule("CH4"), molecule("CH4")
    >>> b.translate([4.0, 0.0, 0.0])                    # methane dimer, C...C = 4 A
    >>> from cage_isomer_builder.utils.dispersion import uff_interaction_energy
    >>> bool(uff_interaction_energy([a, b]) < 0 < uff_interaction_energy([a, b], repulsion_only=True))
    True
    """
    total = 0.0
    params = [_uff_element_params(f.numbers) for f in fragments]
    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            r = np.linalg.norm(
                fragments[i].positions[:, None, :] - fragments[j].positions[None, :, :], axis=2
            )
            x = np.sqrt(params[i][0][:, None] * params[j][0][None, :])
            d = np.sqrt(params[i][1][:, None] * params[j][1][None, :])
            attraction = 0.0 if repulsion_only else 2 * (x / r) ** 6
            total += float(np.sum(d * ((x / r) ** 12 - attraction)))
    return total


def pair_model(host, guests, dispersion="auto"):
    """
    The pair model for ``dispersion``:

    * ``"auto"``: D4 with PBE parameters (plus UFF repulsion) if dftd4 is
      installed, otherwise UFF Lennard-Jones (with a warning).
    * ``"uff"``: UFF Lennard-Jones.
    * a D4 method name (e.g. ``"pbe"``) or a dict of D4 damping parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from ase.build import molecule
    >>> a, b = molecule("CH4"), molecule("CH4")
    >>> b.translate([4.0, 0.0, 0.0])                    # methane dimer, C...C = 4 A
    >>> from cage_isomer_builder.utils.dispersion import pair_model
    >>> pair_model(a, [b], dispersion="uff").name
    'UFF'
    """
    if isinstance(dispersion, str) and dispersion.lower() == "auto":
        if dftd4_available():
            return D4PairModel(host, guests, "pbe")
        warnings.warn(
            "dftd4 is not installed, so guest sites are scored with the UFF "
            "Lennard-Jones term instead of D4. Install it with "
            "'pip install cage_isomer_builder[d4]' (no Windows wheels), or pass "
            "dispersion='uff' to silence this warning."
        )
        return UFFPairModel(host, guests)
    if isinstance(dispersion, str) and dispersion.lower() == "uff":
        return UFFPairModel(host, guests)
    return D4PairModel(host, guests, dispersion)
