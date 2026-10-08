#!/usr/bin/env python3
"""
Cavity and window geometry of a finite cage, from its atoms.

* :func:`pore_sphere`: the largest sphere that fits inside the cage, i.e. the
  point whose clearance to the nearest atom *surface* (distance minus the
  atom's van der Waals radius) is largest, found by maximising that
  clearance from the cage's centroid. This is the "optimised pore
  diameter" used by cage-analysis codes such as pywindow (Miklitz &
  Jelfs).
* :func:`largest_window`: the widest opening out of the cavity. Rays are
  cast from the pore centre in many directions; a ray that leaves the cage
  without crossing any atom passes through an opening, and the narrowest
  clearance along it is that opening's free radius. The widest such
  opening, as a diameter, is returned. Straight rays give a lower bound on
  a window's free size (a guest could also pass at an angle).

Van der Waals radii are Bondi's where defined, else Alvarez (2013), which
also covers the metals.
"""
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

import numpy as np
from ase.data import vdw_radii as _bondi
from ase.data.vdw_alvarez import vdw_radii as _alvarez
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def vdw_radius(numbers):
    """
    Van der Waals radii (Angstrom): Bondi where defined, else Alvarez, else
    2.0.

    Examples
    --------
    >>> from cage_isomer_builder.utils.cavity import vdw_radius
    >>> vdw_radius([1, 6, 40]).tolist()            # H, C (Bondi), Zr (Alvarez)
    [1.2, 1.7, 2.52]
    """
    numbers = np.asarray(numbers, dtype=int)
    r = np.array(_bondi, dtype=float)[numbers]
    alt = np.array(_alvarez, dtype=float)[numbers]
    r = np.where(np.isnan(r), alt, r)
    return np.where(np.isnan(r), 2.0, r)


def _clearance_function(atoms):
    keep = atoms.numbers > 0
    positions = atoms.positions[keep]
    radii = vdw_radius(atoms.numbers[keep])
    tree = cKDTree(positions)
    k = min(16, len(positions))

    def clearance(points):
        points = np.atleast_2d(points)
        dist, idx = tree.query(points, k=k)
        dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
        return (dist - radii[idx]).min(axis=1)

    return clearance, positions


def pore_sphere(atoms, start=None):
    """
    Centre and diameter of the largest sphere inside a finite cage.

    The clearance (distance to the nearest atom surface) is maximised from
    ``start`` (default: the centroid of the atoms), staying within the
    sphere that is free at the start, so the search cannot leave the cage
    through a window.

    Parameters
    ----------
    atoms : ase.Atoms
        Finite structure (``X`` markers are ignored).
    start : array-like, optional
        Where to start the search.

    Returns
    -------
    centre : np.ndarray, shape (3,)
    diameter : float
        Zero or negative if the start point is inside an atom (no cavity).

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.cavity import pore_sphere
    >>> centre, diameter = pore_sphere(molecule("C60"))
    >>> round(diameter, 2)                       # 2 * (3.51 A cage radius - 1.70 A for C)
    3.62
    """
    clearance, positions = _clearance_function(atoms)
    x0 = positions.mean(axis=0) if start is None else np.asarray(start, dtype=float)
    r0 = float(clearance(x0)[0])
    if r0 <= 0:
        return x0, 2 * r0
    limit = r0

    def objective(x):
        shift = np.linalg.norm(x - x0)
        penalty = 10.0 * max(0.0, shift - limit) ** 2
        return -float(clearance(x)[0]) + penalty

    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    centre = result.x if -result.fun > r0 else x0
    return centre, 2 * float(clearance(centre)[0])


def _fibonacci_directions(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def largest_window(atoms, centre=None, n_directions=2000, step=0.1):
    """
    Free diameter of the widest opening out of a finite cage's cavity (see
    the module docstring), or None if no straight ray escapes (a closed
    cage such as C60).

    Parameters
    ----------
    atoms : ase.Atoms
    centre : array-like, optional
        Default: :func:`pore_sphere`'s centre.
    n_directions : int, default 2000
        Number of ray directions (evenly spread over the sphere).
    step : float, default 0.1
        Sampling step along each ray (Angstrom).

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.cavity import largest_window
    >>> print(largest_window(molecule("C60")))     # closed: no way out
    None
    """
    clearance, positions = _clearance_function(atoms)
    if centre is None:
        centre, _ = pore_sphere(atoms)
    centre = np.asarray(centre, dtype=float)
    reach = np.linalg.norm(positions - centre, axis=1).max() + 3.0
    s = np.arange(0.0, reach, step)
    best = None
    for d in np.array_split(_fibonacci_directions(n_directions), 20):
        points = centre[None, None, :] + s[None, :, None] * d[:, None, :]
        c = clearance(points.reshape(-1, 3)).reshape(len(d), len(s))
        bottleneck = c.min(axis=1)
        escaping = bottleneck > 0
        if escaping.any():
            value = float(bottleneck[escaping].max())
            best = value if best is None else max(best, value)
    return None if best is None else 2 * best
