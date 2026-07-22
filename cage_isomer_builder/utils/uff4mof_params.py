#!/usr/bin/env python3
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

"""
UFF4MOF parameter database and force-field combining-rule math.

Implements the bond, angle, torsion and van der Waals equations from
Rappe et al., J. Am. Chem. Soc. 1992, 114, 10024, using the UFF4MOF
parameter extensions from Addicoat et al., J. Chem. Theory Comput. 2014,
10, 880 and Coupry et al., J. Chem. Theory Comput. 2016, both bundled in
``cage_isomer_builder/__data__/uff4mof_data.py``.

"""

import math
from dataclasses import dataclass
from typing import Optional

from cage_isomer_builder.__data__.uff4mof_data import UFF4MOF_DATA

_HYBRIDIZATION_CHARS = {"1", "2", "R", "3", "4", "5", "6"}

# Group 6 elements get special-cased torsion barriers (Rappe 1992 Table 3).
_GROUP_6_SYMBOLS = {"O", "S", "Se", "Te", "Po"}
_GROUP_6_TORSION_V = {"O": 2.0, "S": 6.8, "Se": 6.8, "Te": 6.8, "Po": 6.8}

BOND_ORDER_LAMBDA = 0.1332
BOND_FORCE_CONSTANT = 664.12  # kcal/mol/Angstrom, = 2 * 332.06 (Coulomb constant)


@dataclass(frozen=True)
class UFFAtomParams:
    """One row of the UFF4MOF parameter table."""

    type_name: str
    r1: float          # ri, Angstrom - valence bond radius
    theta0_deg: float  # phi, degrees - equilibrium valence angle
    x1: float          # xi, Angstrom - van der Waals distance parameter
    D1: float           # di, kcal/mol - van der Waals well depth
    zeta: float          # nonbond scaling exponent (unused here)
    Z1: float            # zmm, e - effective charge (bond + angle force constants)
    V1: float            # vsp3, kcal/mol - sp3 torsion barrier
    U1: float            # vsp2, kcal/mol - sp2 torsion barrier
    chi: float          # GMP electronegativity
    hard: float          # QEq hardness (unused here - no charge equilibration)


def load_uff4mof_db() -> dict[str, UFFAtomParams]:
    """Return the UFF4MOF type -> params dict from the vendored data table."""
    params: dict[str, UFFAtomParams] = {}
    for name, fields in UFF4MOF_DATA.items():
        ri, phi, xi, di, zeta, zmm, vsp3, vsp2, chi, hard = fields[:10]
        params[name] = UFFAtomParams(
            type_name=name,
            r1=float(ri),
            theta0_deg=float(phi),
            x1=float(xi),
            D1=float(di),
            zeta=float(zeta),
            Z1=float(zmm),
            V1=float(vsp3),
            U1=float(vsp2),
            chi=float(chi),
            hard=float(hard),
        )
    return params


def hybridization_from_typename(type_name: str) -> Optional[str]:
    """
    Return the UFF hybridization/geometry code for a type name, or None.

    Per the UFF naming convention, the 3rd character encodes local geometry:
    1=linear, 2/R=trigonal(planar/resonant), 3=tetrahedral, 4=square planar,
    5=trigonal bipyramidal, 6=octahedral. Terminal types (H_, F_, Cl, ...)
    have no 3rd character and return None; they can never be an angle/torsion
    center since they have only one neighbour.
    """
    if len(type_name) < 3:
        return None
    code = type_name[2]
    return code if code in _HYBRIDIZATION_CHARS else None


def _element_symbol(type_name: str) -> str:
    """First one or two characters of a UFF type name, e.g. 'Zr3+4' -> 'Zr'."""
    if len(type_name) >= 2 and type_name[1].islower():
        return type_name[:2]
    return type_name[0]


def bond_params(pi: UFFAtomParams, pj: UFFAtomParams, bond_order: float) -> tuple[float, float]:
    """
    Bond-stretch equilibrium length and force constant (Rappe 1992 eqs 2-6).

    Returns (R_IJ in Angstrom, K_IJ in kcal/mol/Angstrom**2).
    """
    r_bo = -BOND_ORDER_LAMBDA * (pi.r1 + pj.r1) * math.log(bond_order)
    sqrt_chi_i = math.sqrt(pi.chi)
    sqrt_chi_j = math.sqrt(pj.chi)
    r_en = (
        pi.r1 * pj.r1 * (sqrt_chi_i - sqrt_chi_j) ** 2
        / (pi.chi * pi.r1 + pj.chi * pj.r1)
    )
    r_ij = pi.r1 + pj.r1 + r_bo - r_en
    k_ij = BOND_FORCE_CONSTANT * pi.Z1 * pj.Z1 / r_ij ** 3
    return r_ij, k_ij


def angle_coefficients(
    p_i: UFFAtomParams,
    p_center: UFFAtomParams,
    p_k: UFFAtomParams,
    bo_ic: float,
    bo_ck: float,
) -> tuple[float, float, tuple[float, float, float, float, float]]:
    """
    Angle-bend force constant and Fourier coefficients (Rappe 1992 eqs 12-15).

    theta0 is the CENTER atom's equilibrium angle. Reuses bond_params for
    r_IJ/r_JK so the angle term stays consistent with the bond term.

    Returns (theta0_rad, K in kcal/mol, (C0, C1, C2, C3, C4)) such that
    E = K * (C0 + C1*cos(theta) + C2*cos(2*theta) + C3*cos(3*theta) + C4*cos(4*theta)).
    """
    theta0_deg = p_center.theta0_deg
    theta0 = math.radians(theta0_deg)
    r_ij, _ = bond_params(p_i, p_center, bo_ic)
    r_jk, _ = bond_params(p_center, p_k, bo_ck)

    cos_t0 = math.cos(theta0)
    sin_t0 = math.sin(theta0)
    r_ik = math.sqrt(r_ij ** 2 + r_jk ** 2 - 2 * r_ij * r_jk * cos_t0)

    beta = BOND_FORCE_CONSTANT / (r_ij * r_jk)
    k = (
        beta
        * (p_i.Z1 * p_k.Z1 / r_ik ** 5)
        * (3 * r_ij * r_jk * (1 - cos_t0 ** 2) - r_ik ** 2 * cos_t0)
    )

    if abs(theta0_deg - 180.0) < 1e-6:
        coeffs = (1.0, 1.0, 0.0, 0.0, 0.0)
    elif abs(theta0_deg - 120.0) < 1e-6:
        coeffs = (1.0, 0.0, 0.0, -1.0, 0.0)
    elif abs(theta0_deg - 90.0) < 1e-6:
        coeffs = (1.0, 0.0, 0.0, 0.0, -1.0)
    else:
        c2 = 1.0 / (4.0 * sin_t0 ** 2)
        c1 = -4.0 * c2 * cos_t0
        c0 = c2 * (2 * cos_t0 ** 2 + 1)
        coeffs = (c0, c1, c2, 0.0, 0.0)

    return theta0, k, coeffs


def torsion_params(
    p_j: UFFAtomParams,
    p_k: UFFAtomParams,
    bond_order_jk: float,
    hyb_j: Optional[str],
    hyb_k: Optional[str],
) -> tuple[int, int, float]:
    """
    Torsion periodicity, cosine-sign term and barrier (Rappe 1992 Table 3).

    Returns (n, cosTerm, V in kcal/mol) for E = 0.5*V*(1 - cosTerm*cos(n*phi)).
    cosTerm is always exactly +1 or -1 (never a general phase).
    """
    sp3_j = hyb_j == "3"
    sp3_k = hyb_k == "3"
    sp2_j = hyb_j in ("2", "R")
    sp2_k = hyb_k in ("2", "R")

    elem_j = _element_symbol(p_j.type_name)
    elem_k = _element_symbol(p_k.type_name)

    if sp3_j and sp3_k:
        if (
            elem_j in _GROUP_6_SYMBOLS
            and elem_k in _GROUP_6_SYMBOLS
            and abs(bond_order_jk - 1.0) < 1e-6
        ):
            v_j = _GROUP_6_TORSION_V[elem_j]
            v_k = _GROUP_6_TORSION_V[elem_k]
            return 2, -1, 5.0 * math.sqrt(v_j * v_k)
        return 3, -1, math.sqrt(p_j.V1 * p_k.V1)

    if sp2_j and sp2_k:
        v = 5.0 * math.sqrt(p_j.U1 * p_k.U1) * (1 + 4.18 * math.log(bond_order_jk))
        return 2, 1, v

    # Group-6 sp3 atom singly-bonded to a non-Group-6 sp2 atom.
    group6_sp3_pair = (
        (sp3_j and elem_j in _GROUP_6_SYMBOLS and sp2_k and elem_k not in _GROUP_6_SYMBOLS)
        or (sp3_k and elem_k in _GROUP_6_SYMBOLS and sp2_j and elem_j not in _GROUP_6_SYMBOLS)
    )
    if group6_sp3_pair and abs(bond_order_jk - 1.0) < 1e-6:
        v = 5.0 * math.sqrt(p_j.U1 * p_k.U1) * (1 + 4.18 * math.log(bond_order_jk))
        return 2, -1, v

    # Default "everything else" rule (mixed sp3/sp2, metals, unresolved types).
    return 6, 1, 1.0


def vdw_params(pi: UFFAtomParams, pj: UFFAtomParams) -> tuple[float, float]:
    """van der Waals distance and well depth (Rappe 1992 eqs 20-22), geometric mean."""
    x_ij = math.sqrt(pi.x1 * pj.x1)
    d_ij = math.sqrt(pi.D1 * pj.D1)
    return x_ij, d_ij


def enumerate_bonds(bond_matrix) -> list[tuple[int, int, float]]:
    """List of (i, j, bond_order) with i < j and bond_order > 0."""
    n = bond_matrix.shape[0]
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            order = bond_matrix[i, j]
            if order > 0:
                bonds.append((i, j, float(order)))
    return bonds


def _adjacency(bond_matrix) -> dict[int, list[int]]:
    n = bond_matrix.shape[0]
    return {
        i: [j for j in range(n) if bond_matrix[i, j] > 0]
        for i in range(n)
    }


def enumerate_angles(bond_matrix) -> list[tuple[int, int, int]]:
    """List of (i, center, k) — every unordered pair of neighbours sharing a center."""
    adjacency = _adjacency(bond_matrix)
    angles = []
    for center, neighbours in adjacency.items():
        for a in range(len(neighbours)):
            for b in range(a + 1, len(neighbours)):
                angles.append((neighbours[a], center, neighbours[b]))
    return angles


def enumerate_torsions(bond_matrix) -> list[tuple[int, int, int, int]]:
    """List of (i, j, k, l) dihedral quadruples, one central bond j-k at a time."""
    adjacency = _adjacency(bond_matrix)
    torsions = []
    for j, k, _ in enumerate_bonds(bond_matrix):
        for i in adjacency[j]:
            if i == k:
                continue
            for l in adjacency[k]:
                if l == j or l == i:
                    continue
                torsions.append((i, j, k, l))
    return torsions


def bond_graph_distance(bond_matrix):
    """
    All-pairs shortest path length (in bonds) via BFS; ``float('inf')`` if
    disconnected. Used to exclude 1-2 and 1-3 neighbours from the van der
    Waals term (Rappe 1992 sec. on nonbonded interactions).
    """
    from collections import deque

    import numpy as np

    n = bond_matrix.shape[0]
    adjacency = [
        [j for j in range(n) if bond_matrix[i, j] > 0] for i in range(n)
    ]
    dist = np.full((n, n), np.inf)
    for start in range(n):
        dist[start, start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if dist[start, v] == np.inf:
                    dist[start, v] = dist[start, u] + 1
                    queue.append(v)
    return dist
