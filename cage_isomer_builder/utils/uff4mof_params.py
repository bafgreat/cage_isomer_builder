#!/usr/bin/env python3
"""
UFF4MOF parameter database and force-field combining-rule math.

Implements the bond, angle, torsion and van der Waals equations from
Rappe et al., J. Am. Chem. Soc. 1992, 114, 10024, using the UFF4MOF
parameter extensions from Addicoat et al., J. Chem. Theory Comput. 2014,
10, 880 and Coupry et al., J. Chem. Theory Comput. 2016, both bundled in
``cage_isomer_builder/__data__/uff4mof_data.py``.
"""
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cage_isomer_builder.__data__.uff4mof_data import DERIVED_UFF4MOF_DATA, UFF4MOF_DATA

_HYBRIDIZATION_CHARS = {"1", "2", "R", "3", "4", "5", "6"}

# Group 6 elements get special-cased torsion barriers (Rappe 1992 Table 3).
_GROUP_6_SYMBOLS = {"O", "S", "Se", "Te", "Po"}
_GROUP_6_TORSION_V = {"O": 2.0, "S": 6.8, "Se": 6.8, "Te": 6.8, "Po": 6.8}

BOND_ORDER_LAMBDA = 0.1332
BOND_FORCE_CONSTANT = 664.12  # kcal/mol/Angstrom, = 2 * 332.06 (Coulomb constant)


@dataclass(frozen=True)
class UFFAtomParams:
    """One row of the UFF4MOF parameter table.

    Examples
    --------
    >>> from cage_isomer_builder.utils.uff4mof_params import load_uff4mof_db
    >>> db = load_uff4mof_db()
    >>> zr = db["Zr8f4"]                           # the 8-coordinate Zr of UiO-66
    >>> zr.r1, zr.theta0_deg
    (1.68, 109.47)
    """

    type_name: str
    r1: float          # ri (Angstrom): valence bond radius
    theta0_deg: float  # phi (degrees): equilibrium valence angle
    x1: float          # xi (Angstrom): van der Waals distance parameter
    D1: float          # di (kcal/mol): van der Waals well depth
    zeta: float        # nonbond scaling exponent (unused here)
    Z1: float          # zmm (e): effective charge (bond + angle force constants)
    V1: float          # vsp3 (kcal/mol): sp3 torsion barrier
    U1: float          # vsp2 (kcal/mol): sp2 torsion barrier
    chi: float         # GMP electronegativity
    hard: float        # QEq hardness (unused: no charge equilibration)


def load_uff4mof_db(include_derived: bool = True) -> dict[str, UFFAtomParams]:
    """
    Return the UFF4MOF type -> params dict from the vendored data table.

    ``include_derived`` adds the types in ``DERIVED_UFF4MOF_DATA`` (assignable
    by the atom typing but without a published parameter set; see that
    table's comment).

    Examples
    --------
    >>> from cage_isomer_builder.utils.uff4mof_params import load_uff4mof_db
    >>> db = load_uff4mof_db()
    >>> "C_R" in db, "Zr6f3" in db, "Zr6f3" in load_uff4mof_db(include_derived=False)
    (True, True, False)
    """
    params: dict[str, UFFAtomParams] = {}
    table = dict(UFF4MOF_DATA)
    if include_derived:
        table.update(DERIVED_UFF4MOF_DATA)
    for name, fields in table.items():
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

    Examples
    --------
    >>> from cage_isomer_builder.utils.uff4mof_params import hybridization_from_typename
    >>> hybridization_from_typename("C_R"), hybridization_from_typename("Zr6f3"), hybridization_from_typename("H_")
    ('R', '6', None)
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

    Examples
    --------
    >>> from cage_isomer_builder.utils.uff4mof_params import load_uff4mof_db
    >>> db = load_uff4mof_db()
    >>> from cage_isomer_builder.utils.uff4mof_params import bond_params
    >>> r, k = bond_params(db["C_3"], db["C_3"], 1.0)
    >>> round(r, 3), round(k, 1)                   # C-C single bond: 1.514 A, kcal/mol/A^2
    (1.514, 699.6)
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

    Examples
    --------
    >>> from cage_isomer_builder.utils.uff4mof_params import load_uff4mof_db
    >>> db = load_uff4mof_db()
    >>> import math
    >>> from cage_isomer_builder.utils.uff4mof_params import angle_coefficients
    >>> theta0, k, coeffs = angle_coefficients(db["H_"], db["C_3"], db["H_"], 1.0, 1.0)
    >>> round(math.degrees(theta0), 2), len(coeffs)
    (109.47, 5)
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
        * r_ij * r_jk
        * (3 * r_ij * r_jk * (1 - cos_t0 ** 2) - r_ik ** 2 * cos_t0)
    )

    # Linear, trigonal and square-planar/octahedral centres use
    # E = K/n^2 * (1 - cos(n*theta)) (n = 1, 3, 4); linear is K*(1 + cos(theta)).
    if abs(theta0_deg - 180.0) < 1e-6:
        coeffs = (1.0, 1.0, 0.0, 0.0, 0.0)
    elif abs(theta0_deg - 120.0) < 1e-6:
        coeffs = (1.0 / 9, 0.0, 0.0, -1.0 / 9, 0.0)
    elif abs(theta0_deg - 90.0) < 1e-6:
        coeffs = (1.0 / 16, 0.0, 0.0, 0.0, -1.0 / 16)
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
    hyb_i: Optional[str] = None,
    hyb_l: Optional[str] = None,
) -> tuple[int, int, float]:
    """
    Torsion periodicity, cosine-sign term and barrier for torsion i-j-k-l
    (Rappe 1992 eqs 16-17 and the rules below them).

    Returns (n, cosTerm, V in kcal/mol) for E = 0.5*V*(1 - cosTerm*cos(n*phi)).
    V is the barrier for the whole j-k bond; the calculator divides it by
    the number of torsions about that bond. cosTerm is always +1 or -1.
    Only sp2 and sp3 central atoms get a torsion; any other pair returns V = 0.

    ``hyb_i`` and ``hyb_l`` are the hybridizations of the outer atoms. They
    are only used for an sp2-sp3 bond where the sp2 atom's outer neighbour
    is also sp2 (as in propene), which takes n = 3 and V = 2 kcal/mol.

    Examples
    --------
    >>> from cage_isomer_builder.utils.uff4mof_params import load_uff4mof_db
    >>> db = load_uff4mof_db()
    >>> from cage_isomer_builder.utils.uff4mof_params import torsion_params
    >>> n, cos_term, barrier = torsion_params(db["C_3"], db["C_3"], 1.0, "3", "3")
    >>> n, cos_term, round(barrier, 3)             # ethane: threefold, V = sqrt(V_C * V_C)
    (3, -1, 2.119)
    """
    sp3_j = hyb_j == "3"
    sp3_k = hyb_k == "3"
    sp2_j = hyb_j in ("2", "R")
    sp2_k = hyb_k in ("2", "R")

    elem_j = _element_symbol(p_j.type_name)
    elem_k = _element_symbol(p_k.type_name)
    single = abs(bond_order_jk - 1.0) < 1e-6

    if sp3_j and sp3_k:
        if elem_j in _GROUP_6_SYMBOLS and elem_k in _GROUP_6_SYMBOLS and single:
            v_j = _GROUP_6_TORSION_V[elem_j]
            v_k = _GROUP_6_TORSION_V[elem_k]
            return 2, -1, math.sqrt(v_j * v_k)
        return 3, -1, math.sqrt(p_j.V1 * p_k.V1)

    if sp2_j and sp2_k:
        v = 5.0 * math.sqrt(p_j.U1 * p_k.U1) * (1 + 4.18 * math.log(bond_order_jk))
        return 2, 1, v

    if (sp2_j and sp3_k) or (sp3_j and sp2_k):
        if single:
            # Group-6 sp3 atom bonded to a non-group-6 sp2 atom.
            if (sp3_j and elem_j in _GROUP_6_SYMBOLS and elem_k not in _GROUP_6_SYMBOLS) or (
                sp3_k and elem_k in _GROUP_6_SYMBOLS and elem_j not in _GROUP_6_SYMBOLS
            ):
                v = 5.0 * math.sqrt(p_j.U1 * p_k.U1) * (1 + 4.18 * math.log(bond_order_jk))
                return 2, -1, v
            # sp3 bonded to an sp2 atom that has another sp2 neighbour (propene).
            outer_of_sp2 = hyb_i if sp2_j else hyb_l
            if outer_of_sp2 in ("2", "R"):
                return 3, -1, 2.0
        return 6, 1, 1.0

    # sp, metals with other geometries, terminal atoms: no torsion.
    return 6, 1, 0.0


# Group 15 out-of-plane equilibrium angles (degrees), Rappe 1992.
_GROUP_15_INVERSION_W0 = {"P": 84.4339, "As": 86.9735, "Sb": 87.7047, "Bi": 90.0}


def inversion_params(
    center_type: str, neighbour_types: tuple[str, str, str]
) -> Optional[tuple[float, float, float, float]]:
    """
    Out-of-plane (inversion) term for a centre with exactly three neighbours
    (Rappe 1992 eq 18).

    Applies to sp2 carbon and nitrogen (C_2, C_R, N_2, N_R) and to P, As, Sb
    and Bi. Returns (K, C0, C1, C2) for
    E = K*(C0 + C1*cos(w) + C2*cos(2w)), where w is the angle between one
    bond and the plane of the other two. K is already divided by 3, because
    each centre gets three such terms. Returns None for any other centre.

    Examples
    --------
    >>> from cage_isomer_builder.utils.uff4mof_params import inversion_params
    >>> k, c0, c1, c2 = inversion_params("C_R", ("C_R", "C_R", "H_"))
    >>> round(k, 3)                                # 6 kcal/mol shared over three terms
    2.0
    >>> print(inversion_params("C_3", ("H_", "H_", "H_")))   # sp3 carbon has none
    None
    """
    element = _element_symbol(center_type)
    hyb = hybridization_from_typename(center_type)
    if element in ("C", "N") and hyb in ("2", "R"):
        bound_to_sp2_o = element == "C" and any(
            _element_symbol(t) == "O" and hybridization_from_typename(t) in ("2", "R")
            for t in neighbour_types
        )
        k = 50.0 if bound_to_sp2_o else 6.0
        return k / 3, 1.0, -1.0, 0.0
    if element in _GROUP_15_INVERSION_W0:
        w0 = math.radians(_GROUP_15_INVERSION_W0[element])
        c2 = 1.0
        c1 = -4.0 * math.cos(w0)
        c0 = -(c1 * math.cos(w0) + c2 * math.cos(2 * w0))
        k = 22.0 / (c0 + c1 + c2)
        return k / 3, c0, c1, c2
    return None


def vdw_params(pi: UFFAtomParams, pj: UFFAtomParams) -> tuple[float, float]:
    """van der Waals distance and well depth (Rappe 1992 eqs 20-22), geometric mean.

    Examples
    --------
    >>> from cage_isomer_builder.utils.uff4mof_params import load_uff4mof_db
    >>> db = load_uff4mof_db()
    >>> from cage_isomer_builder.utils.uff4mof_params import vdw_params
    >>> x, d = vdw_params(db["C_3"], db["C_3"])
    >>> round(x, 3), round(d, 3)                   # geometric means of x and D
    (3.851, 0.105)
    """
    x_ij = math.sqrt(pi.x1 * pj.x1)
    d_ij = math.sqrt(pi.D1 * pj.D1)
    return x_ij, d_ij


def enumerate_bonds(bond_matrix) -> list[tuple[int, int, float]]:
    """List of (i, j, bond_order) with i < j and bond_order > 0.

    Examples
    --------
    >>> import numpy as np
    >>> bonds = np.zeros((4, 4))                   # a chain 0-1-2-3
    >>> for i in range(3):
    ...     bonds[i, i + 1] = bonds[i + 1, i] = 1.0
    >>> from cage_isomer_builder.utils.uff4mof_params import enumerate_bonds
    >>> enumerate_bonds(bonds)
    [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    """
    bond_matrix = np.asarray(bond_matrix)
    rows, cols = np.nonzero(np.triu(bond_matrix, k=1) > 0)
    return [(int(i), int(j), float(bond_matrix[i, j])) for i, j in zip(rows, cols)]


def _adjacency(bond_matrix) -> dict[int, list[int]]:
    bond_matrix = np.asarray(bond_matrix)
    return {i: np.flatnonzero(row > 0).tolist() for i, row in enumerate(bond_matrix)}


def enumerate_angles(bond_matrix) -> list[tuple[int, int, int]]:
    """List of (i, center, k): every unordered pair of neighbours of a center.

    Examples
    --------
    >>> import numpy as np
    >>> bonds = np.zeros((4, 4))                   # a chain 0-1-2-3
    >>> for i in range(3):
    ...     bonds[i, i + 1] = bonds[i + 1, i] = 1.0
    >>> from cage_isomer_builder.utils.uff4mof_params import enumerate_angles
    >>> enumerate_angles(bonds)                    # (outer, centre, outer)
    [(0, 1, 2), (1, 2, 3)]
    """
    adjacency = _adjacency(bond_matrix)
    angles = []
    for center, neighbours in adjacency.items():
        for a in range(len(neighbours)):
            for b in range(a + 1, len(neighbours)):
                angles.append((neighbours[a], center, neighbours[b]))
    return angles


def enumerate_torsions(bond_matrix) -> list[tuple[int, int, int, int]]:
    """List of (i, j, k, l) dihedral quadruples, one central bond j-k at a time.

    Examples
    --------
    >>> import numpy as np
    >>> bonds = np.zeros((4, 4))                   # a chain 0-1-2-3
    >>> for i in range(3):
    ...     bonds[i, i + 1] = bonds[i + 1, i] = 1.0
    >>> from cage_isomer_builder.utils.uff4mof_params import enumerate_torsions
    >>> enumerate_torsions(bonds)
    [(0, 1, 2, 3)]
    """
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


def enumerate_inversions(bond_matrix) -> list[tuple[int, int, int, int]]:
    """
    List of (i, center, k, l) for every centre with exactly three
    neighbours: three entries per centre, each with a different neighbour
    as ``l``, the atom whose angle to the i-center-k plane is measured.

    Examples
    --------
    >>> import numpy as np
    >>> from cage_isomer_builder.utils.uff4mof_params import enumerate_inversions
    >>> bonds = np.zeros((4, 4))
    >>> bonds[0, 1:] = bonds[1:, 0] = 1.0          # atom 0 with three neighbours
    >>> len(enumerate_inversions(bonds))           # one term per neighbour taken as the "out" bond
    3
    """
    inversions = []
    for center, (a, b, c) in (
        (center, nbrs) for center, nbrs in _adjacency(bond_matrix).items() if len(nbrs) == 3
    ):
        inversions.extend([(a, center, b, c), (b, center, c, a), (c, center, a, b)])
    return inversions


def bond_graph_distance(bond_matrix):
    """
    All-pairs shortest path length (in bonds); ``inf`` if disconnected.
    Used to exclude 1-2 and 1-3 neighbours from the van der Waals term
    (Rappe 1992 sec. on nonbonded interactions).

    Examples
    --------
    >>> import numpy as np
    >>> bonds = np.zeros((4, 4))                   # a chain 0-1-2-3
    >>> for i in range(3):
    ...     bonds[i, i + 1] = bonds[i + 1, i] = 1.0
    >>> from cage_isomer_builder.utils.uff4mof_params import bond_graph_distance
    >>> bond_graph_distance(bonds)[0].tolist()     # bonds between atom 0 and each atom
    [0.0, 1.0, 2.0, 3.0]
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    graph = csr_matrix((np.asarray(bond_matrix) > 0).astype(float))
    return shortest_path(graph, unweighted=True, directed=False)
