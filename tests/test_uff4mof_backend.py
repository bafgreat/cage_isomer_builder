#!/usr/bin/env python3
"""
Standalone verification for the UFF4MOF/OpenMM optimiser backend.

Run from the repository root:

    python3 tests/test_uff4mof_backend.py

Prints PASS/FAIL per check; exits nonzero if any check fails. There is no
pytest infrastructure in this repo (see tests/test_cage.py), so this follows
the same plain-script convention.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from gulp_setup import mmanalysis

from cage_isomer_builder.cage import Tri4Di6CageBuilder
from cage_isomer_builder.utils.optimise import UFF4MOFCalculator
from cage_isomer_builder.utils.read_write import stk_2_ase_atoms
from cage_isomer_builder.utils.uff4mof_params import (
    UFFAtomParams,
    angle_coefficients,
    bond_graph_distance,
    bond_params,
    enumerate_angles,
    enumerate_bonds,
    enumerate_torsions,
    hybridization_from_typename,
    load_uff4mof_db,
    torsion_params,
    vdw_params,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
NODE_FILE = DATA_DIR / "uio66_tri_node.xyz"
LINKER_FILE = DATA_DIR / "bdc.xyz"

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    suffix = f" - {detail}" if detail and not condition else ""
    print(f"[{status}] {name}{suffix}")
    if not condition:
        FAILURES.append(name)


def section(title: str) -> None:
    print(f"\n== {title} ==")


# ---------------------------------------------------------------------------
# 1. Parameter-module unit checks (no ASE/OpenMM system construction)
# ---------------------------------------------------------------------------

def test_parameter_module() -> dict[str, UFFAtomParams]:
    section("Parameter database + combining rules")
    db = load_uff4mof_db()
    check("db loads with many entries", len(db) > 100, f"{len(db)} entries")
    check("C_3 has correct r1", math.isclose(db["C_3"].r1, 0.757))
    check("Zr3+4 present", "Zr3+4" in db)

    check("hybridization C_3 -> '3'", hybridization_from_typename("C_3") == "3")
    check("hybridization C_R -> 'R'", hybridization_from_typename("C_R") == "R")
    check("hybridization N_1 -> '1'", hybridization_from_typename("N_1") == "1")
    check("hybridization H_ -> None", hybridization_from_typename("H_") is None)
    check("hybridization H_b -> None", hybridization_from_typename("H_b") is None)

    c3 = db["C_3"]
    r1, _k1 = bond_params(c3, c3, 1.0)
    check(
        "C_3-C_3 single bond close to sum of radii",
        math.isclose(r1, 2 * c3.r1, abs_tol=0.05),
        f"r1={r1}",
    )
    r2, _ = bond_params(c3, c3, 2.0)
    check("higher bond order shortens bond", r2 < r1, f"r1={r1} r2={r2}")

    # theta0 must be a stationary point of E(theta) in every sub-case
    # (linear / trigonal-planar / square-planar-octahedral / general) -
    # a formula-consistency check, not a memorised constant.
    def stationary_at_theta0(type_name: str) -> tuple[float, float]:
        p = db[type_name]
        theta0, k, coeffs = angle_coefficients(p, p, p, 1.0, 1.0)

        def energy(theta: float) -> float:
            c0, c1, c2, c3_, c4 = coeffs
            return k * (
                c0
                + c1 * math.cos(theta)
                + c2 * math.cos(2 * theta)
                + c3_ * math.cos(3 * theta)
                + c4 * math.cos(4 * theta)
            )

        eps = 1e-4
        return energy(theta0), (energy(theta0 + eps) - energy(theta0 - eps)) / (2 * eps)

    for label, type_name in {
        "linear (N_1)": "N_1",
        "trigonal (C_R)": "C_R",
        "square-planar (Pt4+2)": "Pt4+2",
        "general/tetrahedral (C_3)": "C_3",
    }.items():
        e0, slope = stationary_at_theta0(type_name)
        check(f"angle {label}: E(theta0)==0", math.isclose(e0, 0.0, abs_tol=1e-6), f"E={e0}")
        check(
            f"angle {label}: theta0 stationary",
            math.isclose(slope, 0.0, abs_tol=1e-3),
            f"dE/dtheta={slope}",
        )

    sp3_sp3 = torsion_params(db["C_3"], db["C_3"], 1.0, "3", "3")
    check("sp3-sp3 cosTerm in {-1,1}", sp3_sp3[1] in (-1, 1), f"{sp3_sp3}")
    sp2_sp2 = torsion_params(db["C_R"], db["C_R"], 1.5, "R", "R")
    check("sp2-sp2 cosTerm in {-1,1}", sp2_sp2[1] in (-1, 1), f"{sp2_sp2}")
    mixed = torsion_params(db["C_3"], db["C_R"], 1.0, "3", "R")
    check("mixed default n=6", mixed[0] == 6, f"{mixed}")
    group6 = torsion_params(db["O_3"], db["O_3"], 1.0, "3", "3")
    check("group-6 sp3-sp3 exception n=2", group6[0] == 2, f"{group6}")

    x_ij, d_ij = vdw_params(c3, c3)
    check(
        "vdw self-pair reduces to own params",
        math.isclose(x_ij, c3.x1) and math.isclose(d_ij, c3.D1),
    )

    # Butane-like chain 0-1-2-3 (single bonds only) has an exact expected
    # bond/angle/torsion count.
    n = 4
    bm = np.zeros((n, n))
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        bm[a, b] = bm[b, a] = 1.0
    check("enumerate_bonds count", len(enumerate_bonds(bm)) == 3)
    check("enumerate_angles count", len(enumerate_angles(bm)) == 2)
    check("enumerate_torsions count", len(enumerate_torsions(bm)) == 1)

    return db


# ---------------------------------------------------------------------------
# 2. Independent-recomputation cross-check against the OpenMM calculator
# ---------------------------------------------------------------------------

def _reference_energy_kcal(
    atoms: Atoms, bond_matrix: np.ndarray, mmtypes: np.ndarray, db: dict
) -> float:
    """
    Independently recompute the total UFF4MOF energy (kcal/mol) directly from
    uff4mof_params' combining rules and the actual ASE geometry, as a
    cross-check against UFF4MOFCalculator's OpenMM-side wiring and unit
    conversions (a bug there would not show up when testing uff4mof_params
    in isolation, since both would otherwise agree by construction).
    """
    params = [db[t] for t in mmtypes]
    hybridizations = [hybridization_from_typename(t) for t in mmtypes]
    total = 0.0

    for i, j, order in enumerate_bonds(bond_matrix):
        r0, k = bond_params(params[i], params[j], order)
        r = atoms.get_distance(i, j, mic=False)
        total += 0.5 * k * (r - r0) ** 2

    for i, c, k_idx in enumerate_angles(bond_matrix):
        bo_ic = bond_matrix[i, c]
        bo_ck = bond_matrix[c, k_idx]
        _theta0, k_ijk, coeffs = angle_coefficients(
            params[i], params[c], params[k_idx], bo_ic, bo_ck
        )
        theta = math.radians(atoms.get_angle(i, c, k_idx, mic=False))
        c0, c1, c2, c3_, c4 = coeffs
        total += k_ijk * (
            c0
            + c1 * math.cos(theta)
            + c2 * math.cos(2 * theta)
            + c3_ * math.cos(3 * theta)
            + c4 * math.cos(4 * theta)
        )

    for i, j, k_idx, l in enumerate_torsions(bond_matrix):
        order_jk = bond_matrix[j, k_idx]
        n, cos_term, v = torsion_params(
            params[j], params[k_idx], order_jk, hybridizations[j], hybridizations[k_idx]
        )
        phi = math.radians(atoms.get_dihedral(i, j, k_idx, l, mic=False))
        total += 0.5 * v * (1 - cos_term * math.cos(n * phi))

    graph_dist = bond_graph_distance(bond_matrix)
    n_atoms = len(atoms)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if graph_dist[i, j] <= 2:
                continue
            x_ij, d_ij = vdw_params(params[i], params[j])
            r = atoms.get_distance(i, j, mic=False)
            total += d_ij * ((x_ij / r) ** 12 - 2 * (x_ij / r) ** 6)

    return total


def test_synthetic_system_energy(db: dict) -> None:
    section("Synthetic-system energy cross-check (OpenMM vs. independent recomputation)")

    atoms = Atoms(
        "C4",
        positions=[
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [1.5, 1.5, 0.3],
            [0.2, 2.3, 1.1],
        ],
    )
    n = 4
    bond_matrix = np.zeros((n, n))
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        bond_matrix[a, b] = bond_matrix[b, a] = 1.0
    mmtypes = np.array(["C_3", "C_3", "C_3", "C_3"])

    calc = UFF4MOFCalculator(bond_matrix=bond_matrix, mmtypes=mmtypes, uff_db=db)
    atoms.calc = calc
    energy_ev = atoms.get_potential_energy()

    expected_kcal = _reference_energy_kcal(atoms, bond_matrix, mmtypes, db)
    from ase import units
    expected_ev = expected_kcal * units.kcal / units.mol

    check(
        "OpenMM energy matches independent recomputation",
        math.isclose(energy_ev, expected_ev, rel_tol=1e-4, abs_tol=1e-6),
        f"openmm={energy_ev} eV, expected={expected_ev} eV",
    )

    forces = atoms.get_forces()
    check(
        "forces are finite and nonzero",
        np.all(np.isfinite(forces)) and np.any(np.abs(forces) > 1e-8),
    )


# ---------------------------------------------------------------------------
# 3. Full-cage smoke test
# ---------------------------------------------------------------------------

def test_full_cage_smoke(db: dict, tmp_dir: Path) -> None:
    section("Full-cage build + optimise smoke test")
    if not NODE_FILE.exists() or not LINKER_FILE.exists():
        check("fixture files exist", False, f"missing {NODE_FILE} or {LINKER_FILE}")
        return

    cage = Tri4Di6CageBuilder(
        node=str(NODE_FILE),
        linker=str(LINKER_FILE),
        optimizer_step=0.01,
        distance_threshold=5.0,
    )
    cage.build()
    ase_atoms, bond_matrix = stk_2_ase_atoms(cage._stk_cage)
    n_before = len(ase_atoms)

    corrected_bonds, mmtypes = mmanalysis.analyze_mm(ase_atoms, bond_order=bond_matrix)
    check(
        "analyze_mm(bond_order=...) returns the input matrix unchanged",
        np.array_equal(corrected_bonds, bond_matrix),
    )
    aromatic_pairs = np.argwhere(np.isclose(bond_matrix, 1.5))
    check(
        "order-1.5 bonds map to '_R' types",
        all(mmtypes[i].endswith("R") for i, _j in aromatic_pairs) if len(aromatic_pairs) else True,
        f"{[mmtypes[i] for i, _j in aromatic_pairs]}",
    )

    energy_before_ev = UFF4MOFCalculator(
        bond_matrix=corrected_bonds, mmtypes=mmtypes, uff_db=db
    ).get_potential_energy(ase_atoms)

    traj_path = tmp_dir / "smoke_test.traj"
    try:
        optimised = cage.optimise(
            fmax=0.3, steps=150, optimizer_cls=BFGS, trajectory=str(traj_path),
            method="UFF4MOF",
        )
        ran_ok = True
    except Exception as exc:  # noqa: BLE001 - want to report, not crash the suite
        ran_ok = False
        check("optimise() runs without exception", False, repr(exc))
        return

    check("optimise() runs without exception", ran_ok)
    check("atom count unchanged", len(optimised) == n_before)
    check("trajectory file written", traj_path.exists())

    energy_after_ev = UFF4MOFCalculator(
        bond_matrix=corrected_bonds, mmtypes=mmtypes, uff_db=db
    ).get_potential_energy(optimised)
    check(
        "potential energy decreased",
        energy_after_ev < energy_before_ev,
        f"before={energy_before_ev} eV, after={energy_after_ev} eV",
    )

    # optimised.get_forces() (not a fresh unconstrained calculator) so the
    # FixRigidBodies projection matches what the optimizer actually saw -
    # rigid_sbus=True lets per-atom forces exceed fmax as long as each SBU's
    # net rigid-body force/torque has converged.
    final_forces = optimised.get_forces()
    max_force = np.sqrt((final_forces ** 2).sum(axis=1).max())
    check("max force <= fmax", max_force <= 0.3 + 1e-3, f"max_force={max_force}")


def main() -> int:
    import tempfile

    db = test_parameter_module()
    test_synthetic_system_energy(db)
    with tempfile.TemporaryDirectory() as tmp:
        test_full_cage_smoke(db, Path(tmp))

    print(f"\n{len(FAILURES)} failing check(s)." if FAILURES else "\nAll checks passed.")
    for name in FAILURES:
        print(f"  - {name}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
