#!/usr/bin/env python3
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

"""
UFF4MOF geometry optimisation, built directly from an explicit bond-order
matrix rather than distances alone.

Unlike GFN-xTB (see :mod:`cage_isomer_builder.utils.tblite_optimise`), UFF is
a classical, topology-based force field: every energy term (bond, angle,
torsion, van der Waals) is enumerated from the bond matrix and an explicit
per-atom UFF type name, so the intended connectivity from the STK-constructed
cage is respected exactly even when interatomic distances are atypical
(bulky multi-atom nodes, expanded cages, ...). This is the tradeoff versus
xTB: no bond matrix is required for xTB, but it also cannot be told to
honour one.

``UFF4MOFCalculator`` evaluates energies and forces with OpenMM
(``openmm.CustomBondForce`` / ``CustomAngleForce`` / ``CustomTorsionForce``
for bonded terms, plus a pairwise ``CustomBondForce`` for the van der Waals
term beyond 1-3 neighbours), using the combining rules in
:mod:`cage_isomer_builder.utils.uff4mof_params`.
"""

from typing import Optional

import numpy as np
from ase import units as ase_units
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import LBFGS

from cage_isomer_builder.utils.uff4mof_params import (
    angle_coefficients,
    bond_graph_distance,
    bond_params,
    enumerate_angles,
    enumerate_bonds,
    enumerate_torsions,
    hybridization_from_typename,
    torsion_params,
    vdw_params,
)

KCAL_TO_KJ = 4.184
NM_PER_ANGSTROM = 0.1
# OpenMM's ONE_4PI_EPS0 (kJ/mol * nm / e^2), i.e. 1/(4*pi*epsilon_0) in MD units.
COULOMB_CONSTANT = 138.935456


class UFF4MOFCalculator(Calculator):
    """
    ASE Calculator evaluating the UFF4MOF energy/forces via OpenMM.

    Parameters
    ----------
    bond_matrix : np.ndarray, shape (n_atoms, n_atoms)
        Symmetric bond-order matrix (0 = no bond).
    mmtypes : sequence of str
        UFF4MOF type name for each atom (e.g. ``"C_3"``, ``"Zr3+4"``),
        as returned by :func:`gulp_setup.mmanalysis.analyze_mm`.
    uff_db : dict[str, UFFAtomParams]
        Parameter table from
        :func:`cage_isomer_builder.utils.uff4mof_params.load_uff4mof_db`.
    charges : np.ndarray, shape (n_atoms,), optional
        Fixed per-atom partial charges (elementary charge units). When
        given, a Coulomb term (``ke*qi*qj/r``) is added over the same
        non-bonded pairs as the van der Waals term (graph distance > 2) -
        UFF4MOF, like the original UFF, is parameterised assuming a QEq-like
        electrostatic contribution is present alongside the bond/angle/
        torsion/vdW terms; omitting it understates the electrostatic pull
        between charged fragments (e.g. an anionic carboxylate node and a
        cationic metal cluster) that helps hold a MOF-like assembly together
        correctly. Charges are held fixed for the whole optimisation (not
        re-equilibrated every step), matching standard UFF/QEq practice.
        When ``None`` (default), no electrostatic term is added at all.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, bond_matrix, mmtypes, uff_db, charges=None, **kwargs):
        super().__init__(**kwargs)
        self.bond_matrix = np.asarray(bond_matrix, dtype=float)
        self.mmtypes = np.asarray(mmtypes)
        self.uff_db = uff_db
        self.charges = None if charges is None else np.asarray(charges, dtype=float)
        self._context = None

    def _build_context(self, atoms):
        import openmm
        from openmm import unit

        params = [self.uff_db[t] for t in self.mmtypes]
        hybridizations = [hybridization_from_typename(t) for t in self.mmtypes]

        system = openmm.System()
        for mass in atoms.get_masses():
            system.addParticle(float(mass) if mass > 0 else 1.0)

        bond_force = openmm.CustomBondForce("0.5*k*(r-r0)^2")
        bond_force.addPerBondParameter("r0")
        bond_force.addPerBondParameter("k")
        for i, j, order in enumerate_bonds(self.bond_matrix):
            r0, k = bond_params(params[i], params[j], order)
            bond_force.addBond(
                i, j,
                [r0 * NM_PER_ANGSTROM, k * KCAL_TO_KJ / NM_PER_ANGSTROM ** 2],
            )
        system.addForce(bond_force)

        angle_force = openmm.CustomAngleForce(
            "k*(c0 + c1*cos(theta) + c2*cos(2*theta) "
            "+ c3*cos(3*theta) + c4*cos(4*theta))"
        )
        for name in ("k", "c0", "c1", "c2", "c3", "c4"):
            angle_force.addPerAngleParameter(name)
        for i, c, k_idx in enumerate_angles(self.bond_matrix):
            bo_ic = self.bond_matrix[i, c]
            bo_ck = self.bond_matrix[c, k_idx]
            _theta0, k_ijk, coeffs = angle_coefficients(
                params[i], params[c], params[k_idx], bo_ic, bo_ck
            )
            angle_force.addAngle(i, c, k_idx, [k_ijk * KCAL_TO_KJ, *coeffs])
        system.addForce(angle_force)

        torsion_force = openmm.CustomTorsionForce("0.5*v*(1 - cosTerm*cos(n*theta))")
        for name in ("v", "cosTerm", "n"):
            torsion_force.addPerTorsionParameter(name)
        for i, j, k_idx, l in enumerate_torsions(self.bond_matrix):
            order_jk = self.bond_matrix[j, k_idx]
            n, cos_term, v = torsion_params(
                params[j], params[k_idx], order_jk,
                hybridizations[j], hybridizations[k_idx],
            )
            torsion_force.addTorsion(i, j, k_idx, l, [v * KCAL_TO_KJ, cos_term, n])
        system.addForce(torsion_force)

        graph_dist = bond_graph_distance(self.bond_matrix)
        vdw_force = openmm.CustomBondForce("d*((x/r)^12 - 2*(x/r)^6)")
        vdw_force.addPerBondParameter("x")
        vdw_force.addPerBondParameter("d")
        coulomb_force = None
        if self.charges is not None:
            coulomb_force = openmm.CustomBondForce(f"{COULOMB_CONSTANT}*qq/r")
            coulomb_force.addPerBondParameter("qq")
        n_atoms = len(atoms)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                if graph_dist[i, j] <= 2:
                    continue
                x_ij, d_ij = vdw_params(params[i], params[j])
                vdw_force.addBond(i, j, [x_ij * NM_PER_ANGSTROM, d_ij * KCAL_TO_KJ])
                if coulomb_force is not None:
                    coulomb_force.addBond(i, j, [float(self.charges[i] * self.charges[j])])
        system.addForce(vdw_force)
        if coulomb_force is not None:
            system.addForce(coulomb_force)

        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        self._context = openmm.Context(system, integrator)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        from openmm import unit

        if self._context is None:
            self._build_context(self.atoms)

        positions_nm = self.atoms.get_positions() * NM_PER_ANGSTROM
        self._context.setPositions(positions_nm * unit.nanometer)
        state = self._context.getState(getEnergy=True, getForces=True)

        energy_kjmol = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces_kjmolnm = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )

        self.results["energy"] = energy_kjmol * ase_units.kJ / ase_units.mol
        self.results["forces"] = (
            np.asarray(forces_kjmolnm) * (ase_units.kJ / ase_units.mol) / ase_units.nm
        )


def run_uff4mof_optimisation(
    atoms,
    bond_matrix,
    fmax: float = 0.05,
    steps: int = 500,
    optimizer_cls=None,
    trajectory: Optional[str] = None,
    charges: Optional[np.ndarray] = None,
):
    """
    Relax a molecular geometry with ASE using a UFF4MOF (OpenMM) calculator.

    UFF atom types are assigned automatically from ``bond_matrix`` via
    :func:`gulp_setup.mmanalysis.analyze_mm` (a heuristic cost function over
    local coordination number, angle and radius - see that module for
    details); the bond matrix itself is taken as-is, never re-derived from
    geometry.

    Parameters
    ----------
    atoms : ase.Atoms
        Starting geometry. Any constraint already set on ``atoms``
        (e.g. ``FixRigidBodies``) is preserved through the optimisation.
    bond_matrix : np.ndarray, shape (len(atoms), len(atoms))
        Symmetric bond-order matrix, e.g. from
        :func:`cage_isomer_builder.utils.read_write.stk_2_ase_atoms`.
    fmax : float, default 0.05
        Force convergence threshold in eV/Angstrom.
    steps : int, default 500
        Maximum number of optimisation steps.
    optimizer_cls : ASE optimizer class, optional
        Any ASE ``Optimizer`` subclass (LBFGS, BFGS, FIRE, ...).
        Defaults to ``LBFGS``.
    trajectory : str, optional
        File path for an ASE ``.traj`` trajectory.
    charges : np.ndarray, shape (len(atoms),), optional
        Fixed per-atom partial charges forwarded to
        :class:`UFF4MOFCalculator` to add its Coulomb term. When ``None``
        (default), the optimisation runs without electrostatics, as before.

    Returns
    -------
    ase.Atoms
        Optimised structure (a copy - the input ``atoms`` is not modified).
    """
    from gulp_setup.mmanalysis import analyze_mm

    from cage_isomer_builder.utils.uff4mof_params import load_uff4mof_db

    if optimizer_cls is None:
        optimizer_cls = LBFGS

    opt_atoms = atoms.copy()
    opt_atoms.pbc = False

    _, mmtypes = analyze_mm(opt_atoms, bond_order=np.asarray(bond_matrix, dtype=float))
    uff_db = load_uff4mof_db()
    opt_atoms.calc = UFF4MOFCalculator(
        bond_matrix=bond_matrix, mmtypes=mmtypes, uff_db=uff_db, charges=charges
    )

    opt = optimizer_cls(opt_atoms, trajectory=trajectory)
    opt.run(fmax=fmax, steps=steps)

    return opt_atoms

