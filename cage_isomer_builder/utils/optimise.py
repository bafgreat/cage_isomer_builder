#!/usr/bin/env python3
"""
UFF4MOF geometry optimisation, built directly from an explicit bond-order
matrix rather than distances alone.

Unlike GFN-xTB (see :mod:`cage_isomer_builder.utils.tblite_optimise`), UFF is
a classical, topology-based force field: every energy term (bond, angle,
torsion, van der Waals) is enumerated from the bond matrix and an explicit
per-atom UFF type name, so the intended connectivity from the STK-constructed
cage is respected exactly even when interatomic distances are atypical
(bulky multi-atom nodes, expanded cages, ...). xTB needs no bond matrix,
but it also cannot be held to one.

``UFF4MOFCalculator`` evaluates energies and forces with OpenMM
(``openmm.CustomBondForce`` / ``CustomAngleForce`` / ``CustomTorsionForce``
for bond, angle and torsion terms, ``CustomCompoundBondForce`` for the
out-of-plane inversion term, and ``CustomNonbondedForce`` for van der Waals
and Coulomb beyond 1-3 neighbours), using the combining rules in
:mod:`cage_isomer_builder.utils.uff4mof_params`. The energy matches RDKit's
UFF implementation for organic molecules (see tests/test_uff4mof_backend.py).
"""
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

from collections import Counter
from typing import Optional

import numpy as np
from ase import units as ase_units
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import LBFGS

from cage_isomer_builder.utils.uff4mof_params import (
    angle_coefficients,
    bond_params,
    enumerate_angles,
    enumerate_bonds,
    enumerate_inversions,
    enumerate_torsions,
    hybridization_from_typename,
    inversion_params,
    torsion_params,
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
        Fixed partial charges (elementary charges). When given, a Coulomb
        term ``ke*qi*qj/r`` is added over the same non-bonded pairs as the
        van der Waals term (more than two bonds apart). UFF, and so UFF4MOF,
        assumes such an electrostatic term; without it the attraction
        between charged fragments (e.g. carboxylates and a metal cluster) is
        missing. The charges are not updated during the optimisation, as is
        usual for UFF with QEq charges. ``None`` (default) adds no Coulomb
        term.
    platform : str, optional
        OpenMM platform name ("Reference", "CPU", "OpenCL", "CUDA").
        Defaults to the fastest one available.
    precision : str, default "mixed"
        Precision for platforms that support it (OpenCL, CUDA): "single",
        "mixed" or "double". "mixed" computes forces in single precision
        and adds them up in double precision. The Reference platform is
        always double precision.

    Examples
    --------
    >>> import numpy as np
    >>> from ase.build import molecule
    >>> methane = molecule("CH4")
    >>> bonds = np.zeros((5, 5))
    >>> bonds[0, 1:] = bonds[1:, 0] = 1.0              # four C-H bonds
    >>> from gulp_setup.mmanalysis import analyze_mm
    >>> from cage_isomer_builder.utils.optimise import UFF4MOFCalculator
    >>> from cage_isomer_builder.utils.uff4mof_params import load_uff4mof_db
    >>> _, types = analyze_mm(methane, bond_order=bonds)
    >>> methane.calc = UFF4MOFCalculator(bonds, types, load_uff4mof_db())
    >>> [str(t) for t in types], methane.get_forces().shape
    (['C_3', 'H_', 'H_', 'H_', 'H_'], (5, 3))
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, bond_matrix, mmtypes, uff_db, charges=None,
                 platform=None, precision="mixed", **kwargs):
        super().__init__(**kwargs)
        self.bond_matrix = np.asarray(bond_matrix, dtype=float)
        self.mmtypes = np.asarray(mmtypes)
        self.uff_db = uff_db
        self.charges = None if charges is None else np.asarray(charges, dtype=float)
        self.platform = platform
        self.precision = precision
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

        torsions = enumerate_torsions(self.bond_matrix)
        # Each bond's barrier is shared out over all torsions about it.
        per_bond = Counter((min(j, k), max(j, k)) for _, j, k, _ in torsions)
        torsion_force = openmm.CustomTorsionForce("0.5*v*(1 - cosTerm*cos(n*theta))")
        for name in ("v", "cosTerm", "n"):
            torsion_force.addPerTorsionParameter(name)
        for i, j, k_idx, l in torsions:
            n, cos_term, v = torsion_params(
                params[j], params[k_idx], self.bond_matrix[j, k_idx],
                hybridizations[j], hybridizations[k_idx],
                hybridizations[i], hybridizations[l],
            )
            if v == 0.0:
                continue
            v /= per_bond[(min(j, k_idx), max(j, k_idx))]
            torsion_force.addTorsion(i, j, k_idx, l, [v * KCAL_TO_KJ, cos_term, n])
        system.addForce(torsion_force)

        # Out-of-plane term. With a = r1 - r2, b = r3 - r2, l = r4 - r2
        # (particle 2 is the centre), cos(w) = sin(angle between l and a x b).
        inversion_force = openmm.CustomCompoundBondForce(
            4,
            "k*(c0 + c1*sinY + c2*(2*sinY^2 - 1));"
            "sinY = sqrt(max(1e-10, 1 - cosY^2));"
            "cosY = (nx*lx + ny*ly + nz*lz) / sqrt((nx^2 + ny^2 + nz^2)*(lx^2 + ly^2 + lz^2));"
            "nx = ay*bz - az*by; ny = az*bx - ax*bz; nz = ax*by - ay*bx;"
            "ax = x1 - x2; ay = y1 - y2; az = z1 - z2;"
            "bx = x3 - x2; by = y3 - y2; bz = z3 - z2;"
            "lx = x4 - x2; ly = y4 - y2; lz = z4 - z2",
        )
        for name in ("k", "c0", "c1", "c2"):
            inversion_force.addPerBondParameter(name)
        for i, c, k_idx, l in enumerate_inversions(self.bond_matrix):
            term = inversion_params(
                self.mmtypes[c], (self.mmtypes[i], self.mmtypes[k_idx], self.mmtypes[l])
            )
            if term is not None:
                k_inv, c0, c1, c2 = term
                inversion_force.addBond([i, c, k_idx, l], [k_inv * KCAL_TO_KJ, c0, c1, c2])
        system.addForce(inversion_force)

        # van der Waals (and Coulomb when charges are given) over every pair
        # more than two bonds apart. Geometric-mean combining rules.
        nonbonded_energy = "d*((x/r)^12 - 2*(x/r)^6)"
        if self.charges is not None:
            nonbonded_energy += f" + {COULOMB_CONSTANT}*q1*q2/r"
        nonbonded = openmm.CustomNonbondedForce(
            nonbonded_energy + "; x = sqrt(x1*x2); d = sqrt(d1*d2)"
        )
        nonbonded.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
        for name in ("x", "d", "q"):
            nonbonded.addPerParticleParameter(name)
        charges = self.charges if self.charges is not None else np.zeros(len(atoms))
        for p, q in zip(params, charges):
            nonbonded.addParticle([p.x1 * NM_PER_ANGSTROM, p.D1 * KCAL_TO_KJ, float(q)])
        nonbonded.createExclusionsFromBonds(
            [(i, j) for i, j, _ in enumerate_bonds(self.bond_matrix)], 2
        )
        system.addForce(nonbonded)

        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        if self.platform is not None:
            candidates = [openmm.Platform.getPlatformByName(self.platform)]
        else:
            # Fastest first. A GPU platform can fail to start, e.g. Apple's
            # OpenCL has no double support, so it can't run "mixed".
            candidates = sorted(
                (openmm.Platform.getPlatform(i) for i in range(openmm.Platform.getNumPlatforms())),
                key=lambda p: -p.getSpeed(),
            )
        for n_tried, platform in enumerate(candidates, start=1):
            properties = {}
            if "Precision" in platform.getPropertyNames():
                properties["Precision"] = self.precision
            try:
                self._context = openmm.Context(system, integrator, platform, properties)
                break
            except openmm.OpenMMException:
                if n_tried == len(candidates):
                    raise
                integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        """
        Evaluate the UFF4MOF energy (eV) and forces (eV/Angstrom) with
        OpenMM (ASE calculator interface).

        Examples
        --------
        >>> import numpy as np
        >>> from ase.build import molecule
        >>> water = molecule("H2O")
        >>> bonds = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        >>> from gulp_setup.mmanalysis import analyze_mm
        >>> from cage_isomer_builder.utils.optimise import UFF4MOFCalculator
        >>> from cage_isomer_builder.utils.uff4mof_params import load_uff4mof_db
        >>> _, types = analyze_mm(water, bond_order=bonds)
        >>> water.calc = UFF4MOFCalculator(bonds, types, load_uff4mof_db())
        >>> isinstance(water.get_potential_energy(), float)      # calls calculate()
        True
        """
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
    logfile: Optional[str] = "-",
):
    """
    Relax a molecular geometry with ASE using a UFF4MOF (OpenMM) calculator.

    UFF atom types are assigned automatically from ``bond_matrix`` via
    :func:`gulp_setup.mmanalysis.analyze_mm` (a heuristic cost function over
    local coordination number, angle and radius); the bond matrix is used as
    given, never re-derived from the geometry.

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
        Fixed partial charges, passed to :class:`UFF4MOFCalculator` for its
        Coulomb term. ``None`` (default) runs without electrostatics.
    logfile : str or None, default "-"
        Where the optimiser writes its step log: ``"-"`` for the terminal,
        a file path, or ``None`` for no output.

    Returns
    -------
    ase.Atoms
        Optimised structure (a copy; the input ``atoms`` is not modified).

    Examples
    --------
    >>> import numpy as np
    >>> from ase.build import molecule
    >>> methane = molecule("CH4")
    >>> bonds = np.zeros((5, 5))
    >>> bonds[0, 1:] = bonds[1:, 0] = 1.0              # four C-H bonds
    >>> from cage_isomer_builder.utils.optimise import run_uff4mof_optimisation
    >>> relaxed = run_uff4mof_optimisation(methane, bonds, logfile=None)
    >>> round(float(relaxed.get_distance(0, 1)), 2), round(float(relaxed.get_angle(1, 0, 2)), 1)
    (1.11, 109.5)
    """
    import warnings

    from gulp_setup.mmanalysis import analyze_mm

    from cage_isomer_builder.__data__.uff4mof_data import DERIVED_UFF4MOF_DATA, UFF4MOF_DATA
    from cage_isomer_builder.utils.uff4mof_params import load_uff4mof_db

    if optimizer_cls is None:
        optimizer_cls = LBFGS

    opt_atoms = atoms.copy()
    opt_atoms.pbc = False

    _, mmtypes = analyze_mm(opt_atoms, bond_order=np.asarray(bond_matrix, dtype=float))
    uff_db = load_uff4mof_db()
    unknown = sorted({str(t) for t in mmtypes if t not in uff_db})
    if unknown:
        raise ValueError(
            f"no UFF4MOF parameters for atom type(s) {unknown} assigned by "
            "gulp_setup; use method='GFN2-xTB' for this structure."
        )

    published = {tuple(v) for v in UFF4MOF_DATA.values()}
    # Derived types identical to a published one (O_HH = O_3, H_OH = H_,
    # N_3+4 = N_3, K = K_, W_3f4 = W_3f2) are aliases: nothing to warn about.
    derived = sorted({str(t) for t in mmtypes if t in DERIVED_UFF4MOF_DATA
                      and tuple(DERIVED_UFF4MOF_DATA[t]) not in published})
    if derived:
        warnings.warn(
            f"UFF4MOF type(s) {derived} have no published parameter set; using "
            "parameters derived from the same element's type (see "
            "DERIVED_UFF4MOF_DATA in cage_isomer_builder/__data__/uff4mof_data.py)."
        )
    opt_atoms.calc = UFF4MOFCalculator(
        bond_matrix=bond_matrix, mmtypes=mmtypes, uff_db=uff_db, charges=charges
    )

    opt = optimizer_cls(opt_atoms, trajectory=trajectory, logfile=logfile)
    opt.run(fmax=fmax, steps=steps)

    return opt_atoms

