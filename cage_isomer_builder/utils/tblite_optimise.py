#!/usr/bin/env python3
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

"""
Geometry optimisation for finite (non-periodic) structures via tblite.

tblite (github.com/tblite/tblite) is a light-weight GFN-xTB implementation.
It operates directly on molecular (non-periodic) structures as a standard
ASE calculator, so any ``ase.optimize`` optimiser and any ``ase.constraints``
constraint (e.g. :class:`~cage_isomer_builder.utils.rigid_body.FixRigidBodies`)
work with it unchanged.
"""

from typing import Optional

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS

from cage_isomer_builder.utils.tblite_charges import DEFAULT_METHOD, require_tblite


def run_tblite_optimisation(
    atoms: Atoms,
    fmax: float = 0.05,
    steps: int = 500,
    optimizer_cls=None,
    trajectory: Optional[str] = None,
    charge: float = 0.0,
    method: str = DEFAULT_METHOD,
    logfile: Optional[str] = "-",
) -> tuple[Atoms, np.ndarray]:
    """
    Relax a molecular geometry with ASE using a GFN-xTB (tblite) calculator.

    Parameters
    ----------
    atoms : ase.Atoms
        Starting geometry. Any constraint already set on ``atoms``
        (e.g. ``FixRigidBodies``) is preserved through the optimisation.
    fmax : float, default 0.05
        Force convergence threshold in eV/Angstrom.
    steps : int, default 500
        Maximum number of optimisation steps.
    optimizer_cls : ASE optimizer class, optional
        Any ASE ``Optimizer`` subclass (LBFGS, BFGS, FIRE, ...).
        Defaults to ``LBFGS``.
    trajectory : str, optional
        File path for an ASE ``.traj`` trajectory.
    charge : float, default 0.0
        Total charge of the structure.
    method : str, default "GFN2-xTB"
        tblite xTB Hamiltonian to use (e.g. "GFN1-xTB", "GFN2-xTB").
    logfile : str or None, default "-"
        Where the optimiser writes its step log: ``"-"`` for the terminal,
        a file path, or ``None`` for no output.

    Returns
    -------
    ase.Atoms
        Optimised structure (a copy; the input ``atoms`` is not modified).
    np.ndarray, shape (len(atoms),)
        Per-atom partial charges from the final electronic structure,
        obtained as a byproduct of the last energy/force evaluation.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.tblite_optimise import run_tblite_optimisation
    >>> relaxed, charges = run_tblite_optimisation(molecule("H2O"), logfile=None)   # needs tblite
    >>> round(float(relaxed.get_distance(0, 1)), 2), charges.shape
    (0.96, (3,))
    """
    require_tblite()
    from tblite.ase import TBLite  # slow import, kept local

    if optimizer_cls is None:
        optimizer_cls = LBFGS

    opt_atoms = atoms.copy()
    opt_atoms.pbc = False
    opt_atoms.calc = TBLite(method=method, charge=charge, verbosity=0)

    opt = optimizer_cls(opt_atoms, trajectory=trajectory, logfile=logfile)
    opt.run(fmax=fmax, steps=steps)

    charges = np.asarray(opt_atoms.calc.get_charges(), dtype=float)

    return opt_atoms, charges
