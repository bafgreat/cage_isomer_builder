#!/usr/bin/env python3
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

"""
Partial-charge prediction for finite (non-periodic) structures via tblite.

tblite (github.com/tblite/tblite) is a light-weight GFN-xTB implementation.
Unlike PACMAN-charge, it operates directly on molecular (non-periodic)
structures, so no vacuum-padded artificial cell or CIF round-trip is
required.
"""

import numpy as np
from ase import Atoms

DEFAULT_METHOD = "GFN2-xTB"


def predict_tblite_charges(
    atoms: Atoms,
    method: str = DEFAULT_METHOD,
    charge: float = 0.0,
) -> np.ndarray:
    """
    Predict per-atom partial charges for a non-periodic structure via tblite.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to charge. Atom order is preserved in the returned array.
    method : str, default "GFN2-xTB"
        tblite xTB Hamiltonian to use (e.g. "GFN1-xTB", "GFN2-xTB").
    charge : float, default 0.0
        Total charge of the structure.

    Returns
    -------
    np.ndarray, shape (len(atoms),)
        Per-atom partial charges (elementary charge units), neutralised to
        sum to exactly ``charge``.
    """
    from tblite.ase import TBLite  # heavy import - keep local

    calc_atoms = atoms.copy()
    calc_atoms.pbc = False
    calc_atoms.calc = TBLite(method=method, charge=charge, verbosity=0)
    calc_atoms.get_potential_energy()
    charges = np.asarray(calc_atoms.calc.get_charges(), dtype=float)

    if len(charges) != len(atoms):
        raise RuntimeError(
            f"tblite returned {len(charges)} charges for a "
            f"{len(atoms)}-atom structure."
        )
    if not np.any(charges):
        raise RuntimeError(
            "tblite output contained only zero charges."
        )

    charges += (charge - charges.sum()) / len(charges)
    return charges
