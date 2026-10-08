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

import importlib.util

import numpy as np
from ase import Atoms

DEFAULT_METHOD = "GFN2-xTB"


def tblite_available() -> bool:
    """True if the optional tblite package can be imported.

    Examples
    --------
    >>> from cage_isomer_builder.utils.tblite_charges import tblite_available
    >>> isinstance(tblite_available(), bool)
    True
    """
    return importlib.util.find_spec("tblite") is not None


def require_tblite() -> None:
    """Raise a clear ImportError if tblite is not installed.

    Examples
    --------
    >>> from cage_isomer_builder.utils.tblite_charges import require_tblite
    >>> require_tblite()        # returns quietly when tblite is installed, else ImportError
    """
    if not tblite_available():
        raise ImportError(
            "GFN-xTB needs the tblite package. Install it with "
            "'pip install cage_isomer_builder[xtb]' (no Windows wheels; on "
            "Windows use conda-forge or WSL), or use method='UFF4MOF'."
        )


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

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.tblite_charges import predict_tblite_charges
    >>> q = predict_tblite_charges(molecule("H2O"))          # needs tblite
    >>> bool(q[0] < 0 < q[1]), round(float(q.sum()), 6)      # O negative, neutral overall
    (True, 0.0)
    """
    require_tblite()
    from tblite.ase import TBLite  # slow import, kept local

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
