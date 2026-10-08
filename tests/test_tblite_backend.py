#!/usr/bin/env python3
"""
Standalone smoke tests for the GFN2-xTB (tblite) optimisation backend.

Run directly (python3 tests/test_tblite_backend.py) or under pytest.
test_full_cage_smoke is marked @pytest.mark.slow (builds/optimises a real
cage) and is skipped by default; run it explicitly with `pytest -m slow`.
"""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.optimize import BFGS

from cage_isomer_builder.utils.rigid_body import FixRigidBodies
from cage_isomer_builder.utils.tblite_optimise import run_tblite_optimisation


def test_small_molecule_optimisation() -> None:
    """A stretched water molecule should relax and its forces should converge."""
    atoms = molecule("H2O")
    atoms.positions[1] += [0.0, 0.0, 0.4]  # stretch one O-H bond

    stretched = atoms.copy()
    stretched.pbc = False
    from tblite.ase import TBLite
    stretched.calc = TBLite(method="GFN2-xTB", charge=0.0, verbosity=0)
    e_initial = stretched.get_potential_energy()

    optimised, charges = run_tblite_optimisation(
        atoms, fmax=0.01, steps=200, optimizer_cls=BFGS
    )
    e_final = optimised.get_potential_energy()

    forces = optimised.get_forces()
    fmax = np.sqrt((forces**2).sum(axis=1).max())
    assert fmax < 0.01, f"forces did not converge: fmax={fmax}"
    assert e_final < e_initial, "energy did not decrease during optimisation"

    assert charges.shape == (3,)
    assert np.isfinite(charges).all()
    assert abs(charges.sum()) < 1e-4, f"charges do not sum to ~0: {charges.sum()}"
    print("test_small_molecule_optimisation passed:",
          f"E_initial={e_initial:.4f} eV, E_final={e_final:.4f} eV, fmax={fmax:.5f}")


def test_rigid_body_constraint() -> None:
    """FixRigidBodies must keep each group's internal shape fixed under xTB forces."""
    atoms = molecule("CH3CH2OH")
    n = len(atoms)
    # Treat the whole molecule as one rigid group and displace it. The
    # optimisation may move and rotate it but must not change its bond
    # lengths.
    group = list(range(n))
    ref = FixRigidBodies.from_atoms(atoms, [group])

    displaced = atoms.copy()
    displaced.positions += [1.5, 0.7, -0.3]
    displaced.set_constraint(ref)

    optimised, _ = run_tblite_optimisation(displaced, fmax=0.05, steps=100)

    def bond_lengths(a: Atoms) -> np.ndarray:
        pos = a.get_positions()
        return np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)

    before = bond_lengths(atoms)
    after = bond_lengths(optimised)
    max_shape_drift = np.abs(before - after).max()
    assert max_shape_drift < 1e-6, (
        f"rigid body internal shape drifted by {max_shape_drift} Angstrom"
    )
    print("test_rigid_body_constraint passed:",
          f"max internal shape drift={max_shape_drift:.2e} Angstrom")


@pytest.mark.slow
def test_full_cage_smoke(tmp_dir: Path) -> None:
    """cage.optimise() should run end-to-end on a real cage with GFN2-xTB."""
    from cage_isomer_builder.cage import Tri4Di6CageBuilder

    data_dir = Path(__file__).parent / "data"
    cage = Tri4Di6CageBuilder(
        linker=str(data_dir / "bdc.xyz"), node=str(data_dir / "uio66_tri_node.xyz")
    )
    cage.build()
    n_atoms_before = len(cage.to_ase())

    optimised = cage.optimise(
        fmax=0.3, steps=25, rigid_sbus=True, include_charges=True
    )

    assert len(optimised) == n_atoms_before, "atom count changed during optimisation"

    charges = optimised.get_initial_charges()
    assert np.isfinite(charges).all(), "non-finite charges returned"
    assert abs(charges.sum()) < 1e-2, f"charges do not sum to ~0: {charges.sum()}"

    out_path = tmp_dir / "tblite_cage_smoke.xyz"
    cage.save(str(out_path))
    assert out_path.exists()
    print("test_full_cage_smoke passed:",
          f"{len(optimised)} atoms, charge sum={charges.sum():.4f}")


if __name__ == "__main__":
    import tempfile

    test_small_molecule_optimisation()
    test_rigid_body_constraint()
    with tempfile.TemporaryDirectory() as tmp:
        test_full_cage_smoke(Path(tmp))
    print("\nAll tblite backend tests passed.")
