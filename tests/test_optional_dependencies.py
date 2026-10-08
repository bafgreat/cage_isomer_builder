"""
The package must still work without its optional extras, tblite and dftd4
(neither has Windows wheels). These tests hide both packages and check that
each feature falls back to what is possible, with a warning, or raises a
clear error when there is no fallback.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from ase.build import molecule
from ase.io import read

from cage_isomer_builder.utils import dispersion, tblite_charges
from cage_isomer_builder.utils.functionalise import place_guest_in_host

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def no_dftd4(monkeypatch):
    """Behave as if dftd4 is not installed: any import of it fails."""
    monkeypatch.setattr(dispersion, "dftd4_available", lambda: False)
    monkeypatch.setitem(sys.modules, "dftd4", None)
    monkeypatch.setitem(sys.modules, "dftd4.interface", None)


@pytest.fixture
def no_tblite(monkeypatch):
    """Behave as if tblite is not installed: any import of it fails."""
    monkeypatch.setattr(tblite_charges, "tblite_available", lambda: False)
    monkeypatch.setitem(sys.modules, "tblite", None)
    monkeypatch.setitem(sys.modules, "tblite.ase", None)


def test_uff_kernel_matches_direct_sum():
    host = molecule("C6H6")
    guest = molecule("CH3CH2OH")
    guest.translate([0.0, 0.0, 3.6])
    model = dispersion.UFFPairModel(host, [guest])
    energy, penalty, overlap = model.score(
        np.arange(len(host)), host.positions,
        model.offsets[1] + np.arange(len(guest)), guest.positions[None],
        np.zeros((len(host), len(guest))),
    )
    assert penalty[0] == 0.0 and overlap[0] == 0.0
    assert energy[0] == pytest.approx(dispersion.uff_interaction_energy([host, guest]))


def test_auto_falls_back_to_uff_with_a_warning(no_dftd4):
    host, guest = molecule("C6H6"), molecule("CH4")
    with pytest.warns(UserWarning, match="dftd4 is not installed"):
        model = dispersion.pair_model(host, [guest], "auto")
    assert isinstance(model, dispersion.UFFPairModel)


def test_asking_for_d4_without_dftd4_raises(no_dftd4):
    host, guest = molecule("C6H6"), molecule("CH4")
    with pytest.raises(ImportError, match=r"cage_isomer_builder\[d4\]"):
        dispersion.pair_model(host, [guest], "pbe")


def test_energy_placement_works_without_dftd4(no_dftd4):
    """Methane in C60 still goes to the centre, scored with UFF."""
    host = molecule("C60")
    guest = read(str(DATA_DIR / "methane.xyz"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        complex_ = place_guest_in_host(host, guest, method="energy", seed=0)[0]

    assert complex_.atoms.info["dispersion_model"] == "UFF"
    placed = complex_.atoms[complex_.guest_atom_indices]
    assert np.linalg.norm(placed.get_center_of_mass() - host.get_center_of_mass()) < 0.1
    assert complex_.atoms.info["interaction_energy"] == pytest.approx(
        dispersion.uff_interaction_energy([host, placed])
    )


def test_xtb_without_tblite_raises_a_clear_error(no_tblite):
    from cage_isomer_builder.utils.tblite_optimise import run_tblite_optimisation

    with pytest.raises(ImportError, match=r"cage_isomer_builder\[xtb\]"):
        run_tblite_optimisation(molecule("CH4"))
    with pytest.raises(ImportError, match=r"cage_isomer_builder\[xtb\]"):
        tblite_charges.predict_tblite_charges(molecule("CH4"))


def test_uff4mof_optimise_works_without_tblite(no_tblite):
    """UFF4MOF runs without the Coulomb term instead of failing."""
    from cage_isomer_builder.cage import Tri4Di6CageBuilder

    cage = Tri4Di6CageBuilder(
        node=str(DATA_DIR / "uio66_tri_node.xyz"),
        linker=str(DATA_DIR / "bdc.xyz"),
    )
    cage.build()
    with pytest.warns(UserWarning, match="tblite is not installed"):
        optimised = cage.optimise(method="UFF4MOF", steps=2)
    assert not np.any(optimised.get_initial_charges())
