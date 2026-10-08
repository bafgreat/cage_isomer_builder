"""
Tests for the D4 dispersion module (cage_isomer_builder.utils.dispersion).

The fast pair kernel is checked against dftd4 itself, so these tests do not
depend on any value hard-coded here.
"""

import numpy as np
import pytest
from ase.build import molecule

from cage_isomer_builder.utils.dispersion import (
    D4_PARAMETERS,
    D4PairModel,
    d4_interaction_energy,
    uff_interaction_energy,
)


def _dimer(separation):
    host = molecule("C6H6")
    guest = molecule("CH3CH2OH")
    guest.translate(-guest.get_center_of_mass() + [0.0, 0.0, separation])
    return host, guest


def _kernel_energy(model, host, guest, min_distance=None):
    if min_distance is None:
        min_distance = np.zeros((len(host), len(guest)))
    return model.score(
        np.arange(len(host)), host.positions,
        model.offsets[1] + np.arange(len(guest)), guest.positions[None],
        min_distance,
    )


@pytest.mark.parametrize("separation", [3.6, 4.5, 6.0])
def test_kernel_matches_dftd4_two_body(separation):
    """The numba kernel gives dftd4's two-body energy plus the UFF repulsion."""
    host, guest = _dimer(separation)
    model = D4PairModel(host, [guest])
    energy, penalty, overlap = _kernel_energy(model, host, guest)

    two_body = dict(D4_PARAMETERS["pbe"], s9=0.0)
    reference = d4_interaction_energy([host, guest], two_body) + uff_interaction_energy(
        [host, guest], repulsion_only=True
    )
    assert penalty[0] == 0.0 and overlap[0] == 0.0
    assert energy[0] == pytest.approx(reference, rel=0.01)


def test_clash_adds_penalty_and_reports_overlap():
    host, guest = _dimer(3.6)
    model = D4PairModel(host, [guest])
    limit = np.full((len(host), len(guest)), 10.0)
    _, penalty, overlap = _kernel_energy(model, host, guest, limit)
    assert penalty[0] > 0.0
    assert 0.0 < overlap[0] < 10.0


def test_interaction_energy_is_attractive():
    host, guest = _dimer(3.6)
    assert d4_interaction_energy([host, guest]) < 0.0


def test_unknown_method_raises():
    host, guest = _dimer(3.6)
    with pytest.raises(ValueError, match="Unknown D4 method"):
        D4PairModel(host, [guest], method="not-a-functional")
