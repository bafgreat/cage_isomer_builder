#!/usr/bin/env python3
"""
Standalone smoke tests for host-guest docking (RSA and energy methods)
(cage_isomer_builder.utils.functionalise) and its writers
(cage_isomer_builder.utils.read_write.write_host_guest_complex).

Run directly: python3 tests/test_guest_docking.py

Random packing uses the tetrahedral cage of UiO-66 (a Tri4Di6 cage of the
UiO-66 node and 1,4-phenylene, built once and optimised): a real cavity
(7.5 A) with real windows (4.1 A). The energy method uses methane in C60,
whose best site is the cage centre by symmetry.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from ase.build import molecule
from ase.io import read

from cage_isomer_builder.utils.functionalise import (
    HostGuestComplex,
    max_guests_in_host,
    place_guest_in_host,
)
from cage_isomer_builder.utils.read_write import write_host_guest_complex

DATA_DIR = Path(__file__).parent / "data"


from functools import lru_cache


@lru_cache(maxsize=1)
def _uio66_cage():
    from cage_isomer_builder.cage import Tri4Di6CageBuilder

    cage = Tri4Di6CageBuilder(node=str(DATA_DIR / "uio66_tri_node.xyz"),
                              linker=str(DATA_DIR / "bdc.xyz"), scale_multiplier=0.225)
    cage.build()
    cage.optimise(include_charges=False, logfile=None)
    return cage.to_ase()


def _host_and_guest():
    """UiO-66 tetrahedral cage and methane (random packing)."""
    return _uio66_cage().copy(), read(str(DATA_DIR / "methane.xyz"))


def _c60_and_methane():
    """C60 and methane (energy method: the best site is the centre)."""
    return molecule("C60"), read(str(DATA_DIR / "methane.xyz"))


def test_max_guests_in_host_finds_a_real_capacity() -> None:
    host, guest = _host_and_guest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_max = max_guests_in_host(host, guest)
    assert 0 < n_max <= 8, f"a 7.5 A cavity holds a few methanes, not {n_max}"


def test_place_guest_in_host_respects_n_guests() -> None:
    """n_guests=1 should place exactly one guest copy per complex, not more."""
    host, guest = _host_and_guest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        complexes = place_guest_in_host(host, guest, n_guests=1, n_complexes=3, seed=0)

    assert len(complexes) == 3
    for complex_ in complexes:
        assert len(complex_.guest_labels) == 1
        assert len(complex_.guest_atom_indices) == len(guest)


def test_complexes_are_distinct_and_well_spread() -> None:
    """
    Different complexes from one call must not repeat the same guest
    position.
    """
    host, guest = _host_and_guest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        complexes = place_guest_in_host(host, guest, n_guests=1, n_complexes=4, seed=1)

    coms = [c.atoms.positions[c.guest_atom_indices].mean(axis=0) for c in complexes]
    unique = {tuple(np.round(com, 2)) for com in coms}
    assert len(unique) == len(coms), f"expected {len(coms)} distinct positions, got {unique}"


def test_returns_host_guest_complex_with_consistent_bond_matrix() -> None:
    host, guest = _host_and_guest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        complexes = place_guest_in_host(host, guest, n_guests=2, n_complexes=1, seed=2)
    complex_ = complexes[0]

    assert isinstance(complex_, HostGuestComplex)
    n_host = len(host)
    n_total = len(complex_.atoms)
    assert n_total == n_host + 2 * len(guest)
    assert complex_.bond_matrix.shape == (n_total, n_total)
    assert np.array_equal(complex_.bond_matrix, complex_.bond_matrix.T), "bond matrix must be symmetric"

    # No bond should ever be recorded between host and guest, or between
    # two different guest copies: encapsulation is non-covalent.
    cross_block = complex_.bond_matrix[:n_host, n_host:]
    assert not np.any(cross_block), "host-guest cross terms must stay zero"


def test_n_guests_beyond_capacity_warns_and_caps() -> None:
    host, guest = _host_and_guest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_max = max_guests_in_host(host, guest, seed=0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        complexes = place_guest_in_host(host, guest, n_guests=n_max + 50, seed=0)

    assert len(complexes[0].guest_labels) <= n_max
    assert any("capping" in str(w.message) for w in caught)


def test_write_host_guest_complex_formats(tmp_dir: Path) -> None:
    host, guest = _host_and_guest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        complex_ = place_guest_in_host(host, guest, n_guests=1, seed=0)[0]

    for suffix in (".gin", ".run", ".xyz"):
        out_path = tmp_dir / f"complex{suffix}"
        write_host_guest_complex(complex_, str(out_path))
        assert out_path.exists() and out_path.stat().st_size > 0

    bare_atoms = complex_.atoms
    try:
        write_host_guest_complex(bare_atoms, str(tmp_dir / "bad.gin"))
        raised = False
    except TypeError:
        raised = True
    assert raised, "plain ase.Atoms with a .gin path should raise TypeError"


def test_energy_method_finds_the_best_site() -> None:
    """
    Methane in C60: by symmetry the best site is the cage centre. The
    energy method must find it, and must beat every pose of a brute-force
    search scored with full dftd4 plus the UFF repulsion.
    """
    from cage_isomer_builder.utils.dispersion import (
        d4_interaction_energy,
        uff_interaction_energy,
    )

    def full_energy(fragments):
        return d4_interaction_energy(fragments) + uff_interaction_energy(
            fragments, repulsion_only=True
        )
    from cage_isomer_builder.utils.geometry import random_rotation_matrix

    host, guest = _c60_and_methane()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        complexes = place_guest_in_host(host, guest, method="energy", seed=0)

    assert len(complexes) == 1
    complex_ = complexes[0]
    assert isinstance(complex_, HostGuestComplex)
    placed = complex_.atoms[complex_.guest_atom_indices]
    centre = host.get_center_of_mass()
    assert np.linalg.norm(placed.get_center_of_mass() - centre) < 0.1

    energy = complex_.atoms.info["interaction_energy"]
    assert energy == pytest.approx(full_energy([host, placed]))

    np.random.seed(1)
    centred = guest.copy()
    centred.translate(-centred.get_center_of_mass())
    for _ in range(100):
        trial = centred.copy()
        trial.positions = trial.positions @ random_rotation_matrix().T + centre
        trial.translate(np.random.uniform(-0.5, 0.5, 3))
        assert energy <= full_energy([host, trial]) + 1e-6


def test_energy_method_warns_when_guests_do_not_fit() -> None:
    """Only one methane fits in C60, so asking for two caps with a warning."""
    host, guest = _c60_and_methane()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        complexes = place_guest_in_host(host, guest, n_guests=2, method="energy", seed=0)

    assert len(complexes[0].guest_labels) == 1
    assert any("no allowed site" in str(w.message) for w in caught)


def test_unknown_method_raises() -> None:
    host, guest = _host_and_guest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="method="):
            place_guest_in_host(host, guest, method="best")


def test_cavity_and_windows_are_physical() -> None:
    from cage_isomer_builder.utils.cavity import largest_window, pore_sphere

    host = _uio66_cage()
    centre, diameter = pore_sphere(host)
    assert 7.0 < diameter < 8.0                    # UiO-66 tetrahedral cage
    assert 3.5 < largest_window(host, centre) < 4.5
    c60 = molecule("C60")
    assert largest_window(c60) is None             # closed
    assert pore_sphere(c60)[1] < 3.8               # too small for methane
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert max_guests_in_host(c60, read(str(DATA_DIR / "methane.xyz")), seed=0) == 0


def test_guest_wider_than_every_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="window"):
        max_guests_in_host(_uio66_cage(), read(str(DATA_DIR / "ibuprofen.xyz")), seed=0)


if __name__ == "__main__":
    import tempfile

    test_max_guests_in_host_finds_a_real_capacity()
    test_place_guest_in_host_respects_n_guests()
    test_complexes_are_distinct_and_well_spread()
    test_returns_host_guest_complex_with_consistent_bond_matrix()
    test_n_guests_beyond_capacity_warns_and_caps()
    with tempfile.TemporaryDirectory() as tmp:
        test_write_host_guest_complex_formats(Path(tmp))
    test_energy_method_finds_the_best_site()
    test_energy_method_warns_when_guests_do_not_fit()
    test_unknown_method_raises()

    print("\nAll guest-docking tests passed.")
