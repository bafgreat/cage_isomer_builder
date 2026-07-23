#!/usr/bin/env python3
"""
Standalone smoke tests for RSA host-guest docking
(cage_isomer_builder.utils.functionalise) and its writers
(cage_isomer_builder.utils.read_write.write_host_guest_complex).

Run directly: python3 tests/test_guest_docking.py

Uses a buckyball (ase.build.molecule("C60")) as the host - a real,
fast-to-construct enclosed cavity, so pywindow's pore analysis succeeds
without needing a full cage build.
"""

import warnings
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import read

from cage_isomer_builder.utils.functionalise import (
    HostGuestComplex,
    max_guests_in_host,
    place_guest_in_host,
)
from cage_isomer_builder.utils.read_write import write_host_guest_complex

DATA_DIR = Path(__file__).parent / "data"


def _host_and_guest():
    host = molecule("C60")
    guest = read(str(DATA_DIR / "methane.xyz"))
    return host, guest


def test_max_guests_in_host_finds_a_real_capacity() -> None:
    host, guest = _host_and_guest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_max = max_guests_in_host(host, guest, seed=0)
    assert n_max > 0, "expected at least one methane to fit inside C60"
    print(f"test_max_guests_in_host_finds_a_real_capacity passed: n_max={n_max}")


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
    print("test_place_guest_in_host_respects_n_guests passed")


def test_complexes_are_distinct_and_well_spread() -> None:
    """
    Different complexes from one call must not repeat the same guest
    position - this was the original bug (n_complexes independently
    resampling could put every complex's guest in the same spot).
    """
    host, guest = _host_and_guest()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        complexes = place_guest_in_host(host, guest, n_guests=1, n_complexes=4, seed=1)

    coms = [c.atoms.positions[c.guest_atom_indices].mean(axis=0) for c in complexes]
    unique = {tuple(np.round(com, 2)) for com in coms}
    assert len(unique) == len(coms), f"expected {len(coms)} distinct positions, got {unique}"
    print("test_complexes_are_distinct_and_well_spread passed:", coms)


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
    # two different guest copies - encapsulation is non-covalent.
    cross_block = complex_.bond_matrix[:n_host, n_host:]
    assert not np.any(cross_block), "host-guest cross terms must stay zero"
    print("test_returns_host_guest_complex_with_consistent_bond_matrix passed")


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
    print("test_n_guests_beyond_capacity_warns_and_caps passed:",
          f"requested {n_max + 50}, got {len(complexes[0].guest_labels)}")


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
    print("test_write_host_guest_complex_formats passed")


if __name__ == "__main__":
    import tempfile

    test_max_guests_in_host_finds_a_real_capacity()
    test_place_guest_in_host_respects_n_guests()
    test_complexes_are_distinct_and_well_spread()
    test_returns_host_guest_complex_with_consistent_bond_matrix()
    test_n_guests_beyond_capacity_warns_and_caps()
    with tempfile.TemporaryDirectory() as tmp:
        test_write_host_guest_complex_formats(Path(tmp))

    print("\nAll guest-docking tests passed.")
