#!/usr/bin/env python3
"""
Vertex codes of the UiO-66 tetrahedral pore: unique vertex names, exact
conversion between vertex and site names, and agreement with the point
group found independently from the atoms.
"""

import itertools
import os

import pytest

from cage_isomer_builder import example_data
from cage_isomer_builder.cage import CageBuilder, Tri4Di6CageBuilder
from cage_isomer_builder.utils.sites import read_isomer
from cage_isomer_builder.cli import vertex_name_main
from cage_isomer_builder.utils.vertex_codes import (
    VertexScheme,
    environment_label,
    environments,
    mirror_label,
)

# The 11 environments of a 3-connected vertex, as tabulated by hand.
TABLE = {"000", "100", "400", "110", "440", "140", "111", "114", "144", "444", "104"}


@pytest.fixture(scope="module")
def cage():
    cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
                              linker=example_data.path("bdc"), scale_multiplier=0.225)
    cage.build()
    cage.optimise(include_charges=False, logfile=None)
    cage.functionalise()
    return cage


@pytest.fixture(scope="module")
def isomers(cage):
    return cage.enumerate_isomers(generate_files=False)


def test_labels_are_the_tabulated_environments() -> None:
    assert set(environments(3)) == TABLE
    for word in map("".join, itertools.product("014", repeat=3)):
        label = environment_label(word)
        assert label in TABLE
        assert mirror_label(mirror_label(label)) == label
    # Mirror pairs of the table (equal energies) and the self-mirror ones.
    assert {mirror_label(a): a for a in ("100", "110", "111", "114")} == \
        {"400": "100", "440": "110", "444": "111", "144": "114"}
    assert mirror_label("140") == "140" and mirror_label("104") == "104"


def test_hand_assigned_example(cage) -> None:
    # Assigned by eye: 000 114 100 140 (in no particular vertex order).
    scheme = cage.vertex_scheme()
    iso = [0, 4, 9, 13, 16, 22]
    assert sorted(scheme.labels(iso)) == sorted(["000", "114", "100", "140"])
    assert scheme.name(iso) == "000-100-141-401"


def test_vertex_names_are_unique_and_convert_back(cage, isomers) -> None:
    scheme = cage.vertex_scheme()
    assert len(isomers) == 176
    names = [scheme.name(iso) for iso in isomers]
    assert len(set(names)) == 176
    for iso, name in zip(isomers, names):
        assert scheme.isomer(name) == sorted(iso)
        assert scheme.resolve(name) == scheme.resolve("-".join(map(str, iso))) == sorted(iso)
        assert all(label in TABLE for label in scheme.labels(iso))


def test_every_placement_has_its_own_name(cage) -> None:
    # All 4**6 placements, not just the 176 representatives.
    scheme = cage.vertex_scheme()
    seen = set()
    for choice in itertools.product(range(4), repeat=6):
        iso = [4 * L + c for L, c in enumerate(choice)]
        name = scheme.name(iso)
        assert scheme.isomer(name) == iso
        seen.add(name)
    assert len(seen) == 4 ** 6


def test_labels_follow_the_point_group(cage, isomers) -> None:
    # Rotations keep every vertex environment; reflections turn each into
    # its mirror image. The operations come from the symmetry detector.
    scheme = cage.vertex_scheme()
    library = cage._get_transformation_library(cage._fg_anchors.copy())
    operations = list(zip(*library))
    assert len(operations) == 23                    # Td without the identity
    for iso in isomers:
        labels = sorted(scheme.labels(iso))
        mirrored = sorted(mirror_label(c) for c in labels)
        kept = 1                                     # the identity
        for op in operations:
            image = sorted(scheme.labels([op[s] for s in iso]))
            assert image in (labels, mirrored)
            kept += image == labels
        assert kept == (24 if labels == mirrored else 12)


def test_files_carry_both_names(cage, isomers, tmp_dir) -> None:
    by_sites, by_vertex = tmp_dir / "sites", tmp_dir / "vertex"
    cage.enumerate_isomers(output_path=str(by_sites))
    cage.enumerate_isomers(output_path=str(by_vertex), naming="vertex")
    scheme = cage.vertex_scheme()
    assert len(os.listdir(by_sites)) == len(os.listdir(by_vertex)) == 176
    for iso in isomers:
        site_name, vertex_name = "-".join(map(str, iso)), scheme.name(iso)
        a = read_isomer(str(by_sites / f"{site_name}.xyz"))
        b = read_isomer(str(by_vertex / f"{vertex_name}.xyz"))
        for info in (a.info, b.info):
            assert info["isomer"] == site_name
            assert info["vertex_name"] == vertex_name
            assert info["vertex_labels"] == "-".join(scheme.labels(iso))
        assert (a.positions == b.positions).all()
        assert a.get_chemical_symbols() == b.get_chemical_symbols()


def test_errors(cage) -> None:
    scheme = cage.vertex_scheme()
    with pytest.raises(ValueError, match="both on linker"):
        scheme.name([0, 1, 4, 8, 12, 16])
    with pytest.raises(ValueError, match="not a vertex name"):
        scheme.isomer("000-100")
    with pytest.raises(ValueError, match="naming"):
        cage.enumerate_isomers(generate_files=False, naming="bogus")
    mof = CageBuilder.load_from_file(example_data.path("RUBTAK01"), primitive=True)
    mof.functionalise()
    with pytest.raises(ValueError, match="finite cage"):
        mof.vertex_scheme()
    with pytest.raises(ValueError, match="finite cage"):
        mof.enumerate_isomers(generate_files=False, limit=1, naming="vertex")


def test_files_carry_the_scheme(cage, tmp_dir) -> None:
    cage.enumerate_isomers(output_path=str(tmp_dir), limit=3)
    for name in os.listdir(tmp_dir):
        info = read_isomer(str(tmp_dir / name)).info
        assert VertexScheme.from_info(info) == cage.vertex_scheme()


def test_command_line(cage, tmp_dir, capsys) -> None:
    scheme = cage.vertex_scheme()
    by_sites = tmp_dir / "sites"
    isomers = cage.enumerate_isomers(output_path=str(by_sites))
    ref = str(by_sites / "0-4-8-12-16-20.xyz")

    # Names, either way, from any file of the cage.
    assert vertex_name_main(["--ref", ref, "0-4-9-13-16-22", "000-100-141-401"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "0-4-9-13-16-22 -> 000-100-141-401   labels 000-100-114-140",
        "000-100-141-401 -> 0-4-9-13-16-22   labels 000-100-114-140"]

    # Files: a site-named copy gets its vertex name and back, unchanged.
    for iso in isomers[:20]:
        site, vertex = "-".join(map(str, iso)), scheme.name(iso)
        src = by_sites / f"{site}.xyz"
        assert vertex_name_main([str(src), "-o", str(tmp_dir / "v")]) == 0
        assert vertex_name_main([str(tmp_dir / "v" / f"{vertex}.xyz"),
                                 "-o", str(tmp_dir / "s")]) == 0
        assert (tmp_dir / "s" / f"{site}.xyz").read_bytes() == src.read_bytes()
    capsys.readouterr()

    # Errors: an existing target, a bad name, a file without vertex codes.
    assert vertex_name_main([ref, "-o", str(tmp_dir / "v")]) == 1
    assert "exists" in capsys.readouterr().err
    assert vertex_name_main(["--ref", ref, "0-1-4-8-12-16"]) == 1
    assert "both on linker" in capsys.readouterr().err
    plain = tmp_dir / "plain.xyz"
    plain.write_text("1\nisomer=0\nH 0 0 0\n")
    assert vertex_name_main([str(plain)]) == 1
    assert "no vertex codes" in capsys.readouterr().err
