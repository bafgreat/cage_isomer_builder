#!/usr/bin/env python3
"""
MOFs with more than one kind of linker.

ABIXEQ: a Co MOF with two different linkers (C18H18N4 with 8 aromatic C-H,
and a C6H4 phenylene with 4). ABINOQ: a Zn MOF whose linker is a
bipyridine, with terminal thiophenolates bound to one metal each, which
are not linkers. Every count is checked against brute force over all raw
configurations under the structure's own space group.
"""

import itertools
import warnings

import numpy as np
import pytest

from cage_isomer_builder.cage import CageBuilder
from cage_isomer_builder.cage.base import find_linkers
from cage_isomer_builder.utils import rgroup
from cage_isomer_builder.utils.distributions import pair_descriptor
from cage_isomer_builder.utils.ensemble import ensemble_descriptor
from cage_isomer_builder.utils.features import aromatic_rings, bond_graph
from cage_isomer_builder.utils.sites import site_array


def _load(name, **kwargs):
    m = CageBuilder.load_from_file(f"tests/data/{name}.cif")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.functionalise(**kwargs)
    return m


@pytest.fixture(scope="module")
def two_linkers():
    return _load("ABIXEQ")


def _configurations(spec):
    layout = spec.layout
    local = []
    for p, comp in enumerate(spec.patterns):
        F = layout.type_size(spec.pattern_types[p])
        labels = [0] * (F - sum(comp))
        for c, k in enumerate(comp):
            labels += [c + 1] * k
        local.append(sorted(set(itertools.permutations(labels))))
    for assignment in rgroup._pattern_assignments(spec):
        for combo in itertools.product(*(local[p] for p in assignment)):
            yield tuple(itertools.chain.from_iterable(combo))


def _brute(spec, group):
    seen, n = set(), 0
    for f in _configurations(spec):
        if f in seen:
            continue
        n += 1
        arr = np.array(f)
        for inv in group.inverse:
            seen.add(tuple(arr[inv]))
    return n


def test_terminal_ligands_are_not_linkers() -> None:
    from ase.io import read

    _, _, names, _ = find_linkers(read("tests/data/ABINOQ.cif"))
    assert names == ["C10H8N2"]                     # bipyridine, not C6H5S
    m = _load("ABINOQ")
    assert m.slot_layout().type_names == ("C10H8N2",)


def test_both_linker_types_are_found(two_linkers) -> None:
    layout = two_linkers.slot_layout()
    assert layout.type_names == ("C18H18N4", "C6H4")
    assert layout.sizes == (8, 4)
    assert two_linkers._fg_per_linker is None


def test_slots_are_aromatic_c_h_only(two_linkers) -> None:
    atoms = two_linkers.to_ase()
    adj = bond_graph(atoms)
    ring_c = {k for ring in aromatic_rings(atoms, adj) for k in ring}
    for s in two_linkers._fg_anchor_indices:
        carbon = [j for j in adj[s] if atoms.numbers[j] > 1][0]
        assert carbon in ring_c
    # the linker's CH2 groups give 8 extra (non-aromatic) sites when allowed
    assert len(_load("ABIXEQ", aromatic_only=False)._fg_anchor_indices) == 20


@pytest.mark.parametrize("case", [
    dict(groups="A"),
    dict(groups="AB"),
    dict(groups={"C18H18N4": "AB", "C6H4": "A"}),
    dict(groups={"C18H18N4": "A", "C6H4": ""}),
    dict(linker_patterns={"C18H18N4": {"A": 1}, "C6H4": {"B": 1}}),
])
def test_counts_match_brute_force(two_linkers, case) -> None:
    group = two_linkers._slot_group()
    spec = rgroup.make_spec(two_linkers.slot_layout(), None, **{"groups": "A", **case})
    expected = _brute(spec, group)
    assert two_linkers.count_rgroup_isomers(**case) == expected
    assert len(two_linkers.enumerate_rgroup_isomers(generate_files=False, **case)) == expected


def test_single_group_isomers(two_linkers) -> None:
    assert len(two_linkers.enumerate_isomers(generate_files=False)) \
        == two_linkers.count_unique_isomers_burnside()


def test_defects_per_linker_type(two_linkers) -> None:
    layout = two_linkers.slot_layout()
    for missing in ({"C18H18N4": 1}, {"C6H4": 1}):
        n = two_linkers.count_defect_isomers(missing, groups="A")
        structures = two_linkers.enumerate_defect_isomers(missing, groups="A",
                                                          generate_files=False)
        assert len(structures) == n
        t = layout.type_names.index(next(iter(missing)))
        first_of_type = layout.types.index(t)
        size = len(two_linkers._linker_atom_indices[two_linkers._slot_linkers[first_of_type]])
        for iso, atoms in structures:
            assert len(atoms) == len(two_linkers.to_ase()) - size
            labels = site_array(atoms)
            assert set(labels[labels > 0]) == {1}          # only R1 = A remains
            assert atoms.info["rgroup_labels"] == "A"
    with pytest.raises(ValueError):
        two_linkers.count_defect_isomers(1)                 # ambiguous with two types


def test_periodic_ensemble_with_two_linker_types(two_linkers) -> None:
    layout = two_linkers.slot_layout()
    case = dict(linker_patterns={"C18H18N4": {"A": 1}, "C6H4": {"B": 1}})
    slots = two_linkers.slot_features()
    slots.owners = layout.slot_linker
    spec = rgroup.make_spec(layout, None, **case)
    grid, R = np.linspace(0, 14, 701), 12.0
    acc, n = {}, 0
    for f in _configurations(spec):
        n += 1
        lab = np.array(f)
        active = list(np.nonzero(lab)[0])
        sub = slots.subset(active)
        sub.labels = np.array([spec.labels[lab[x] - 1] for x in active], dtype="<U16")
        d = pair_descriptor(sub, by="label", max_distance=R)
        for k in d.keys():
            acc[k] = acc.get(k, 0) + d.kde(grid, [k])
    exact = ensemble_descriptor(slots, layout, by="label", max_distance=R, **case)
    assert set(exact.keys()) == set(acc)
    for k in acc:
        assert np.abs(exact.kde(grid, [k]) - acc[k] / n).max() < 1e-9
    built = two_linkers.get_ensemble_descriptor(by="label", max_distance=R, **case)
    assert np.abs(built.kde(grid) - exact.kde(grid)).max() < 1e-9


def test_statistics_and_features(two_linkers) -> None:
    stats = two_linkers.get_statistics(max_distance=12.0)
    assert stats.n_linkers == 2 and stats.n_fg_slots == 12
    assert sum(stats.total_fg_pairs) == sum(stats.other_pairs)      # periodic: unclassified
    counts = two_linkers.get_features().counts()
    assert counts["fg"] == 12 and counts["pi"] == 5
