#!/usr/bin/env python3
"""
Tests for the descriptor modules: R sites, feature detection, pair
distributions (additivity, periodic images), exact isomer-ensemble
averages (multivariate, ratios, defects), flexibility and matching.

Every exact result is checked against an independent computation: brute
force over all raw configurations, explicit orbit counting, supercells,
or a hand-derivable molecule.
"""

import itertools
import warnings

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read

from cage_isomer_builder.utils import rgroup
from cage_isomer_builder.utils.distributions import (
    Descriptor, PairSet, additivity_residual, assemble, overlap, pair_descriptor,
)
from cage_isomer_builder.utils.ensemble import (
    boltzmann_weights, ensemble_descriptor, interpolate_descriptor,
    isomer_ensemble_descriptor, orbit_sizes, ratio_series, vacancy_pattern,
)
from cage_isomer_builder.utils.features import (
    FeatureSet, bond_graph, coordination_numbers, detect_features,
)
from cage_isomer_builder.utils.flexibility import rotate_rotor, scale_structure
from cage_isomer_builder.utils.matching import conformer_descriptor, match_descriptors
from cage_isomer_builder.utils.sites import (
    mark_site, read_isomer, site_indices, write_isomer,
)

GRID = np.linspace(0.0, 30.0, 1501)


@pytest.fixture(scope="module")
def cage():
    """Tet2Di4 (D4h): 4 phenylenediamine linkers x 4 slots, small enough
    for brute force over every raw configuration."""
    from cage_isomer_builder.cage import Tet2Di4CageBuilder

    c = Tet2Di4CageBuilder()
    c.build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c.functionalise()
    return c


def _all_configurations(spec):
    """Every labelling allowed by spec, no symmetry (as in test_rgroup)."""
    F = spec.fg_per_linker
    local = []
    for comp in spec.patterns:
        labels = [0] * (F - sum(comp))
        for c, k in enumerate(comp):
            labels += [c + 1] * k
        local.append(sorted(set(itertools.permutations(labels))))
    ids = [p for p, n in enumerate(spec.pattern_counts) for _ in range(n)]
    for assignment in set(itertools.permutations(ids)):
        for combo in itertools.product(*(local[p] for p in assignment)):
            yield tuple(itertools.chain.from_iterable(combo))


def _brute_orbits(spec, group):
    seen, n = set(), 0
    for f in _all_configurations(spec):
        if f in seen:
            continue
        n += 1
        arr = np.array(f)
        for inv in group.inverse:
            seen.add(tuple(arr[inv]))
    return n


# ---------------------------------------------------------------------------
# R sites
# ---------------------------------------------------------------------------

def test_r_site_round_trip(tmp_path) -> None:
    atoms = molecule("C6H6")
    hs = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "H"]
    mark_site(atoms, hs[0], 1)
    mark_site(atoms, hs[2], 2)
    path = write_isomer(tmp_path / "iso.xyz", atoms)
    back = read_isomer(path)
    assert back.info["r_sites"] == {hs[0]: "R1", hs[2]: "R2"}
    assert site_indices(back) == sorted([hs[0], hs[2]])
    assert back.get_chemical_symbols()[hs[0]] == "X"
    plain = read(path)                      # readable without our reader too
    assert plain.info["r_sites"] == f"{hs[0]}:R1 {hs[2]}:R2"
    # the map is rebuilt from the per-atom array after deleting atoms
    trimmed = back[[i for i in range(len(back)) if i != 0]]
    assert set(read_isomer(write_isomer(tmp_path / "t.xyz", trimmed)).info["r_sites"].values()) \
        == {"R1", "R2"}


def test_r_labels_in_smiles_and_sdf() -> None:
    pytest.importorskip("rdkit")
    from rdkit import Chem

    atoms = molecule("C6H6")
    hs = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "H"]
    mark_site(atoms, hs[0], 1)
    mark_site(atoms, hs[3], 2)
    bonds = np.zeros((12, 12))
    for i, j in bond_graph(atoms).items():
        for k in j:
            bonds[i, k] = 1.5 if atoms.numbers[i] == atoms.numbers[k] == 6 else 1.0
    smiles = rgroup.rgroup_smiles(atoms, bonds)
    assert "_R1" in smiles and "_R2" in smiles
    assert "c1" in smiles                    # aromatic, not C=C=C
    block = Chem.MolToMolBlock(rgroup.rgroup_atoms_to_rdkit(atoms, bonds))
    assert "M  RGP  2" in block


def test_isomer_files_use_r_sites(cage, tmp_path) -> None:
    import os

    cage.enumerate_isomers(output_path=str(tmp_path), limit=3)
    files = sorted(os.listdir(tmp_path))
    atoms = read_isomer(tmp_path / files[0])
    n_linkers = len(cage._fg_anchors) // cage._fg_per_linker
    assert len(atoms.info["r_sites"]) == n_linkers
    assert set(atoms.info["r_sites"].values()) == {"R1"}
    assert "At" not in atoms.get_chemical_symbols()


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def test_features_of_reference_molecules() -> None:
    expected = {
        "C6H6": {"pi": 1},
        "H2O": {"acceptor": 1, "donor": 2},
        "NH3": {"acceptor": 1, "donor": 3},
        "C5H5N": {"acceptor": 1, "pi": 1},
        "CH3CONH2": {"acceptor": 1, "donor": 2},       # planar amide N excluded
        "CO2": {"acceptor": 2},
        "CH4": {},
    }
    for name, counts in expected.items():
        assert detect_features(molecule(name)).counts() == counts, name

    benzene = detect_features(molecule("C6H6"))
    assert abs(abs(benzene.vectors[0][2]) - 1) < 1e-6        # normal along z
    assert not benzene.directed[0]
    water = molecule("H2O")
    acc = detect_features(water).select("acceptor")
    bisector = -(water.positions[1] + water.positions[2] - 2 * water.positions[0])
    bisector /= np.linalg.norm(bisector)
    assert acc.vectors[0] @ bisector > 0.999


def test_open_metal_sites_appear_where_a_ligand_was_removed() -> None:
    sbu = read("tests/data/uio66_sbu.xyz")
    sbu = sbu[[i for i in range(len(sbu)) if sbu[i].symbol != "X"]]
    adj = bond_graph(sbu)
    assert set(coordination_numbers(sbu, adj).values()) == {8}
    assert len(detect_features(sbu, kinds=("open_metal",))) == 0
    c = [i for i, s in enumerate(sbu.get_chemical_symbols()) if s == "C"][0]
    oxygens = [j for j in adj[c] if sbu[j].symbol == "O"]
    defect = sbu[[i for i in range(len(sbu)) if i not in [c] + oxygens]]
    oms = detect_features(defect, kinds=("open_metal",))
    assert len(oms) == 2 and set(oms.labels) == {"Zr(CN=7)"}
    for p, v in zip(oms.positions, oms.vectors):
        gone = min(sbu.positions[oxygens], key=lambda q: np.linalg.norm(q - p))
        assert v @ ((gone - p) / np.linalg.norm(gone - p)) > 0.9


def test_x_site_markers_are_not_metals(cage) -> None:
    linkers, nodes = cage.building_blocks()
    assert len(linkers) == 4 and len(nodes) == 2


# ---------------------------------------------------------------------------
# Pair distributions
# ---------------------------------------------------------------------------

def _random_features(rng, n, kind, owners_from=0, pbc=False, cell=None):
    return FeatureSet.from_rows([
        dict(position=rng.uniform(0, 10, 3), vector=rng.normal(size=3) / 2, kind=kind,
             owner=owners_from + k // 2, atom=k)
        for k in range(n)
    ], cell, np.array([pbc] * 3))


def test_additivity_is_exact_with_cross_terms() -> None:
    rng = np.random.default_rng(0)
    parts = [_random_features(rng, 8, "fg"), _random_features(rng, 5, "pi", 10),
             _random_features(rng, 6, "acceptor", 20)]
    residual, missing = additivity_residual(parts, GRID)
    assert residual < 1e-10
    assert missing == pytest.approx(8 * 5 + 8 * 6 + 5 * 6)
    full = assemble(parts)
    assert full.total() == pytest.approx(19 * 18 / 2 - 4 - 2 - 3)   # minus same-owner pairs


def test_kde_integrates_to_pair_weight() -> None:
    rng = np.random.default_rng(1)
    d = pair_descriptor(_random_features(rng, 10, "fg"))
    assert np.trapezoid(d.kde(np.linspace(-5, 30, 7001)), np.linspace(-5, 30, 7001)) \
        == pytest.approx(d.total(), rel=1e-6)


def test_periodic_counts_do_not_depend_on_the_cell() -> None:
    rng = np.random.default_rng(2)
    cell = np.diag([7.0, 8.0, 9.0])
    n = 6
    frac = rng.uniform(size=(n, 3))
    base = FeatureSet.from_rows([dict(position=f @ cell, kind="pi", owner=-1) for f in frac],
                                cell, np.array([True] * 3))
    shifts = np.array(list(itertools.product(range(2), repeat=3)))
    big_cell = 2 * cell
    rows = [dict(position=(f + s) @ cell, kind="pi", owner=-1) for s in shifts for f in frac]
    supercell = FeatureSet.from_rows(rows, big_cell, np.array([True] * 3))
    d1 = pair_descriptor(base, max_distance=12.0)
    d8 = pair_descriptor(supercell, max_distance=12.0)
    assert d8.total() == pytest.approx(8 * d1.total())
    assert np.abs(d8.kde(GRID) - 8 * d1.kde(GRID)).max() < 1e-8


def test_statistics_match_pair_descriptor(cage) -> None:
    stats = cage.get_statistics()
    n = len(cage._fg_anchors)
    F = cage._fg_per_linker
    assert sum(stats.total_fg_pairs) == n * (n - F) // 2
    parts = (stats.endo_endo_pairs, stats.endo_exo_pairs, stats.exo_exo_pairs, stats.other_pairs)
    assert [sum(x) for x in zip(*parts)] == stats.total_fg_pairs
    # D4h Tet2Di4: two inward and two outward slots per linker
    orient = cage.slot_features().orientation
    assert sorted(orient.tolist()) == ["endo"] * (n // 2) + ["exo"] * (n // 2)


# ---------------------------------------------------------------------------
# Isomer ensembles (multivariate, ratios, defects)
# ---------------------------------------------------------------------------

CASES = [
    dict(groups="A"),
    dict(groups="AB"),
    dict(linker_patterns={"A": 2, "B": 2}),
    dict(linker_patterns={"AB": 1, "A": 2, "": 1}),
    dict(linker_patterns={"A": 3, vacancy_pattern(4): 1}),
]


@pytest.mark.parametrize("case", CASES)
def test_exact_ensemble_equals_brute_force(cage, case) -> None:
    slots = cage.slot_features()
    F = cage._fg_per_linker
    spec = rgroup.make_spec(len(slots) // F, F, case.get("groups", "A"),
                            case.get("linker_patterns"))
    configs = [rgroup.RGroupIsomer(f, spec.labels, ()) for f in _all_configurations(spec)]
    brute = isomer_ensemble_descriptor(slots, configs, F)
    exact = ensemble_descriptor(slots, F, **case)
    assert set(exact.keys()) == set(brute.keys())
    for key in exact.keys():
        assert np.abs(exact.kde(GRID, [key]) - brute.kde(GRID, [key])).max() < 1e-9


def test_orbit_weights_reproduce_the_raw_configuration_count(cage) -> None:
    group = cage._slot_group()
    n, F = len(cage._fg_anchors), cage._fg_per_linker
    for case in CASES:
        isomers = rgroup.enumerate_rgroup_isomers(group, n, F, **case)
        spec = rgroup.make_spec(n // F, F, case.get("groups", "A"), case.get("linker_patterns"))
        assert orbit_sizes(group, isomers).sum() == rgroup.identity_fixed_count(spec)


def test_independent_mode_and_ratio_interpolation(cage) -> None:
    slots = cage.slot_features()
    F = cage._fg_per_linker
    n_linkers = len(slots) // F
    n_pairs_per_isomer = n_linkers * (n_linkers - 1) / 2
    x = 0.3
    ind = ensemble_descriptor(slots, F, mode="independent", fractions={"A": x, "B": 1 - x},
                              by="label")
    assert ind.total([("fg:A", "fg:A")]) == pytest.approx(n_pairs_per_isomer * x * x)
    assert ind.total([("fg:A", "fg:B")]) == pytest.approx(n_pairs_per_isomer * 2 * x * (1 - x))
    series = ratio_series(slots, F, by="label")
    assert sorted(series) == [k / n_linkers for k in range(n_linkers + 1)]
    for k in range(2, n_linkers + 1):
        aa = series[k / n_linkers].total([("fg:A", "fg:A")])
        assert aa == pytest.approx(n_pairs_per_isomer * k * (k - 1) / (n_linkers * (n_linkers - 1)))
    mid = interpolate_descriptor(series, 0.6)
    lo, hi = series[0.5], series[0.75]
    t = (0.6 - 0.5) / 0.25
    assert np.allclose(mid.kde(GRID), (1 - t) * lo.kde(GRID) + t * hi.kde(GRID))


def test_defect_counts_match_brute_force(cage) -> None:
    group = cage._slot_group()
    F = cage._fg_per_linker
    n_linkers = len(cage._fg_anchors) // F
    for k in (1, 2):
        for groups in ("A", "AB", ""):
            patterns = {tuple(groups): n_linkers - k, vacancy_pattern(F): k}
            spec = rgroup.make_spec(n_linkers, F, linker_patterns=patterns)
            assert cage.count_defect_isomers(k, groups=groups) == _brute_orbits(spec, group)


def test_defect_structures_lose_exactly_the_missing_linkers(cage) -> None:
    n_atoms = len(cage.to_ase())
    linker_size = len(cage._linker_atom_indices[0])
    structures = cage.enumerate_defect_isomers(1, generate_files=False)
    assert len(structures) == cage.count_defect_isomers(1)
    for iso, atoms in structures:
        assert len(atoms) == n_atoms - linker_size
        n_sites = int((atoms.arrays["rgroup"] > 0).sum())
        assert n_sites == len(cage._fg_anchors) // cage._fg_per_linker - 1
        assert atoms.info["rgroup_labels"] == "A"


def test_node_defects(cage) -> None:
    # The two nodes of a D4h Tet2Di4 are equivalent.
    assert cage.count_node_defect_isomers(1) == 1
    assert cage.count_node_defect_isomers(2) == 1
    (removed, atoms), = cage.enumerate_node_defect_isomers(1, generate_files=False)
    _, nodes = cage.building_blocks()
    assert len(atoms) == len(cage.to_ase()) - len(nodes[removed[0]])


# ---------------------------------------------------------------------------
# Flexibility
# ---------------------------------------------------------------------------

def test_ring_rotation_preserves_bonds_and_maps_slots_at_180(cage) -> None:
    atoms = cage.to_ase()
    adjacency = cage.adjacency()
    rotors = cage.rotors()
    assert len(rotors) == 4
    bonds = [(i, j) for i in adjacency for j in adjacency[i] if i < j]
    for angle in (37.0, 180.0):
        turned = rotate_rotor(atoms, rotors[0], angle)
        worst = max(abs(atoms.get_distance(i, j) - turned.get_distance(i, j)) for i, j in bonds)
        assert worst < 1e-9
    turned = rotate_rotor(atoms, rotors[0], 180.0)
    slots = [s for s in cage._fg_anchor_indices if s in rotors[0].moving]
    gap = np.linalg.norm(turned.positions[slots][:, None] - atoms.positions[slots][None], axis=2)
    # The unoptimised (RDKit-embedded) ring is not exactly two-fold
    # symmetric about its C1-C4 axis (~0.16 A off); an optimised ring is
    # within ~0.04 A.
    assert gap.min(axis=1).max() < 0.25
    assert np.allclose(rotate_rotor(atoms, rotors[0], 360.0).positions, atoms.positions)


def test_rotation_ensemble_conserves_pair_count(cage) -> None:
    static = cage.get_ensemble_descriptor(groups="A")
    flexible = cage.get_ensemble_descriptor(groups="A", rotation_angles=np.arange(0, 360, 45))
    assert flexible.total() == pytest.approx(static.total())
    zero = cage.get_ensemble_descriptor(groups="A", rotation_angles=[0.0])
    assert np.abs(zero.kde(GRID) - static.kde(GRID)).max() < 1e-9


def test_scaling(cage) -> None:
    atoms = cage.to_ase()
    linkers, nodes = cage.building_blocks()
    assert np.allclose(scale_structure(atoms, linkers + nodes, 1.0, cage.pore_centre()).positions,
                       atoms.positions)
    pairs, distances, slopes = cage.distance_scaling((0.9, 1.0, 1.1))
    F = cage._fg_per_linker
    same = pairs[:, 0] // F == pairs[:, 1] // F
    assert np.abs(slopes[same]).max() < 1e-9
    assert (slopes[~same] > 0).all()


# ---------------------------------------------------------------------------
# Guests and matching
# ---------------------------------------------------------------------------

def test_boltzmann_weights() -> None:
    w = boltzmann_weights([0.0, 0.05], temperature=298.15)
    assert w[1] / w[0] == pytest.approx(np.exp(-0.05 / (8.617333262e-5 * 298.15)))


def test_conformer_descriptor_averages_with_weights() -> None:
    water = molecule("H2O")
    stretched = water.copy()
    stretched.positions[1] += 0.2 * (stretched.positions[1] - stretched.positions[0])
    d = conformer_descriptor([water, stretched], weights=[3, 1])
    assert d.total() == pytest.approx(3.0)           # 2 acceptor-donor + 1 donor-donor
    single = conformer_descriptor([water])
    assert d.total([("acceptor", "donor")]) == pytest.approx(single.total([("acceptor", "donor")]))


def _desc(key, distances):
    out = Descriptor()
    out.add(key, PairSet(np.asarray(distances, float), np.full(len(distances), np.nan),
                         np.ones(len(distances))))
    return out


def test_matching_rewards_complementary_pairs_at_the_same_distance() -> None:
    guest = _desc(("acceptor", "donor"), [5.0])
    good = _desc(("acceptor", "donor"), [5.0])
    shifted = _desc(("acceptor", "donor"), [6.0])
    wrong_kind = _desc(("acceptor", "acceptor"), [5.0])
    assert match_descriptors(good, guest).score == pytest.approx(1.0, abs=1e-6)
    assert match_descriptors(shifted, guest).score < 0.05
    result = match_descriptors(wrong_kind, guest)
    assert result.score == 0 and result.unmatched == [("acceptor", "donor")]
    # acceptor-acceptor in the guest is served by host donors or open metal sites
    host = _desc(("donor", "open_metal"), [4.0])
    assert match_descriptors(host, _desc(("acceptor", "acceptor"), [4.0])).score \
        == pytest.approx(1.0, abs=1e-6)
    assert overlap(good.kde(GRID), shifted.kde(GRID), GRID) < 0.05


def test_single_feature_guest_is_flagged_undefined() -> None:
    result = match_descriptors(_desc(("pi", "pi"), [5.0]), conformer_descriptor([molecule("C6H6")]))
    assert not result.defined


def test_unbonded_build_warns() -> None:
    from cage_isomer_builder.cage import Tet2Di4CageBuilder

    c = Tet2Di4CageBuilder()
    c.build()
    with pytest.warns(UserWarning, match="not bonded"):
        c.functionalise()


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mof():
    from cage_isomer_builder.cage import CageBuilder

    m = CageBuilder.load_from_file("tests/data/RUBTAK01.cif", primitive=True)
    m.functionalise()
    return m


@pytest.mark.parametrize("case", [dict(groups="A"),
                                  dict(linker_patterns={"A": 5, vacancy_pattern(4): 1})])
def test_periodic_ensemble_equals_brute_force(mof, case) -> None:
    """Every configuration of the cell, repeated periodically, through the
    (supercell-validated) periodic pair_descriptor."""
    from cage_isomer_builder.utils.ensemble import VACANCY

    slots = mof.slot_features()
    F = mof._fg_per_linker
    slots.owners = np.arange(len(slots)) // F
    spec = rgroup.make_spec(len(slots) // F, F, case.get("groups", "A"),
                            case.get("linker_patterns"))
    grid, R = np.linspace(0, 12, 301), 10.0
    acc, n = {}, 0
    for f in _all_configurations(spec):
        n += 1
        lab = np.array(f)
        act = [x for x in np.nonzero(lab)[0] if spec.labels[lab[x] - 1] != VACANCY]
        sub = slots.subset(act)
        sub.labels = np.array([spec.labels[lab[x] - 1] for x in act], dtype="<U16")
        d = pair_descriptor(sub, by="label", max_distance=R)
        for k in d.keys():
            acc[k] = acc.get(k, 0) + d.kde(grid, [k])
    exact = ensemble_descriptor(slots, F, by="label", max_distance=R, **case)
    assert set(exact.keys()) == set(acc)
    for k in acc:
        assert np.abs(exact.kde(grid, [k]) - acc[k] / n).max() < 1e-9


def test_same_linker_pairs_equal_brute_force(cage) -> None:
    slots = cage.slot_features()
    F = cage._fg_per_linker
    for case in (dict(groups="AB"), dict(linker_patterns={"AA": 2, "B": 2})):
        spec = rgroup.make_spec(len(slots) // F, F, case.get("groups", "A"),
                                case.get("linker_patterns"))
        configs = [rgroup.RGroupIsomer(f, spec.labels, ()) for f in _all_configurations(spec)]
        brute = isomer_ensemble_descriptor(slots, configs, F, include_same_linker=True)
        exact = ensemble_descriptor(slots, F, include_same_linker=True, **case)
        for key in set(exact.keys()) | set(brute.keys()):
            assert np.abs(exact.kde(GRID, [key]) - brute.kde(GRID, [key])).max() < 1e-9


def test_kde2d_conserves_weight_in_the_right_angle_range(cage) -> None:
    d = cage.get_descriptor()
    dg = np.linspace(-2, 32, 681)
    for key, top in ((("pi", "pi"), 90.0), (("fg", "fg"), 180.0)):
        ps = d.merged([key])
        assert ps.max_angle == top and np.nanmax(ps.angles) <= top + 1e-9
        ag = np.linspace(0, top, 361)
        z = d.kde2d(dg, ag, [key], bandwidth=0.3, angle_bandwidth=8.0)
        integral = np.trapezoid(np.trapezoid(z, ag, axis=1), dg)
        assert integral == pytest.approx(ps.weights[~np.isnan(ps.angles)].sum(), rel=1e-4)
    with pytest.raises(ValueError):
        d.kde2d(dg, np.linspace(0, 180, 10), [("pi", "pi"), ("fg", "fg")])


def test_assembly_check_uses_the_current_geometry() -> None:
    """The STK molecule keeps its construction geometry; after optimise()
    the guard must look at the optimised one."""
    from cage_isomer_builder.cage import Tri4Di6CageBuilder

    kwargs = dict(node="tests/data/uio66_tri_node.xyz", linker="tests/data/bdc.xyz")
    loose = Tri4Di6CageBuilder(**kwargs)              # bonds ~17 Angstrom
    loose.build()
    tight = Tri4Di6CageBuilder(scale_multiplier=0.225, **kwargs)   # bonds ~1.5
    tight.build()
    with pytest.warns(UserWarning, match="not bonded"):
        loose._check_assembly()
    loose._ase_cage = tight.to_ase().copy()           # same atoms, bonded geometry
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loose._check_assembly()


# ---------------------------------------------------------------------------
# Open metal sites: mofstructure's geometric detector
# ---------------------------------------------------------------------------

def _paddlewheel(axial_water=False):
    """Cu2(HCO2)4 paddlewheel along z, optionally with axial water."""
    symbols, positions = ["Cu", "Cu"], [[0, 0, 1.3], [0, 0, -1.3]]
    for k in range(4):
        t = k * np.pi / 2
        rot = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
        for s, p in (("C", [2.45, 0, 0]), ("O", [1.98, 0, 1.1]), ("O", [1.98, 0, -1.1]),
                     ("H", [3.53, 0, 0])):
            symbols.append(s)
            positions.append(rot @ np.array(p))
    if axial_water:
        for z in (1, -1):
            symbols += ["O", "H", "H"]
            positions += [[0, 0, z * 3.5], [0.76, 0, z * 4.1], [-0.76, 0, z * 4.1]]
    return Atoms(symbols, positions=positions)


def test_geometric_detector_finds_paddlewheel_sites() -> None:
    bare = detect_features(_paddlewheel(), kinds=("open_metal",))
    assert len(bare) == 2
    for p, v in zip(bare.positions, bare.vectors):
        # the open site is axial, pointing away from the other Cu
        assert v @ np.array([0, 0, np.sign(p[2])]) > 0.95
    assert len(detect_features(_paddlewheel(True), kinds=("open_metal",))) == 0
    # all Cu equally coordinated: the coordination rule alone cannot see it
    assert len(detect_features(_paddlewheel(), kinds=("open_metal",),
                               open_metal_method="coordination")) == 0
    assert len(detect_features(_paddlewheel(), kinds=("open_metal",),
                               open_metal_method="coordination", expected_cn={"Cu": 5})) == 2


def test_both_open_metal_methods_agree_on_a_zr6_defect() -> None:
    sbu = read("tests/data/uio66_sbu.xyz")
    sbu = sbu[[i for i in range(len(sbu)) if sbu[i].symbol != "X"]]
    adj = bond_graph(sbu)
    c = [i for i, s in enumerate(sbu.get_chemical_symbols()) if s == "C"][0]
    gone = [c] + [j for j in adj[c] if sbu[j].symbol == "O"]
    defect = sbu[[i for i in range(len(sbu)) if i not in gone]]
    geo = detect_features(defect, kinds=("open_metal",), open_metal_method="geometry")
    coord = detect_features(defect, kinds=("open_metal",), open_metal_method="coordination")
    assert sorted(geo.atom_indices) == sorted(coord.atom_indices) and len(geo) == 2
    assert np.all(np.einsum("ij,ij->i", geo.vectors, coord.vectors) > 0.9)
    for method in ("geometry", "coordination"):
        assert len(detect_features(sbu, kinds=("open_metal",), open_metal_method=method)) == 0


# ---------------------------------------------------------------------------
# Several linker types
# ---------------------------------------------------------------------------

def _mixed_slots():
    layout = rgroup.SlotLayout(sizes=(4, 4, 2, 2, 2), types=(0, 0, 1, 1, 1),
                               type_names=("bdc", "pyz"))
    rng = np.random.default_rng(5)
    fs = FeatureSet.from_rows([dict(position=rng.uniform(0, 12, 3), vector=rng.normal(size=3),
                                    kind="fg", owner=int(L), atom=k)
                               for k, L in enumerate(layout.slot_linker)])
    fs.vectors /= np.linalg.norm(fs.vectors, axis=1, keepdims=True)
    return layout, fs


def _all_typed_configs(spec):
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


@pytest.mark.parametrize("case", [
    dict(groups="A"),
    dict(groups={"bdc": "AB", "pyz": "A"}),
    dict(linker_patterns={"bdc": {"A": 1, "B": 1}, "pyz": {"A": 2, "": 1}}),
    dict(linker_patterns={"bdc": {"A": 1, vacancy_pattern(4): 1}, "pyz": {"B": 3}}),
])
@pytest.mark.parametrize("same_linker", [False, True])
def test_mixed_linker_ensemble_equals_brute_force(case, same_linker) -> None:
    layout, fs = _mixed_slots()
    spec = rgroup.make_spec(layout, None, **{"groups": "A", **case})
    configs = [rgroup.RGroupIsomer(f, spec.labels, ()) for f in _all_typed_configs(spec)]
    brute = isomer_ensemble_descriptor(fs, configs, layout, by="label",
                                       include_same_linker=same_linker)
    exact = ensemble_descriptor(fs, layout, by="label", include_same_linker=same_linker, **case)
    assert set(exact.keys()) == set(brute.keys())
    for key in exact.keys():
        assert np.abs(exact.kde(GRID, [key]) - brute.kde(GRID, [key])).max() < 1e-9


def test_mixed_linker_independent_mode() -> None:
    """Each linker draws its pattern independently with its type's
    fractions: compare with the explicitly weighted sum over every
    pattern choice and arrangement."""
    layout, fs = _mixed_slots()
    fractions = {"bdc": {"A": 0.3, "B": 0.7}, "pyz": {"A": 0.6, "": 0.4}}
    exact = ensemble_descriptor(fs, layout, mode="independent", fractions=fractions, by="label")
    labels = ("A", "B")
    acc = Descriptor()
    choices = [list(fractions[layout.type_names[t]].items()) for t in layout.types]
    for combo in itertools.product(*choices):
        weight = np.prod([f for _, f in combo])
        locals_ = []
        for L, (pattern, _) in enumerate(combo):
            F = layout.sizes[L]
            lab = [labels.index(c) + 1 for c in pattern] + [0] * (F - len(pattern))
            locals_.append(sorted(set(itertools.permutations(lab))))
        arrangements = list(itertools.product(*locals_))
        isos = [rgroup.RGroupIsomer(tuple(itertools.chain.from_iterable(a)), labels, ())
                for a in arrangements]
        acc = acc + isomer_ensemble_descriptor(fs, isos, layout, by="label").scaled(weight)
    for key in set(exact.keys()) | set(acc.keys()):
        assert np.abs(exact.kde(GRID, [key]) - acc.kde(GRID, [key])).max() < 1e-9
