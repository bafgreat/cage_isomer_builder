#!/usr/bin/env python3
"""
How an active (functionalisation) site is represented on an ``ase.Atoms``.

A site is an ASE dummy atom, symbol ``X`` (atomic number 0), whose R-group
number is stored in the per-atom integer array ``"rgroup"``: 1 means R1, 2
means R2, and so on; every other atom has 0. ASE has no element "R", so the
R notation lives in that array rather than in the chemical symbol.

Written isomer files (extended XYZ, see :func:`write_isomer`) also carry a
human-readable map in ``atoms.info["r_sites"]``, e.g. ``"12:R1 40:R2"``
(atom 12 is an R1 site, atom 40 an R2 site), and the group names in
``atoms.info["rgroup_labels"]`` (e.g. ``"NH2 OH"`` means R1 = NH2,
R2 = OH). :func:`read_isomer` reads such a file back and rebuilds the
``"rgroup"`` array from the map if a plain reader lost it.

Why the array is the source of truth, not the map: deleting or reordering
atoms (attaching fragments, removing a linker for a defect) shifts atom
indices, which invalidates an index map, while a per-atom array is
sliced together with the atoms. The map is regenerated from the array every
time a file is written.

In RDKit exports (SDF/MOL, CXSMILES) every site becomes a real R-group atom
(R1, R2, ...); see :func:`cage_isomer_builder.utils.rgroup.rgroup_atoms_to_rdkit`.
"""
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

import numpy as np
from ase.io import read, write

#: Chemical symbol of a site atom (ASE's dummy atom).
SITE_SYMBOL = "X"
#: Per-atom integer array holding each site's R-group number (0 = no site).
SITE_ARRAY = "rgroup"
#: ``atoms.info`` key of the written ``"index:R_i"`` map.
SITE_INFO = "r_sites"
#: ``atoms.info`` key of the R-group names, space separated, R1 first.
LABEL_INFO = "rgroup_labels"

#: Ideal aromatic C-H bond length, used when an inactive site's original H
#: position is unknown.
CH_BOND = 1.09


def site_array(atoms):
    """The ``"rgroup"`` array of ``atoms`` (zeros if it has none).

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.sites import mark_site, site_array
    >>> benzene = molecule("C6H6")
    >>> site_array(benzene).sum()              # no sites yet
    np.int64(0)
    >>> mark_site(benzene, 6, 2)
    >>> site_array(benzene)[6]
    np.int64(2)
    """
    if SITE_ARRAY in atoms.arrays:
        return np.asarray(atoms.arrays[SITE_ARRAY], dtype=int)
    return np.zeros(len(atoms), dtype=int)


def site_indices(atoms):
    """
    Indices of the site atoms, in ascending order.

    A site is an atom with a non-zero ``"rgroup"`` entry. Structures without
    that array (e.g. a hand-made file) fall back to every ``X`` atom.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.sites import mark_site, site_indices
    >>> benzene = molecule("C6H6")
    >>> mark_site(benzene, 9, 1)
    >>> mark_site(benzene, 6, 1)
    >>> site_indices(benzene)
    [6, 9]
    """
    if SITE_ARRAY in atoms.arrays:
        return [int(i) for i in np.nonzero(site_array(atoms))[0]]
    return [atom.index for atom in atoms if atom.symbol == SITE_SYMBOL]


def mark_site(atoms, index, label=1, position=None):
    """
    Turn atom ``index`` into an R``label`` site in place (symbol ``X``,
    ``"rgroup"`` entry ``label``), optionally moving it to ``position``.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.sites import mark_site
    >>> benzene = molecule("C6H6")
    >>> mark_site(benzene, 6, 1)               # H 6 becomes an R1 site
    >>> benzene[6].symbol, int(benzene.arrays["rgroup"][6])
    ('X', 1)
    """
    if label < 1:
        raise ValueError(f"R-group numbers start at 1, got {label}.")
    labels = site_array(atoms).copy()
    labels[index] = label
    atoms.set_array(SITE_ARRAY, labels)
    atoms[index].symbol = SITE_SYMBOL
    if position is not None:
        atoms[index].position = position


def clear_site(atoms, index, symbol="H", position=None):
    """Turn site atom ``index`` back into an ordinary atom (H by default).

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.sites import clear_site, mark_site, site_indices
    >>> benzene = molecule("C6H6")
    >>> mark_site(benzene, 6, 1)
    >>> clear_site(benzene, 6)                 # back to H
    >>> benzene[6].symbol, site_indices(benzene)
    ('H', [])
    """
    labels = site_array(atoms).copy()
    labels[index] = 0
    atoms.set_array(SITE_ARRAY, labels)
    atoms[index].symbol = symbol
    if position is not None:
        atoms[index].position = position


def site_map(atoms):
    """``{atom_index: "R1", ...}`` for every site atom.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.sites import mark_site, site_map
    >>> benzene = molecule("C6H6")
    >>> mark_site(benzene, 6, 1)
    >>> mark_site(benzene, 9, 2)
    >>> site_map(benzene)
    {6: 'R1', 9: 'R2'}
    """
    labels = site_array(atoms)
    return {i: f"R{int(labels[i])}" for i in site_indices(atoms) if labels[i] > 0} \
        if SITE_ARRAY in atoms.arrays else {i: "R1" for i in site_indices(atoms)}


def format_site_map(mapping):
    """``{12: "R1", 40: "R2"}`` -> ``"12:R1 40:R2"``.

    Examples
    --------
    >>> from cage_isomer_builder.utils.sites import format_site_map
    >>> format_site_map({40: "R2", 12: "R1"})
    '12:R1 40:R2'
    """
    return " ".join(f"{i}:{r}" for i, r in sorted(mapping.items()))


def parse_site_map(text):
    """``"12:R1 40:R2"`` -> ``{12: "R1", 40: "R2"}``.

    Examples
    --------
    >>> from cage_isomer_builder.utils.sites import parse_site_map
    >>> parse_site_map("12:R1 40:R2")
    {12: 'R1', 40: 'R2'}
    """
    out = {}
    for item in str(text).split():
        index, label = item.split(":")
        if not label.startswith("R"):
            raise ValueError(f"site label {label!r} is not of the form R<n>.")
        out[int(index)] = label
    return out


def restore_hydrogen(atoms, index, anchor_neighbour=None, bond_length=CH_BOND):
    """
    Put an H back on an inactive site whose original H position is unknown:
    ``bond_length`` from its bonded heavy atom, along the same direction.
    ``anchor_neighbour`` is that heavy atom; if omitted, the nearest atom
    that is neither H nor a site is used.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.sites import mark_site, restore_hydrogen
    >>> benzene = molecule("C6H6")
    >>> mark_site(benzene, 6, 1)                     # H 6 is bonded to C 0
    >>> benzene.set_distance(0, 6, 1.47, fix=0)      # a site sits 1.47 A out
    >>> restore_hydrogen(benzene, 6)
    >>> benzene[6].symbol, round(float(benzene.get_distance(0, 6)), 2)
    ('H', 1.09)
    """
    position = atoms.positions[index]
    if anchor_neighbour is None:
        sites = set(site_indices(atoms))
        candidates = [a.index for a in atoms
                      if a.index != index and a.symbol not in ("H", SITE_SYMBOL)
                      and a.index not in sites]
        d = np.linalg.norm(atoms.positions[candidates] - position, axis=1)
        anchor_neighbour = candidates[int(np.argmin(d))]
    base = atoms.positions[anchor_neighbour]
    direction = position - base
    direction /= np.linalg.norm(direction)
    clear_site(atoms, index, "H", base + bond_length * direction)


def write_isomer(path, atoms):
    """
    Write ``atoms`` as extended XYZ with its R sites.

    ``atoms.info["r_sites"]`` is regenerated from the ``"rgroup"`` array (so
    it always matches the atom order being written), and the array itself is
    written as a per-atom column. The file reads back with plain
    ``ase.io.read`` as well as with :func:`read_isomer`.

    Examples
    --------
    >>> import os, tempfile
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.sites import mark_site, read_isomer, write_isomer
    >>> benzene = molecule("C6H6")
    >>> mark_site(benzene, 6, 1)
    >>> path = write_isomer(os.path.join(tempfile.mkdtemp(), "iso.xyz"), benzene)
    >>> from ase.io import read                      # a plain reader works too
    >>> read(path).info["r_sites"]
    '6:R1'
    >>> path = write_isomer(os.path.join(tempfile.mkdtemp(), "bare.xyz"), molecule("C6H6"))
    >>> read_isomer(path).info["r_sites"]            # no sites: an empty map
    {}
    """
    out = atoms.copy()
    # An empty value would make the extxyz reader swallow the next key.
    mapping = format_site_map(site_map(atoms))
    if mapping:
        out.info[SITE_INFO] = mapping
    else:
        out.info.pop(SITE_INFO, None)
    if SITE_ARRAY not in out.arrays:
        out.set_array(SITE_ARRAY, np.array(
            [1 if s == SITE_SYMBOL else 0 for s in out.get_chemical_symbols()], dtype=int))
    write(str(path), out, format="extxyz")
    return str(path)


def read_isomer(path, index=-1):
    """
    Read an isomer written by :func:`write_isomer` (or any extended XYZ).

    The ``"rgroup"`` array is rebuilt from ``info["r_sites"]`` when the file
    has the map but not the column. ``info["r_sites"]`` is returned as a
    dict, ``{atom_index: "R1", ...}``.

    Examples
    --------
    >>> import os, tempfile
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.sites import mark_site, read_isomer, write_isomer
    >>> benzene = molecule("C6H6")
    >>> mark_site(benzene, 6, 1)
    >>> mark_site(benzene, 8, 2)
    >>> path = write_isomer(os.path.join(tempfile.mkdtemp(), "iso.xyz"), benzene)
    >>> read_isomer(path).info["r_sites"]
    {6: 'R1', 8: 'R2'}
    """
    atoms = read(str(path), index=index)
    mapping = atoms.info.get(SITE_INFO)
    if isinstance(mapping, str) or mapping is None:
        mapping = parse_site_map(mapping) if mapping else {}
    if SITE_ARRAY not in atoms.arrays:
        labels = np.zeros(len(atoms), dtype=int)
        for i, r in mapping.items():
            labels[i] = int(r[1:])
        if not mapping:
            labels[[a.index for a in atoms if a.symbol == SITE_SYMBOL]] = 1
        atoms.set_array(SITE_ARRAY, labels)
    else:
        atoms.set_array(SITE_ARRAY, np.asarray(atoms.arrays[SITE_ARRAY], dtype=int))
    atoms.info[SITE_INFO] = site_map(atoms)
    return atoms
