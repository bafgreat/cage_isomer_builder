#!/usr/bin/env python3
"""
Small example structures shipped with the package, so every example in the
documentation runs the same from a clone or after ``pip install``.

=====================  ======================================================
name                   contents
=====================  ======================================================
``uio66_tri_node``     UiO-66 Zr6 node with three ``X`` connection points
                       (tritopic, for Tri4Di6 cages)
``uio66_tetra_node``   UiO-66 Zr6 node with four ``X`` connection points
``uio66_sbu``          UiO-66 Zr6O8 cluster with 12 carboxylates
``bdc``                1,4-phenylene linker with two ``X`` connection points
``biphenyl``           4,4'-biphenylene linker with two ``X`` points
``NH2``                NH2 fragment with one ``X`` attachment point
``methane``            methane (guest)
``ibuprofen``          ibuprofen (guest)
``RUBTAK01``           UiO-66 crystal structure (CIF, Zr, one linker type)
``ABIXEQ``             Co MOF with two linker types (CIF)
``ABINOQ``             Zn MOF: bipyridine linker plus terminal thiolates (CIF)
=====================  ======================================================

Examples
--------
>>> from cage_isomer_builder import example_data
>>> "bdc" in example_data.names()
True
>>> example_data.read("methane").get_chemical_formula()
'CH4'
"""
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

from pathlib import Path

_DIR = Path(__file__).resolve().parent / "__data__" / "structures"


def names():
    """
    Names of the shipped example structures, sorted.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> example_data.names()[:3]
    ['ABINOQ', 'ABIXEQ', 'NH2']
    """
    return sorted(p.stem for p in _DIR.iterdir() if p.suffix in (".xyz", ".cif"))


def path(name):
    """
    Path of an example structure, given its name (with or without the file
    extension). Use it wherever a file path is accepted, e.g.
    ``Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"), ...)``.

    Raises
    ------
    KeyError
        If there is no example of that name.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> example_data.path("RUBTAK01").name
    'RUBTAK01.cif'
    """
    stem = Path(name).stem
    for suffix in (".xyz", ".cif"):
        candidate = _DIR / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise KeyError(f"no example structure {name!r}; available: {names()}")


def read(name):
    """
    An example structure as ``ase.Atoms``.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> atoms = example_data.read("RUBTAK01")
    >>> sorted(set(atoms.get_chemical_symbols()))
    ['C', 'H', 'O', 'Zr']
    """
    from ase.io import read as ase_read

    return ase_read(str(path(name)))
