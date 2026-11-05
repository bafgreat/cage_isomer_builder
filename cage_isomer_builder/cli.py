#!/usr/bin/env python3
"""
Command-line tools.

``cage-vertex-name`` converts between the site name of an isomer
(``0-4-9-13-16-22``) and its vertex name (``000-100-141-401``, see
:mod:`cage_isomer_builder.utils.vertex_codes`), for names or for isomer
files written by
:meth:`~cage_isomer_builder.cage.base.CageBuilder.enumerate_isomers`::

    cage-vertex-name 0-4-9-13-16-22.xyz            # writes 000-100-141-401.xyz
    cage-vertex-name 000-100-141-401.xyz           # writes 0-4-9-13-16-22.xyz
    cage-vertex-name isomers/*.xyz -o by_vertex    # a whole folder
    cage-vertex-name --ref isomers/0-4-8-12-16-20.xyz 0-4-9-13-16-22 000-100-141-401

With ``--ref``, any isomer file of the cage supplies the vertex scheme and
the names are converted and printed; nothing is written.
"""

import argparse
import os
import shutil
import sys
from typing import Optional, Sequence

from cage_isomer_builder.utils.vertex_codes import VertexScheme


def _isomer_names(path: str):
    """``(site name, vertex name, labels or None)`` of an isomer file."""
    from cage_isomer_builder.utils.sites import read_isomer

    info = read_isomer(path).info
    try:
        scheme = VertexScheme.from_info(info)
    except ValueError:
        scheme = None
    stem = os.path.splitext(os.path.basename(path))[0]
    if "isomer" in info:
        sites = [int(s) for s in str(info["isomer"]).split("-")]
    elif scheme is not None:
        sites = scheme.resolve(stem)
    elif not info or set(info) <= {"r_sites", "pbc"}:
        raise ValueError(f"{path}: the file has no isomer information in its header (written "
                         "by an older version?); write it again with enumerate_isomers().")
    else:
        raise ValueError(f"{path}: no site name (info['isomer']) in the file.")
    if scheme is not None:
        vertex = scheme.name(sites)
        labels = "-".join(scheme.labels(sites))
    elif "vertex_name" in info:          # files written before the scheme was stored
        vertex, labels = str(info["vertex_name"]), info.get("vertex_labels")
    else:
        raise ValueError(f"{path}: no vertex codes in the file (a MOF, or a cage whose "
                         "linkers do not join two vertices?).")
    return "-".join(map(str, sites)), vertex, labels


def _convert_file(path: str, to: str, outdir: Optional[str], move: bool, force: bool) -> str:
    site, vertex, _ = _isomer_names(path)
    stem, ext = os.path.splitext(os.path.basename(path))
    if to == "auto":
        to = "sites" if stem == vertex else "vertex"
    target = os.path.join(outdir or os.path.dirname(path),
                          (vertex if to == "vertex" else site) + ext)
    if os.path.abspath(target) == os.path.abspath(path):
        return f"{path} is already named by its {to} name"
    if os.path.exists(target) and not force:
        raise ValueError(f"{target} exists (use --force to overwrite).")
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    (shutil.move if move else shutil.copy2)(path, target)
    return f"{path} -> {target}"


def _convert_name(scheme: VertexScheme, name: str) -> str:
    sites = scheme.resolve(name)
    vertex, site = scheme.name(sites), "-".join(map(str, sites))
    out = site if scheme.is_vertex_name(name) else vertex
    return f"{name} -> {out}   labels {'-'.join(scheme.labels(sites))}"


def vertex_name_main(argv: Optional[Sequence[str]] = None) -> int:
    """
    The ``cage-vertex-name`` command: convert isomer files, or with
    ``--ref`` isomer names, between site names and vertex names. Returns
    the exit status: 0, or 1 if any file or name failed.

    Examples
    --------
    An isomer file of a toy cage (two vertices joined by two linkers, sites
    0-3 and 4-7) with the vertex scheme in its ``info``:

    >>> import os, tempfile
    >>> from ase import Atoms
    >>> from cage_isomer_builder.cli import vertex_name_main
    >>> from cage_isomer_builder.utils.sites import write_isomer
    >>> from cage_isomer_builder.utils.vertex_codes import VertexScheme
    >>> scheme = VertexScheme(((0, 1), (0, 1)), ((0, 1, 2, 3), (4, 5, 6, 7)),
    ...                       {0: (0, 1), 1: (0, 4), 2: (1, 1), 3: (1, 4),
    ...                        4: (0, 1), 5: (0, 4), 6: (1, 1), 7: (1, 4)})
    >>> atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    >>> atoms.info.update(isomer="1-6", **scheme.to_info())
    >>> os.chdir(tempfile.mkdtemp())
    >>> _ = write_isomer("1-6.xyz", atoms)

    A file named by its sites gets a copy named by its vertex name, and back:

    >>> vertex_name_main(["1-6.xyz"])
    1-6.xyz -> 40-01.xyz
    0
    >>> vertex_name_main(["40-01.xyz", "-o", "back"])
    40-01.xyz -> back/1-6.xyz
    0

    With ``--ref``, names are converted either way:

    >>> vertex_name_main(["--ref", "1-6.xyz", "0-4", "40-01"])
    0-4 -> 11-00   labels 11-00
    40-01 -> 1-6   labels 40-10
    0
    """
    parser = argparse.ArgumentParser(
        prog="cage-vertex-name",
        description="Convert isomers between site names (0-4-9-13-16-22) and vertex "
                    "names (000-100-141-401). Give isomer files to write copies under "
                    "the other name, or --ref FILE and names to print the conversion.")
    parser.add_argument("items", nargs="+", metavar="FILE_OR_NAME",
                        help="isomer files, or names when --ref is given")
    parser.add_argument("--ref", metavar="FILE",
                        help="an isomer file of the cage: convert the names given instead of files")
    parser.add_argument("--to", choices=("auto", "vertex", "sites"), default="auto",
                        help="name to give files (default: the other one)")
    parser.add_argument("-o", "--outdir", help="write the files here (default: next to each file)")
    parser.add_argument("--move", action="store_true", help="rename instead of copying")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args(argv)

    status = 0
    if args.ref:
        try:
            from cage_isomer_builder.utils.sites import read_isomer
            scheme = VertexScheme.from_info(read_isomer(args.ref).info)
        except (ValueError, OSError) as exc:
            print(f"error: {args.ref}: {exc}", file=sys.stderr)
            return 1
        jobs = [(name, lambda n: _convert_name(scheme, n)) for name in args.items]
    else:
        jobs = [(path, lambda p: _convert_file(p, args.to, args.outdir, args.move, args.force))
                for path in args.items]
    for item, job in jobs:
        try:
            print(job(item))
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(vertex_name_main())
