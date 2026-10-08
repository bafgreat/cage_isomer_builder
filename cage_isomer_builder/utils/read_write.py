#!/usr/bin/env python3
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

from ase import Atoms
import numpy as np

def stk_2_ase_atoms(stk_atoms,
                    pbc=False
                    ):
    """
    Convert an STK molecule to ASE Atoms and its bond-order matrix.

    The bond matrix keeps the bonds of the construction, which matters where
    bonded atoms are further apart than a covalent-radius cutoff would allow
    (e.g. before optimisation). STK dummy atoms (At) become ``X``.

    Parameters
    ----------
    stk_atoms : stk.Molecule
    pbc : bool, default False
        Periodic boundary conditions of the returned Atoms.

    Returns
    -------
    ase_atoms : ase.Atoms
    bond_matrix : np.ndarray, shape (n_atoms, n_atoms)
        Bond orders (0 for no bond).

    Examples
    --------
    >>> import stk
    >>> from cage_isomer_builder.utils.read_write import stk_2_ase_atoms
    >>> atoms, bonds = stk_2_ase_atoms(stk.BuildingBlock("NCCN"))
    >>> atoms.get_chemical_formula(), int(bonds.sum() / 2)       # 3 heavy-atom + 8 X-H bonds
    ('C2H8N2', 11)
    """
    positions = stk_atoms.get_position_matrix()
    atoms = list(stk_atoms.get_atoms())
    n = len(atoms)
    numbers = [0 if a.get_atomic_number() == 85 else a.get_atomic_number() for a in atoms]

    ase_atoms = Atoms(numbers=numbers,
                      positions=positions,
                      pbc=pbc
                      )
    bond_matrix = np.zeros((n, n), dtype=float)
    for bond in stk_atoms.get_bonds():
        id1 = bond.get_atom1().get_id()
        id2 = bond.get_atom2().get_id()
        order = float(bond.get_order())
        bond_matrix[id1, id2] = bond_matrix[id2, id1] = order
    return ase_atoms, bond_matrix


def write_to_xyz(filename, ase_atoms):
    """
    Write ``ase_atoms`` to a plain XYZ file (symbols and positions only).

    Parameters
    ----------
    filename : str
    ase_atoms : ase.Atoms

    Examples
    --------
    >>> import os, tempfile
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.read_write import write_to_xyz
    >>> path = os.path.join(tempfile.mkdtemp(), "water.xyz")
    >>> write_to_xyz(path, molecule("H2O"))
    >>> open(path).read().splitlines()[0]
    '3'
    """

    sym = ase_atoms.symbols
    pos = ase_atoms.positions
    lines = ['%i\n'%len(ase_atoms), '\n']
    for iline in range(len(ase_atoms)):
        lines += ['{0:<3}{1:15.8f}{2:15.8f}{3:15.8f}\n'.format(sym[iline],
                                                  pos[iline][0],
                                                  pos[iline][1],
                                                  pos[iline][2])
                  ]
    file = open(filename, 'w')
    file.writelines(lines)
    file.close()


def build_isomer_atoms(cage,
                       fg_anchors,
                       isomer_descriptor,
                       fg_anchor_indices,
                       fg_anchor_h_positions=None,
                       ):
    """
    One single-group isomer as ``ase.Atoms``: the slots in
    ``isomer_descriptor`` are R1 sites (``X`` with ``"rgroup"`` = 1, see
    :mod:`cage_isomer_builder.utils.sites`) and every other slot is turned
    back into its original H.

    Parameters
    ----------
    cage : ase.Atoms
        The structure with every slot marked as a site
        (``CageBuilder.functionalise``). Not modified.
    fg_anchors : ase.Atoms
        The site atoms in slot order, giving each active site's position.
    isomer_descriptor : list of int
        Slot indices that are active, e.g. ``[0, 5, 9, 14]``.
    fg_anchor_indices : list of int
        Index in ``cage`` of each slot.
    fg_anchor_h_positions : np.ndarray, shape (n_slots, 3), optional
        Original H position of each slot. Without it an inactive slot gets
        an H at 1.09 Angstrom along its C-site direction.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.sites import mark_site
    >>> benzene = molecule("C6H6")
    >>> h_positions = benzene.positions[6:].copy()
    >>> for h in range(6, 12):
    ...     mark_site(benzene, h, 1)
    >>> slots = list(range(6, 12))
    >>> from cage_isomer_builder.utils.read_write import build_isomer_atoms
    >>> isomer = build_isomer_atoms(benzene, benzene[slots], [0, 3], slots, h_positions)
    >>> isomer.get_chemical_symbols()[6:]          # slots 0 and 3 active, the rest H again
    ['X', 'H', 'H', 'X', 'H', 'H']
    """
    from cage_isomer_builder.utils.sites import mark_site, restore_hydrogen, clear_site

    isomer = cage.copy()
    active = set(int(i) for i in isomer_descriptor)
    for slot, i_atom in enumerate(fg_anchor_indices):
        if slot in active:
            mark_site(isomer, i_atom, 1, fg_anchors[slot].position)
        elif fg_anchor_h_positions is not None:
            clear_site(isomer, i_atom, 'H', fg_anchor_h_positions[slot])
        else:
            restore_hydrogen(isomer, i_atom)
    isomer.info['rgroup_labels'] = 'R1'
    isomer.info['isomer'] = '-'.join(str(i) for i in isomer_descriptor)
    return isomer


def generate_isomer_structure_file(cage,
                                   fg_anchors,
                                   isomer_descriptor,
                                   fg_anchor_indices,
                                   output_path,
                                   fg_anchor_h_positions=None,
                                   ):
    """
    Write one single-group isomer (see :func:`build_isomer_atoms`) to
    ``<output_path>/<slot>-<slot>-....xyz`` as extended XYZ, with its sites
    listed in ``info["r_sites"]`` (see
    :func:`cage_isomer_builder.utils.sites.write_isomer`).

    Returns
    -------
    str
        Path of the written file.

    Examples
    --------
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.sites import mark_site
    >>> benzene = molecule("C6H6")
    >>> h_positions = benzene.positions[6:].copy()
    >>> for h in range(6, 12):
    ...     mark_site(benzene, h, 1)
    >>> slots = list(range(6, 12))
    >>> import tempfile
    >>> from cage_isomer_builder.utils.read_write import generate_isomer_structure_file
    >>> from cage_isomer_builder.utils.sites import read_isomer
    >>> path = generate_isomer_structure_file(benzene, benzene[slots], [0, 3], slots,
    ...                                       tempfile.mkdtemp(), h_positions)
    >>> path.endswith("0-3.xyz"), read_isomer(path).info["r_sites"]
    (True, {6: 'R1', 9: 'R1'})
    """
    from cage_isomer_builder.utils.sites import write_isomer

    isomer = build_isomer_atoms(cage, fg_anchors, isomer_descriptor,
                                fg_anchor_indices, fg_anchor_h_positions)
    name = '-'.join(str(i) for i in isomer_descriptor)
    return write_isomer("{}/{}.xyz".format(output_path, name), isomer)


def write_run(path: str, atoms: Atoms, bonds: np.ndarray) -> None:
    """
    Write an AMS geometry-optimisation run script (``Engine DFTB``) with
    explicit bond orders, for ``atoms`` and its bond-order matrix ``bonds``. The
    cell is written as ``Lattice`` for periodic structures.

    Parameters
    ----------
    path : str or Path
    atoms : ase.Atoms
    bonds : np.ndarray, shape (len(atoms), len(atoms))
        Bond-order matrix (only entries above zero are written).

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.read_write import write_run
    >>> path = os.path.join(tempfile.mkdtemp(), "water.run")
    >>> write_run(path, molecule("H2O"), np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]))
    >>> "BondOrders" in open(path).read()
    True
    """
    path = str(path)
    with open(path, "w") as fileobj:
        fileobj.write("#!/bin/sh\n\n")
        fileobj.write('"$AMSBIN/ams" << eor\n\n')
        fileobj.write("Task GeometryOptimization\n")
        fileobj.write("GeometryOptimization\n")
        fileobj.write("   OptimizeLattice Yes\n")
        fileobj.write("End\n")
        fileobj.write("System\n")
        fileobj.write("Atoms\n")

        for (s, (x, y, z)) in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            fileobj.write(("{0:<4} {1:<15.8f} {2:<15.8f} {3:<15.8f}\n").format(s, x, y, z))
        fileobj.write("\n")
        fileobj.write("End\n")

        pbc = atoms.get_pbc()
        if np.any(pbc):
            fileobj.write("Lattice\n")
            cell = atoms.get_cell().tolist()

            # Write only lattice vectors needed. If your z is non-periodic, omit c vector.
            if not pbc[2]:
                fileobj.write("{0:.6f} {1:.6f} {2:.6f}\n".format(*cell[0]))
                fileobj.write("{0:.6f} {1:.6f} {2:.6f}\n".format(*cell[1]))
            else:
                fileobj.write("{0:.6f} {1:.6f} {2:.6f}\n".format(*cell[0]))
                fileobj.write("{0:.6f} {1:.6f} {2:.6f}\n".format(*cell[1]))
                fileobj.write("{0:.6f} {1:.6f} {2:.6f}\n".format(*cell[2]))
            fileobj.write("End\n")

        fileobj.write("BondOrders\n")
        # bonds is NxN matrix
        for (i0, i1), b in np.ndenumerate(bonds):
            if i0 < i1 and b > 0.0:
                fileobj.write(("{0:<4} {1:<4} {2:.3f}\n").format(i0 + 1, i1 + 1, float(b)))
        fileobj.write("End\n")

        fileobj.write("End\n")
        fileobj.write("Engine DFTB\n")
        fileobj.write("EndEngine\n")
        fileobj.write("eor\n")



def write_host_guest_complex(complex_, path, lattice="conv", mechanical=False):
    """
    Write a HostGuestComplex to disk, picking the writer from ``path``'s
    file extension.

    - ``.gin`` -> GULP molecular-mechanics input, via
      ``gulp_setup.mmanalysis.write_gin`` (UFF4MOF atom types assigned by
      ``analyze_mm`` from ``complex_.bond_matrix``). Bond orders are taken
      from ``complex_.bond_matrix``, never re-perceived from the merged
      host+guest geometry (see place_guest_in_host).
    - ``.run`` -> AMS geometry-optimisation run script, via
      utils.read_write.write_run, same bond matrix.
    - anything else -> plain ``ase.io.write(path, atoms)``. No bond
      matrix is used (or needed) here, since none of ASE's own formats
      (xyz, cif, pdb, ...) carry bond order.

    Parameters
    ----------
    complex_ : HostGuestComplex or ase.Atoms
        A HostGuestComplex is required for ``.gin``/``.run`` output,
        since those need its bond_matrix. Plain ase.Atoms works for any
        other (bond-order-free) format.
    path : str or Path
        Output file path; its suffix selects the writer.
    lattice : str, default "conv"
        Forwarded to ``write_gin`` for ``.gin`` output. ``"conv"`` for
        constant volume (no cell optimisation), ``"conp"`` for constant
        pressure (cell optimisation); only relevant for periodic
        structures.
    mechanical : bool, default False
        Forwarded to ``write_gin`` for ``.gin`` output: use its
        "mechanical properties" input instead of the standard
        geometry-optimisation one.

    Raises
    ------
    TypeError
        If ``path`` ends in ``.gin`` or ``.run`` but ``complex_`` is not a
        HostGuestComplex, which has the bond matrix to write.

    Examples
    --------
    >>> import os, tempfile
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.read_write import write_host_guest_complex
    >>> from cage_isomer_builder.utils.functionalise import HostGuestComplex
    >>> import numpy as np
    >>> water = molecule("H2O")
    >>> complex_ = HostGuestComplex(water, np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]), [], [])
    >>> path = os.path.join(tempfile.mkdtemp(), "complex.run")
    >>> write_host_guest_complex(complex_, path)          # .gin, .run or any ASE format
    >>> os.path.exists(path)
    True
    """
    from pathlib import Path
    suffix = Path(path).suffix.lower()

    if suffix in {".gin", ".run"}:
        if not hasattr(complex_, "bond_matrix"):
            raise TypeError(
                f"{suffix} output needs bond orders; pass a HostGuestComplex "
                "(e.g. from place_guest_in_host), not a plain ase.Atoms."
            )
        atoms, bond_matrix = complex_.atoms, complex_.bond_matrix
        if suffix == ".run":
            write_run(str(path), atoms, bond_matrix)
        else:
            from gulp_setup.mmanalysis import analyze_mm, write_gin
            _, mmtypes = analyze_mm(atoms, bond_order=bond_matrix)
            write_gin(
                str(path), atoms, bond_matrix, mmtypes,
                lattice=lattice, mechanical=mechanical,
            )
        return

    from ase.io import write
    atoms = complex_.atoms
    write(str(path), atoms)