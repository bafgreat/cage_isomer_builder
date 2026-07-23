#!/usr/bin/env python3
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

from ase import Atoms
import numpy as np

def stk_2_ase_atoms(stk_atoms,
                    pbc=False
                    ):
    """
    A function that converts and stk molecular object to an
    ase atom object.

    Parameter:
    ----------
        - stk_atoms: stk molecular object
        - pbc : periodic boundary conditions
            default is False.

    Return:
    --------
        - ase_atom:  ase.Atoms
            ASE atom object
        - bond_matrix : bond matrix
            The bond matrix is essential for force field and maintaining
            correct bonding especially when two bonded atoms are far apart
            beyond conventional bond cutoff imposed by covalent radii

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
    A function that writes an ase_atoms to xyz

    Parameter:
        - filename: str
            name of file to save
        - ase_atoms : ASE.atoms
            structure to be written
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


def generate_isomer_structure_file(cage,
                                   fg_anchors,
                                   isomer_descriptor,
                                   fg_anchor_indices,
                                   output_path,
                                   fg_anchor_h_positions=None,
                                   ):
    """
    A function to generate and write an XYZ structure file for a single isomer, placing
    "At" anchor atoms at the active FG slots defined by the isomer descriptor
    and restoring every other slot to its original, unfunctionalised H atom.


    Parameters
    ----------
    cage : ase.Atoms
        The base cage structure containing all FG anchor slots as At atoms.
        A copy is made internally so the original is not modified.
    fg_anchors : ase.Atoms
        Subset of the cage containing all At anchor atoms with their
        positions, as returned by functionalise_all_anchors.
        Used to restore active anchor positions in the isomer.
    isomer_descriptor : list of int
        Base-4 isomer descriptor of length n_linkers, where each element
        is the global FG anchor slot index active on that linker.
        e.g. [0, 5, 9, 14] activates slots 0, 5, 9, and 14.
    fg_anchor_indices : list of int
        Indices of all At anchor atoms in the full cage Atoms object,
        as returned by functionalise_all_anchors.
        Used to map from isomer_descriptor indices to cage atom indices.
    output_path : str
        Directory path where the XYZ file will be saved.
        e.g. 'Tet2Di4_isomers/
    fg_anchor_h_positions : np.ndarray, shape (len(fg_anchor_indices), 3), optional
        Each anchor's original (pre-extension) hydrogen position, aligned
        with ``fg_anchor_indices``, as returned by ``full_functionalisation``.
        Slots not selected in ``isomer_descriptor`` are restored to a real H
        atom at this position rather than left as an "X" placeholder. If
        omitted, inactive slots fall back to "X" (legacy behaviour).
    """

    # Copy cage and mask all anchor slots as inactive - a real H atom back
    # at its original bond length when we know it, else an X placeholder.
    isomer = cage.copy()
    for slot, i_atom in enumerate(fg_anchor_indices):
        if isomer[i_atom].symbol != 'At':
            continue
        if fg_anchor_h_positions is not None:
            isomer[i_atom].symbol = 'H'
            isomer[i_atom].position = fg_anchor_h_positions[slot]
        else:
            isomer[i_atom].symbol = 'X'

    filename = ""
    for i_slot in isomer_descriptor:
        isomer[fg_anchor_indices[i_slot]].symbol = 'At'
        isomer[fg_anchor_indices[i_slot]].position =\
            fg_anchors[i_slot].position
        filename += str(i_slot) + '-'

    # Write to XYZ file named after the isomer descriptor
    write_to_xyz("{}/{}.xyz".format(output_path, filename[:-1]), isomer)


def write_run(path: str, atoms: Atoms, bonds: np.ndarray) -> None:
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
      from ``complex_.bond_matrix`` exactly, never re-perceived from the
      merged host+guest geometry - see place_guest_in_host.
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
        pressure (cell optimisation) - only relevant for periodic
        structures.
    mechanical : bool, default False
        Forwarded to ``write_gin`` for ``.gin`` output - use its
        "mechanical properties" input variant instead of the standard
        geometry-optimisation one.

    Raises
    ------
    TypeError
        If ``path`` ends in ``.gin`` or ``.run`` but ``complex_`` isn't a
        HostGuestComplex - there is no bond matrix to write.
    """
    from pathlib import Path
    suffix = Path(path).suffix.lower()

    if suffix in {".gin", ".run"}:
        if not hasattr(complex_, "bond_matrix"):
            raise TypeError(
                f"{suffix} output needs bond orders - pass a HostGuestComplex "
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