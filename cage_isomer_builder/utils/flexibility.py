#!/usr/bin/env python3
"""
Framework flexibility for descriptors: linker ring rotation and isotropic
pore expansion/contraction.

Ring rotation
-------------
A *rotor* is an aromatic ring of a linker that can turn about the axis
through its two backbone atoms: the ring atoms bonded (outside the ring) to
the rest of the framework, e.g. C1 and C4 of a 1,4-phenylene. Everything
attached to the ring except through those two backbone bonds (its H atoms,
R sites, substituents) turns with it; the backbone itself does not move, so
every bond length and every bond to the node is preserved exactly.

A backbone atom is a ring atom whose exocyclic heavy neighbour leads to a
metal, to another ring, or out of the linker; an exocyclic neighbour that
only leads to a terminal substituent (NH2, CH3, ...) is not backbone. Rings
with other than two backbone atoms (e.g. a tritopic core) are not rotors.

:func:`rotation_samples` gives every slot's position at each angle, ready for
:func:`cage_isomer_builder.utils.ensemble.ensemble_descriptor`: slots on one
ring move together; different rings move independently. Angle weights may
come from an energy profile (e.g. a UFF4MOF torsion scan, via
:func:`cage_isomer_builder.utils.ensemble.boltzmann_weights`); the default is
uniform.

Pore scaling
------------
:func:`scale_structure` moves every building block rigidly so its centroid's
distance from the pore centre is multiplied by a factor (a periodic cell is
scaled instead, keeping each block's internal geometry). Internal geometry
is untouched, so the result shows how each FG-FG distance responds to an
isotropic expansion or contraction; :func:`distance_scaling` gives the slope
d(distance)/d(factor) of every pair.
"""
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

from dataclasses import dataclass

import numpy as np
from ase.geometry import find_mic

from cage_isomer_builder.utils.features import aromatic_rings, bond_graph, is_metal, orientation_class


@dataclass(frozen=True)
class Rotor:
    """
    Attributes
    ----------
    ring : tuple of int
    axis_atoms : (int, int)
        The two backbone ring atoms; the rotation axis passes through them.
    moving : tuple of int
        Every atom that turns (ring atoms other than the axis atoms, and
        everything hanging off the ring).
    linker : int
        Index of the linker (in the list given to :func:`find_rotors`), or -1.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
    >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
    ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
    >>> _ = cage.build()
    >>> _ = cage.optimise(include_charges=False, logfile=None)
    >>> _ = cage.functionalise()
    >>> atoms, adjacency = cage.to_ase(), cage.adjacency()
    >>> linkers, nodes = cage.building_blocks()
    >>> from cage_isomer_builder.utils.flexibility import find_rotors
    >>> rotor = find_rotors(atoms, linkers, adjacency)[0]
    >>> len(rotor.ring), len(rotor.axis_atoms), len(rotor.moving)   # ring C, axis C, moving atoms
    (6, 2, 8)
    """
    ring: tuple
    axis_atoms: tuple
    moving: tuple
    linker: int = -1


def _side(adjacency, start, blocked):
    """Atoms reachable from start without entering ``blocked``."""
    seen, stack = {start}, [start]
    while stack:
        a = stack.pop()
        for b in adjacency[a]:
            if b not in seen and b not in blocked:
                seen.add(b)
                stack.append(b)
    return seen


def find_rotors(atoms, linker_atom_indices=None, adjacency=None):
    """
    Rotatable aromatic rings (see the module docstring).

    Parameters
    ----------
    atoms : ase.Atoms
    linker_atom_indices : list of list of int, optional
        Atoms of each linker; only rings inside a linker are returned, and a
        neighbour outside it counts as backbone. Default: every ring.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
    >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
    ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
    >>> _ = cage.build()
    >>> _ = cage.optimise(include_charges=False, logfile=None)
    >>> _ = cage.functionalise()
    >>> atoms, adjacency = cage.to_ase(), cage.adjacency()
    >>> linkers, nodes = cage.building_blocks()
    >>> from cage_isomer_builder.utils.flexibility import find_rotors
    >>> len(find_rotors(atoms, linkers, adjacency))       # one phenylene ring per linker
    6
    """
    if adjacency is None:
        adjacency = bond_graph(atoms)
    numbers = atoms.numbers
    rings = aromatic_rings(atoms, adjacency)
    ring_atoms = {a for r in rings for a in r}
    owner = {}
    if linker_atom_indices is not None:
        owner = {a: k for k, lk in enumerate(linker_atom_indices) for a in lk}
    rotors = []
    for ring in rings:
        lk = owner.get(ring[0], -1)
        if linker_atom_indices is not None and lk < 0:
            continue
        members = set(ring)
        backbone = []
        for a in ring:
            for b in adjacency[a]:
                if b in members or numbers[b] <= 1:
                    continue
                side = _side(adjacency, b, members)
                leads_out = (any(is_metal(numbers[k]) for k in side)
                             or any(k in ring_atoms for k in side)
                             or (linker_atom_indices is not None
                                 and any(owner.get(k, -1) != lk for k in side)))
                if leads_out:
                    backbone.append((a, b))
        if len({a for a, _ in backbone}) != 2:
            continue
        (a1, b1), (a2, b2) = sorted({a: (a, b) for a, b in backbone}.values())
        # The two axis atoms split the ring into two arcs: collect both,
        # with everything hanging off them.
        moving = set()
        for start in ring:
            if start not in (a1, a2) and start not in moving:
                moving |= _side(adjacency, start, {a1, a2, b1, b2})
        # A ring whose "substituents" reach a metal or another building block
        # (e.g. an ortho group that coordinates the node) cannot turn without
        # dragging the framework along: not a rotor.
        if any(is_metal(numbers[k]) for k in moving) or (
                linker_atom_indices is not None
                and any(owner.get(k, -1) != lk for k in moving)):
            continue
        rotors.append(Rotor(tuple(ring), (a1, a2), tuple(sorted(moving)), lk))
    return rotors


def _rotate(points, origin, axis, angle):
    v = points - origin
    c, s = np.cos(angle), np.sin(angle)
    return origin + v * c + np.cross(axis, v) * s + np.outer(v @ axis, axis) * (1 - c)


def rotor_axis(atoms, rotor):
    """Origin (first axis atom) and unit axis of a rotor.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
    >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
    ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
    >>> _ = cage.build()
    >>> _ = cage.optimise(include_charges=False, logfile=None)
    >>> _ = cage.functionalise()
    >>> atoms, adjacency = cage.to_ase(), cage.adjacency()
    >>> linkers, nodes = cage.building_blocks()
    >>> import numpy as np
    >>> from cage_isomer_builder.utils.flexibility import find_rotors, rotor_axis
    >>> origin, axis = rotor_axis(atoms, find_rotors(atoms, linkers, adjacency)[0])
    >>> round(float(np.linalg.norm(axis)), 6)
    1.0
    """
    p0 = atoms.positions[rotor.axis_atoms[0]]
    d = atoms.positions[rotor.axis_atoms[1]] - p0
    if atoms.pbc.any():
        d, _ = find_mic(d, atoms.cell, atoms.pbc)
    return p0, d / np.linalg.norm(d)


def _unwrapped(atoms, indices, origin):
    pts = atoms.positions[list(indices)] - origin
    if atoms.pbc.any():
        pts, _ = find_mic(pts, atoms.cell, atoms.pbc)
    return origin + pts


def rotate_rotor(atoms, rotor, angle_deg):
    """Copy of ``atoms`` with one ring turned by ``angle_deg``.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
    >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
    ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
    >>> _ = cage.build()
    >>> _ = cage.optimise(include_charges=False, logfile=None)
    >>> _ = cage.functionalise()
    >>> atoms, adjacency = cage.to_ase(), cage.adjacency()
    >>> linkers, nodes = cage.building_blocks()
    >>> from cage_isomer_builder.utils.flexibility import find_rotors, rotate_rotor
    >>> rotor = find_rotors(atoms, linkers, adjacency)[0]
    >>> turned = rotate_rotor(atoms, rotor, 30.0)
    >>> bonds = [(i, j) for i in adjacency for j in adjacency[i] if i < j]
    >>> bool(max(abs(atoms.get_distance(i, j) - turned.get_distance(i, j)) for i, j in bonds) < 1e-9)
    True
    """
    out = atoms.copy()
    origin, axis = rotor_axis(atoms, rotor)
    pts = _unwrapped(atoms, rotor.moving, origin)
    out.positions[list(rotor.moving)] = _rotate(pts, origin, axis, np.radians(angle_deg))
    return out


def rotation_samples(atoms, rotors, slot_indices, angles, centre=None, threshold=0.5,
                     adjacency=None):
    """
    Slot geometry at every rotation angle.

    Parameters
    ----------
    rotors : list of Rotor
    slot_indices : sequence of int
        Site atoms in slot order.
    angles : sequence of float
        Rotation angles in degrees (applied to every ring independently).
    centre : array-like, optional
        Pore centre, to re-classify endo/exo at each angle.

    Returns
    -------
    positions : np.ndarray, shape (n_slots, n_angles, 3)
    vectors : np.ndarray, shape (n_slots, n_angles, 3)
        Unit ring-C -> site direction.
    orientations : np.ndarray of str, shape (n_slots, n_angles)
    rotor_of_slot : np.ndarray of int
        Rotor of each slot (-1: the slot is on no rotor and stays put).

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
    >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
    ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
    >>> _ = cage.build()
    >>> _ = cage.optimise(include_charges=False, logfile=None)
    >>> _ = cage.functionalise()
    >>> atoms, adjacency = cage.to_ase(), cage.adjacency()
    >>> linkers, nodes = cage.building_blocks()
    >>> from cage_isomer_builder.utils.flexibility import find_rotors, rotation_samples
    >>> rotors = find_rotors(atoms, linkers, adjacency)
    >>> pos, vec, orientation, rotor_of_slot = rotation_samples(
    ...     atoms, rotors, cage._fg_anchor_indices, [0, 90, 180, 270],
    ...     centre=cage.pore_centre(), adjacency=adjacency)
    >>> pos.shape, sorted(set(rotor_of_slot.tolist()))      # 24 slots x 4 angles
    ((24, 4, 3), [0, 1, 2, 3, 4, 5])
    """
    if adjacency is None:
        adjacency = bond_graph(atoms)
    slot_indices = [int(s) for s in slot_indices]
    angles = np.radians(np.asarray(angles, dtype=float))
    n, m = len(slot_indices), len(angles)
    positions = np.repeat(atoms.positions[slot_indices][:, None, :], m, axis=1)
    carbons = [[j for j in adjacency[s] if atoms.numbers[j] > 1][0] for s in slot_indices]
    carbon_pos = np.repeat(atoms.positions[carbons][:, None, :], m, axis=1)
    rotor_of_slot = np.full(n, -1)
    for r, rotor in enumerate(rotors):
        origin, axis = rotor_axis(atoms, rotor)
        moving = set(rotor.moving)
        for k, (s, c) in enumerate(zip(slot_indices, carbons)):
            if s not in moving:
                continue
            rotor_of_slot[k] = r
            p_s = _unwrapped(atoms, [s], origin)[0]
            p_c = _unwrapped(atoms, [c], origin)[0]
            for t, ang in enumerate(angles):
                positions[k, t] = _rotate(p_s[None], origin, axis, ang)[0]
                carbon_pos[k, t] = (_rotate(p_c[None], origin, axis, ang)[0]
                                    if c in moving else p_c)
    vectors = positions - carbon_pos
    vectors /= np.linalg.norm(vectors, axis=2, keepdims=True)
    orientations = np.full((n, m), "", dtype="<U16")
    if centre is not None:
        inward = np.asarray(centre)[None, None, :] - carbon_pos
        inward /= np.linalg.norm(inward, axis=2, keepdims=True)
        cos = np.einsum("ijk,ijk->ij", vectors, inward)
        orientations = np.vectorize(lambda x: orientation_class(x, threshold))(cos).astype("<U16")
    return positions, vectors, orientations, rotor_of_slot


# Scaling

def scale_structure(atoms, groups, factor, centre=None):
    """
    Copy of ``atoms`` with every building block moved rigidly so its
    centroid-to-centre distance is multiplied by ``factor``.

    Parameters
    ----------
    groups : list of list of int
        Atoms of each building block (nodes and linkers). Atoms in no group
        stay where they are, so every atom should be in some group.
    centre : array-like, optional
        Default: centroid of all atoms (finite structures). For periodic
        structures the cell is scaled by ``factor`` and every block keeps
        its fractional centroid.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
    >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
    ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
    >>> _ = cage.build()
    >>> _ = cage.optimise(include_charges=False, logfile=None)
    >>> _ = cage.functionalise()
    >>> atoms, adjacency = cage.to_ase(), cage.adjacency()
    >>> linkers, nodes = cage.building_blocks()
    >>> import numpy as np
    >>> from cage_isomer_builder.utils.flexibility import scale_structure
    >>> centre = cage.pore_centre()
    >>> bigger = scale_structure(atoms, linkers + nodes, 1.1, centre)
    >>> node = nodes[0]
    >>> r0 = np.linalg.norm(atoms.positions[node].mean(axis=0) - centre)
    >>> r1 = np.linalg.norm(bigger.positions[node].mean(axis=0) - centre)
    >>> round(float(r1 / r0), 6)                          # each block moves out by 10 %
    1.1
    """
    out = atoms.copy()
    pos = atoms.positions.copy()
    if atoms.pbc.any():
        new_cell = atoms.cell[:] * factor
        for g in groups:
            g = list(g)
            local = _unwrapped(atoms, g, pos[g[0]])
            com = local.mean(axis=0)
            frac = np.linalg.solve(atoms.cell[:].T, com)
            new_com = new_cell.T @ frac
            pos[g] = local - com + new_com
        out.set_cell(new_cell, scale_atoms=False)
        out.positions = pos
        out.wrap()
        return out
    if centre is None:
        centre = pos.mean(axis=0)
    for g in groups:
        g = list(g)
        com = pos[g].mean(axis=0)
        pos[g] += (factor - 1.0) * (com - centre)
    out.positions = pos
    return out


def distance_scaling(atoms, groups, slot_indices, factors=(0.95, 1.0, 1.05), centre=None):
    """
    How every slot-slot distance changes with isotropic scaling.

    Returns
    -------
    pairs : np.ndarray, shape (n_pairs, 2)
        Slot pairs (i < j).
    distances : np.ndarray, shape (n_factors, n_pairs)
    slopes : np.ndarray, shape (n_pairs,)
        Least-squares d(distance)/d(factor). A pair whose two slots sit on
        building blocks that move apart rigidly has slope close to the
        distance between those blocks' centroids; two slots on one block
        have slope 0.

    Examples
    --------
    >>> from cage_isomer_builder import example_data
    >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
    >>> cage = Tri4Di6CageBuilder(node=example_data.path("uio66_tri_node"),
    ...                           linker=example_data.path("bdc"), scale_multiplier=0.225)
    >>> _ = cage.build()
    >>> _ = cage.optimise(include_charges=False, logfile=None)
    >>> _ = cage.functionalise()
    >>> atoms, adjacency = cage.to_ase(), cage.adjacency()
    >>> linkers, nodes = cage.building_blocks()
    >>> from cage_isomer_builder.utils.flexibility import distance_scaling
    >>> pairs, distances, slopes = distance_scaling(
    ...     atoms, linkers + nodes, cage._fg_anchor_indices, (0.95, 1.0, 1.05), cage.pore_centre())
    >>> distances.shape, bool((slopes >= -1e-9).all())    # 3 factors x 276 slot pairs
    ((3, 276), True)
    """
    slot_indices = [int(s) for s in slot_indices]
    i, j = np.triu_indices(len(slot_indices), k=1)
    rows = []
    for f in factors:
        scaled = scale_structure(atoms, groups, f, centre)
        p = scaled.positions[slot_indices]
        d = p[j] - p[i]
        if scaled.pbc.any():
            d, _ = find_mic(d, scaled.cell, scaled.pbc)
        rows.append(np.linalg.norm(d, axis=1))
    distances = np.array(rows)
    f = np.asarray(factors, dtype=float)
    slopes = np.polyfit(f, distances, 1)[0] if len(f) > 1 else np.zeros(len(i))
    return np.stack([i, j], axis=1), distances, slopes
