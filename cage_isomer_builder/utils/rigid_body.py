#!/usr/bin/env python3
"""
ASE constraint that treats groups of atoms as rigid bodies.

Used to relax a cage's inter-SBU placement (node-to-linker distances/angles)
while keeping every SBU's own internal geometry exactly as supplied.
"""
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

from typing import Sequence

import ase.constraints as _ase_constraints
import numpy as np
from ase import Atoms
from ase.constraints.constraint import FixConstraint
from scipy.spatial.transform import Rotation


class FixRigidBodies(FixConstraint):
    """
    Constrain groups of atoms to move only as rigid bodies.

    Each group's reference shape (atom positions relative to its
    mass-weighted centroid) never changes; the constrained optimiser may
    only translate and rotate each group as a whole. Build one from a
    structure with :meth:`FixRigidBodies.from_atoms`. The constructor takes
    the reference data directly (not an ``Atoms`` object) so that
    ``todict()`` and ``dict2constraint`` round-trip, as ASE's
    ``Trajectory`` writer requires.

    Parameters
    ----------
    groups : sequence of sequence of int
        Atom indices per rigid body.
    group_masses : sequence of sequence of float
        Per-atom mass, aligned with ``groups``.
    group_ref_rel : sequence of (N, 3) array-like
        Reference positions of each group's atoms relative to its
        mass-weighted centroid, aligned with ``groups``.

    Examples
    --------
    >>> import numpy as np
    >>> from ase.build import molecule
    >>> from cage_isomer_builder.utils.rigid_body import FixRigidBodies
    >>> a, b = molecule("H2O"), molecule("H2O")
    >>> b.translate([3.0, 0.0, 0.0])
    >>> pair = a + b                               # two waters, atoms 0-2 and 3-5
    >>> constraint = FixRigidBodies.from_atoms(pair, [[0, 1, 2], [3, 4, 5]])
    >>> pair.set_constraint(constraint)
    >>> new = pair.positions.copy()
    >>> new[1] += [0.3, 0.0, 0.0]                  # try to stretch an O-H bond
    >>> pair.set_positions(new)                    # the constraint keeps each water rigid
    >>> round(float(pair.get_distance(0, 1)), 4) == round(float(a.get_distance(0, 1)), 4)
    True
    """

    def __init__(self, groups: Sequence[Sequence[int]], group_masses, group_ref_rel):
        self.group_indices = [np.asarray(g, dtype=int) for g in groups]
        self.group_masses = [np.asarray(m, dtype=float) for m in group_masses]
        self.group_ref_rel = [np.asarray(r, dtype=float) for r in group_ref_rel]

    @classmethod
    def from_atoms(
        cls, atoms: Atoms, groups: Sequence[Sequence[int]]
    ) -> "FixRigidBodies":
        """
        Build from a structure, keeping each group's current shape.

        Groups with fewer than 2 atoms are dropped: a single atom has no
        shape to keep.

        Examples
        --------
        >>> import numpy as np
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.rigid_body import FixRigidBodies
        >>> a, b = molecule("H2O"), molecule("H2O")
        >>> b.translate([3.0, 0.0, 0.0])
        >>> pair = a + b                               # two waters, atoms 0-2 and 3-5
        >>> constraint = FixRigidBodies.from_atoms(pair, [[0, 1, 2], [3, 4, 5]])
        >>> [g.tolist() for g in constraint.group_indices]
        [[0, 1, 2], [3, 4, 5]]
        """
        masses = atoms.get_masses()
        positions = atoms.get_positions()

        kept_groups: list[np.ndarray] = []
        kept_masses: list[np.ndarray] = []
        kept_ref_rel: list[np.ndarray] = []

        for group in groups:
            indices = np.asarray(group, dtype=int)
            if len(indices) < 2:
                continue
            m = masses[indices]
            pos = positions[indices]
            centroid = (m[:, None] * pos).sum(axis=0) / m.sum()
            kept_groups.append(indices)
            kept_masses.append(m)
            kept_ref_rel.append(pos - centroid)

        return cls(kept_groups, kept_masses, kept_ref_rel)

    def get_removed_dof(self, atoms):
        """
        Degrees of freedom removed: 3N - 6 per rigid group of N atoms
        (ASE constraint interface).

        Examples
        --------
        >>> import numpy as np
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.rigid_body import FixRigidBodies
        >>> a, b = molecule("H2O"), molecule("H2O")
        >>> b.translate([3.0, 0.0, 0.0])
        >>> pair = a + b                               # two waters, atoms 0-2 and 3-5
        >>> constraint = FixRigidBodies.from_atoms(pair, [[0, 1, 2], [3, 4, 5]])
        >>> constraint.get_removed_dof(pair)           # 2 waters x (9 - 6)
        6
        """
        return sum(max(0, 3 * len(g) - 6) for g in self.group_indices)

    def adjust_positions(self, atoms, new):
        """
        Replace ``new`` positions in place by the closest rigid-body move of
        each group (ASE constraint interface).

        Examples
        --------
        >>> import numpy as np
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.rigid_body import FixRigidBodies
        >>> a, b = molecule("H2O"), molecule("H2O")
        >>> b.translate([3.0, 0.0, 0.0])
        >>> pair = a + b                               # two waters, atoms 0-2 and 3-5
        >>> constraint = FixRigidBodies.from_atoms(pair, [[0, 1, 2], [3, 4, 5]])
        >>> new = pair.positions.copy()
        >>> new[0] += [0.0, 0.0, 0.5]                  # move one atom only
        >>> constraint.adjust_positions(pair, new)
        >>> bool(np.isclose(np.linalg.norm(new[0] - new[1]), a.get_distance(0, 1)))
        True
        """
        for indices, masses, ref_rel in zip(
            self.group_indices, self.group_masses, self.group_ref_rel
        ):
            trial = new[indices]
            total_mass = masses.sum()
            trial_centroid = (masses[:, None] * trial).sum(axis=0) / total_mass
            trial_rel = trial - trial_centroid

            rotation, _ = Rotation.align_vectors(trial_rel, ref_rel, weights=masses)
            new[indices] = trial_centroid + rotation.apply(ref_rel)

    def adjust_forces(self, atoms, forces):
        """
        Replace forces in place by each group's net force and torque, spread
        back over its atoms as a rigid-body motion (ASE constraint
        interface).

        Examples
        --------
        >>> import numpy as np
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.rigid_body import FixRigidBodies
        >>> a, b = molecule("H2O"), molecule("H2O")
        >>> b.translate([3.0, 0.0, 0.0])
        >>> pair = a + b                               # two waters, atoms 0-2 and 3-5
        >>> constraint = FixRigidBodies.from_atoms(pair, [[0, 1, 2], [3, 4, 5]])
        >>> forces = np.zeros((6, 3))
        >>> forces[1] = [1.0, 0.0, 0.0]                # push one H only
        >>> constraint.adjust_forces(pair, forces)
        >>> round(float(forces[:3].sum(axis=0)[0]), 6)   # the whole water feels the net force
        1.0
        """
        # Newton-Euler rigid-body force projection: keep only the force
        # components that produce net translation/rotation of the group as a
        # whole, discarding anything that would deform it internally.
        positions = atoms.get_positions()
        for indices, masses in zip(self.group_indices, self.group_masses):
            pos = positions[indices]
            total_mass = masses.sum()
            centroid = (masses[:, None] * pos).sum(axis=0) / total_mass
            d = pos - centroid

            f = forces[indices]
            f_net = f.sum(axis=0)
            torque = np.cross(d, f).sum(axis=0)

            inertia = np.zeros((3, 3))
            for m_i, d_i in zip(masses, d):
                inertia += m_i * (np.dot(d_i, d_i) * np.eye(3) - np.outer(d_i, d_i))
            # pinv handles the singular case (e.g. a linear/2-atom group,
            # where rotation about the inter-atom axis is undefined).
            alpha = np.linalg.pinv(inertia) @ torque

            f_trans = (masses[:, None] / total_mass) * f_net[None, :]
            f_rot = masses[:, None] * np.cross(alpha[None, :], d)
            forces[indices] = f_trans + f_rot

    def todict(self):
        """
        Dictionary form, as used by ASE to store the constraint in a trajectory.

        Examples
        --------
        >>> import numpy as np
        >>> from ase.build import molecule
        >>> from cage_isomer_builder.utils.rigid_body import FixRigidBodies
        >>> a, b = molecule("H2O"), molecule("H2O")
        >>> b.translate([3.0, 0.0, 0.0])
        >>> pair = a + b                               # two waters, atoms 0-2 and 3-5
        >>> constraint = FixRigidBodies.from_atoms(pair, [[0, 1, 2], [3, 4, 5]])
        >>> constraint.todict()["name"]
        'FixRigidBodies'
        """
        return {
            "name": "FixRigidBodies",
            "kwargs": {
                "groups": [g.tolist() for g in self.group_indices],
                "group_masses": [m.tolist() for m in self.group_masses],
                "group_ref_rel": [r.tolist() for r in self.group_ref_rel],
            },
        }

    def __repr__(self):
        sizes = [len(g) for g in self.group_indices]
        return f"FixRigidBodies(groups={sizes})"


# Register with ASE's constraint registry so that `dict2constraint` (used
# when appending trajectory frames) can rebuild this constraint from
# todict().
if "FixRigidBodies" not in _ase_constraints.__all__:
    _ase_constraints.__all__.append("FixRigidBodies")
_ase_constraints.FixRigidBodies = FixRigidBodies
