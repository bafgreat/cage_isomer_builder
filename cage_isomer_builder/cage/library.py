#!/usr/bin/env python3
"""
One cage builder per topology. Each sets the STK topology graph, the
number of connection points of its node and, where one exists, its
hand-written symmetry library for ``symmetry_method="topology"``.
"""
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

import stk
from cage_isomer_builder.cage.base import CageBuilder, _TwoPlusThree, _ThreePlusSix
from cage_isomer_builder.utils import symmetry


# Tet topology classes

class Tet2Di4CageBuilder(CageBuilder):
    """Tet2Di4: 2 tetratopic nodes + 4 ditopic linkers (TwoPlusFour). Point group D4h.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tet2Di4CageBuilder
    >>> cage = Tet2Di4CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C24H16N8Zr2'
    """

    topology_name = "Tet2Di4"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.TwoPlusFour

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tet2di4_transformation_library(fg_anchors)


class Tet3_3Di3CageBuilder(CageBuilder):
    """Tet3-3Di3: 3 tetratopic nodes + 6 ditopic linkers (_ThreePlusSix). Point group D3h.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tet3_3Di3CageBuilder
    >>> cage = Tet3_3Di3CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C36H24N12Zr3'
    """

    topology_name = "Tet3-3Di3"
    default_node_fg_count = 4
    _stk_topology_cls = _ThreePlusSix

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tet3_3di3_transformation_library(fg_anchors)


class Tet4_4Di8CageBuilder(CageBuilder):
    """Tet4-4Di8: 4 tetratopic nodes + 8 ditopic linkers (FourPlusEight). Point group D4h.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tet4_4Di8CageBuilder
    >>> cage = Tet4_4Di8CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C48H32N16Zr4'
    """

    topology_name = "Tet4-4Di8"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.FourPlusEight

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tet4_4di8_transformation_library(fg_anchors)


class Tet5Di10CageBuilder(CageBuilder):
    """Tet5Di10: 5 tetratopic nodes (pentagonal arrangement) + 10 ditopic linkers (FivePlusTen). Point group D5h.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tet5Di10CageBuilder
    >>> cage = Tet5Di10CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C60H40N20Zr5'
    """

    topology_name = "Tet5Di10"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.FivePlusTen

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tet5di10_transformation_library(fg_anchors)


class Tet6Di12CageBuilder(CageBuilder):
    """Tet6Di12: 6 tetratopic nodes + 12 ditopic linkers (SixPlusTwelve). Point group Oh.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tet6Di12CageBuilder
    >>> cage = Tet6Di12CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C72H48N24Zr6'
    """

    topology_name = "Tet6Di12"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.SixPlusTwelve
    _use_collapser = False   # SixPlusTwelve converges without Collapser


class Tet8Di16CageBuilder(CageBuilder):
    """Tet8Di16: 8 tetratopic nodes + 16 ditopic linkers (EightPlusSixteen). Point group D4d.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tet8Di16CageBuilder
    >>> cage = Tet8Di16CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C96H64N32Zr8'
    """

    topology_name = "Tet8Di16"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.EightPlusSixteen

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tet8di16_transformation_library(fg_anchors)


class Tet16Di32CageBuilder(CageBuilder):
    """Tet16Di32: M12L24, 12 tetratopic nodes + 24 ditopic linkers. Point group D4h.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tet16Di32CageBuilder
    >>> cage = Tet16Di32CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C144H96N48Zr12'
    """

    topology_name = "Tet16Di32"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M12L24

    def __init__(self, *args, optimizer_step=1.5, distance_threshold=6.0, **kwargs):
        super().__init__(
            *args,
            optimizer_step=optimizer_step,
            distance_threshold=distance_threshold,
            **kwargs,
        )

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tet16di32_transformation_library(fg_anchors)


class Tet24Di48CageBuilder(CageBuilder):
    """Tet24Di48: M24L48, 24 tetratopic nodes + 48 ditopic linkers. Point group Oh.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tet24Di48CageBuilder
    >>> cage = Tet24Di48CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C288H192N96Zr24'
    """

    topology_name = "Tet24Di48"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M24L48

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tet24di48_transformation_library(fg_anchors)


# Tri topology classes

class Tri2Di3CageBuilder(CageBuilder):
    """Tri2Di3: 2 tritopic (triamine) nodes + 3 ditopic linkers (_TwoPlusThree). Point group D3h.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tri2Di3CageBuilder
    >>> cage = Tri2Di3CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C18H12N6Zr2'
    """

    topology_name = "Tri2Di3"
    default_node_fg_count = 3
    _stk_topology_cls = _TwoPlusThree

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tri2di3_transformation_library(fg_anchors)


class Tri4Di6CageBuilder(CageBuilder):
    """
    Tri4Di6: 4 tritopic nodes + 6 ditopic linkers (FourPlusSix). Point group Td.

    With ``symmetry_method="topology"`` the operations come from the four
    node centres (universal_transformation_library), which also works for
    multi-atom nodes such as Zr6 clusters.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tri4Di6CageBuilder
    >>> cage = Tri4Di6CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C36H24N12Zr4'
    """

    topology_name = "Tri4Di6"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.FourPlusSix
    _use_collapser = False


class Tri4_2Di6CageBuilder(CageBuilder):
    """Tri4-2Di6: 4 tritopic nodes + 6 ditopic linkers (FourPlusSix2 isomeric variant). Point group D2h.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tri4_2Di6CageBuilder
    >>> cage = Tri4_2Di6CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C36H24N12Zr4'
    """

    topology_name = "Tri4-2Di6"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.FourPlusSix2

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tri4_2di6_transformation_library(fg_anchors)


class Tri6Di9CageBuilder(CageBuilder):
    """Tri6Di9: 6 tritopic nodes + 9 ditopic linkers (SixPlusNine). Point group D3h.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tri6Di9CageBuilder
    >>> cage = Tri6Di9CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C54H36N18Zr6'
    """

    topology_name = "Tri6Di9"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.SixPlusNine

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tri6di9_transformation_library(fg_anchors)


class Tri8Di12CageBuilder(CageBuilder):
    """
    Tri8Di12: 8 tritopic nodes + 12 ditopic linkers (EightPlusTwelve). Point group Oh.

    With ``symmetry_method="topology"`` the operations come from the eight
    node centres (universal_transformation_library).

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tri8Di12CageBuilder
    >>> cage = Tri8Di12CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C72H48N24Zr8'
    """

    topology_name = "Tri8Di12"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.EightPlusTwelve


class Tri20Di30CageBuilder(CageBuilder):
    """
    Tri20Di30: 20 tritopic nodes + 30 ditopic linkers (TwentyPlusThirty). Point group Ih.

    Full enumeration of 4^30 isomers is impractical; use
    count_unique_isomers_burnside() instead.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tri20Di30CageBuilder
    >>> cage = Tri20Di30CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C180H120N60Zr20'
    """

    topology_name = "Tri20Di30"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.TwentyPlusThirty

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tri20di30_transformation_library(fg_anchors)


# Tet+Tri topology classes

class Tet6Tri8CageBuilder(CageBuilder):
    """
    Tet6Tri8: 6 tetratopic nodes + 8 tritopic linkers (SixPlusEight). Point group Oh.

    Isomers are counted from the point group detected from the atoms (the
    default ``symmetry_method="detect"``); there is no hand-written
    library for ``symmetry_method="topology"``.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tet6Tri8CageBuilder
    >>> cage = Tet6Tri8CageBuilder()           # default single-atom [Zr] node + 1,3,5-triaminobenzene
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C48H24N24Zr6'
    """

    topology_name = "Tet6Tri8"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.SixPlusEight
    default_linker = "Nc1cc(N)cc(N)c1"   # tritopic: 1,3,5-triaminobenzene

    def _get_point_group_transformation_library(self, fg_anchors):
        raise NotImplementedError(
            "symmetry_method='topology' has no library for Tet6Tri8 (tritopic linkers); "
            "use the default symmetry_method='detect'."
        )


# Tri+Tri topology classes

class Tri4Tri4CageBuilder(CageBuilder):
    """
    Tri4Tri4: 4 + 4 tritopic building blocks (FourPlusFour). Point group Td.

    Isomers are counted from the point group detected from the atoms (the
    default ``symmetry_method="detect"``); there is no hand-written
    library for ``symmetry_method="topology"``.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tri4Tri4CageBuilder
    >>> cage = Tri4Tri4CageBuilder()           # default single-atom [Zr] node + 1,3,5-triaminobenzene
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C24H12N12Zr4'
    """

    topology_name = "Tri4Tri4"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.FourPlusFour
    default_linker = "Nc1cc(N)cc(N)c1"   # tritopic: 1,3,5-triaminobenzene

    def _get_point_group_transformation_library(self, fg_anchors):
        raise NotImplementedError(
            "symmetry_method='topology' has no library for Tri4Tri4 (tritopic linkers); "
            "use the default symmetry_method='detect'."
        )


class Tri1Tri1CageBuilder(CageBuilder):
    """
    Tri1Tri1: 1 + 1 tritopic building blocks (OnePlusOne). Point group C3v.

    Isomers are counted from the point group detected from the atoms (the
    default ``symmetry_method="detect"``); there is no hand-written
    library for ``symmetry_method="topology"``.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tri1Tri1CageBuilder
    >>> cage = Tri1Tri1CageBuilder()           # default single-atom [Zr] node + 1,3,5-triaminobenzene
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C6H3N3Zr'
    """

    topology_name = "Tri1Tri1"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.OnePlusOne
    default_linker = "Nc1cc(N)cc(N)c1"   # tritopic: 1,3,5-triaminobenzene

    def _get_point_group_transformation_library(self, fg_anchors):
        raise NotImplementedError(
            "symmetry_method='topology' has no library for Tri1Tri1 (tritopic linkers); "
            "use the default symmetry_method='detect'."
        )


class Tri2_2Tri2CageBuilder(CageBuilder):
    """
    Tri2²Tri2: 2 + 2 tritopic building blocks (TwoPlusTwo). Point group D2h.

    Isomers are counted from the point group detected from the atoms (the
    default ``symmetry_method="detect"``); there is no hand-written
    library for ``symmetry_method="topology"``.

    Examples
    --------
    >>> from cage_isomer_builder.cage import Tri2_2Tri2CageBuilder
    >>> cage = Tri2_2Tri2CageBuilder()           # default single-atom [Zr] node + 1,3,5-triaminobenzene
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C12H6N6Zr2'
    """

    topology_name = "Tri2-2Tri2"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.TwoPlusTwo
    default_linker = "Nc1cc(N)cc(N)c1"   # tritopic: 1,3,5-triaminobenzene

    def _get_point_group_transformation_library(self, fg_anchors):
        raise NotImplementedError(
            "symmetry_method='topology' has no library for Tri2-2Tri2 (tritopic linkers); "
            "use the default symmetry_method='detect'."
        )


# MxLy metal-organic cages.
# The node fg count is the metal vertex degree of the STK topology graph,
# which is not always 4. Topologies with more than 8 metal nodes have no
# library for symmetry_method="topology": the node-permutation search of
# universal_transformation_library is only practical up to 8! = 40320.
# Their isomers are counted from the detected point group (the default).

class M2L4LanternCageBuilder(CageBuilder):
    """M2L4Lantern: 2 tetratopic nodes + 4 ditopic linkers. Point group D4h.

    Examples
    --------
    >>> from cage_isomer_builder.cage import M2L4LanternCageBuilder
    >>> cage = M2L4LanternCageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C24H16N8Zr2'
    """

    topology_name = "M2L4Lantern"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M2L4Lantern


class M3L6CageBuilder(CageBuilder):
    """M3L6: 3 tetratopic nodes + 6 ditopic linkers. Point group D3h.

    Examples
    --------
    >>> from cage_isomer_builder.cage import M3L6CageBuilder
    >>> cage = M3L6CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C36H24N12Zr3'
    """

    topology_name = "M3L6"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M3L6
    _use_collapser = False   # stk.Collapser hits an internal bug on this topology


class M4L6TetrahedronSpacerCageBuilder(CageBuilder):
    """M4L6TetrahedronSpacer: 4 tritopic nodes + 6 ditopic (spacered) linkers. Point group Td.

    Examples
    --------
    >>> from cage_isomer_builder.cage import M4L6TetrahedronSpacerCageBuilder
    >>> cage = M4L6TetrahedronSpacerCageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C36H24N12Zr4'
    """

    topology_name = "M4L6TetrahedronSpacer"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.M4L6TetrahedronSpacer
    _use_collapser = False   # stk.Collapser hits an internal bug on this topology


class M4L8CageBuilder(CageBuilder):
    """M4L8: 4 tetratopic nodes + 8 ditopic linkers. Point group D2d.

    Examples
    --------
    >>> from cage_isomer_builder.cage import M4L8CageBuilder
    >>> cage = M4L8CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C48H32N16Zr4'
    """

    topology_name = "M4L8"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M4L8


class M4L82CageBuilder(CageBuilder):
    """M4L82: 4 tetratopic nodes + 8 ditopic linkers (M4L8 vertex-arrangement isomer). Point group D2d.

    Examples
    --------
    >>> from cage_isomer_builder.cage import M4L82CageBuilder
    >>> cage = M4L82CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C48H32N16Zr4'
    """

    topology_name = "M4L8-2"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M4L82


class M4L8xCageBuilder(CageBuilder):
    """M4L8x: 4 tetratopic nodes + 8 ditopic linkers (M4L8 vertex-arrangement isomer). Point group D2d.

    Examples
    --------
    >>> from cage_isomer_builder.cage import M4L8xCageBuilder
    >>> cage = M4L8xCageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C48H32N16Zr4'
    """

    topology_name = "M4L8x"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M4L8x


class M6L12CubeCageBuilder(CageBuilder):
    """M6L12Cube: 6 tetratopic nodes + 12 ditopic linkers. Point group Oh.

    Examples
    --------
    >>> from cage_isomer_builder.cage import M6L12CubeCageBuilder
    >>> cage = M6L12CubeCageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C72H48N24Zr6'
    """

    topology_name = "M6L12Cube"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M6L12Cube


class M8L6CubeCageBuilder(CageBuilder):
    """
    M8L6Cube: 8 tritopic corner nodes + 6 tetratopic face linkers. Point group Oh.

    Unlike most other topologies here, ``linker`` must be tetratopic (4
    functional groups): it caps a cube face, not an edge.

    Examples
    --------
    >>> from cage_isomer_builder.cage import M8L6CubeCageBuilder
    >>> cage = M8L6CubeCageBuilder()           # default single-atom [Zr] node + 1,2,4,5-tetraaminobenzene
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C36H12N24Zr8'
    """

    topology_name = "M8L6Cube"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.M8L6Cube
    default_linker = "Nc1cc(N)c(N)cc1N"   # tetratopic: 1,2,4,5-tetraaminobenzene


class M9L18CageBuilder(CageBuilder):
    """
    M9L18: 9 tetratopic nodes + 18 ditopic linkers.

    Isomers are counted from the point group detected from the atoms (the
    default ``symmetry_method="detect"``). ``symmetry_method="topology"`` is
    not available: with 9 metal nodes, universal_transformation_library's
    node-permutation search (N! permutations) is past its practical limit.

    Examples
    --------
    >>> from cage_isomer_builder.cage import M9L18CageBuilder
    >>> cage = M9L18CageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C108H72N36Zr9'
    """

    topology_name = "M9L18"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M9L18

    def _get_point_group_transformation_library(self, fg_anchors):
        raise NotImplementedError(
            "symmetry_method='topology' has no library for M9L18: 9 node centres "
            "are more than universal_transformation_library can search (8); "
            "use the default symmetry_method='detect'."
        )


class TenPlusTwentyCageBuilder(CageBuilder):
    """
    TenPlusTwenty: 10 tetratopic nodes + 20 ditopic linkers.

    Isomers are counted from the detected point group (the default); the
    node-permutation library of ``symmetry_method="topology"`` is not
    available (see M9L18).

    Examples
    --------
    >>> from cage_isomer_builder.cage import TenPlusTwentyCageBuilder
    >>> cage = TenPlusTwentyCageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C120H80N40Zr10'
    """

    topology_name = "TenPlusTwenty"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.TenPlusTwenty
    _use_collapser = False   # stk.Collapser hits an internal bug on this topology

    def _get_point_group_transformation_library(self, fg_anchors):
        raise NotImplementedError(
            "symmetry_method='topology' has no library for TenPlusTwenty: 10 node centres "
            "are more than universal_transformation_library can search (8); "
            "use the default symmetry_method='detect'."
        )


class TwelvePlusThirtyCageBuilder(CageBuilder):
    """
    TwelvePlusThirty: 12 pentatopic nodes + 30 ditopic linkers. Point group Ih.

    Isomers are counted from the detected point group (the default); the
    node-permutation library of ``symmetry_method="topology"`` is not
    available (see M9L18).

    Examples
    --------
    >>> from cage_isomer_builder.cage import TwelvePlusThirtyCageBuilder
    >>> cage = TwelvePlusThirtyCageBuilder()           # default single-atom [Zr] node + p-phenylenediamine
    >>> _ = cage.build()
    >>> cage.to_ase().get_chemical_formula()
    'C180H120N60Zr12'
    """

    topology_name = "TwelvePlusThirty"
    default_node_fg_count = 5
    _stk_topology_cls = stk.cage.TwelvePlusThirty
    _use_collapser = False   # stk.Collapser hits an internal bug on this topology

    def _get_point_group_transformation_library(self, fg_anchors):
        raise NotImplementedError(
            "symmetry_method='topology' has no library for TwelvePlusThirty: 12 node centres "
            "are more than universal_transformation_library can search (8); "
            "use the default symmetry_method='detect'."
        )
