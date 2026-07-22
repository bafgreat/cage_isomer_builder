#!/usr/bin/env python3
__author__ = "Dr. Dinga Wonanke, Dr. Maryam Nurhuda, Dr. Matthew Addicoat"
__status__ = "production"

import stk
from cage_isomer_builder.cage.base import CageBuilder, _TwoPlusThree, _ThreePlusSix
from cage_isomer_builder.utils import symmetry


# ============================================================================
#  Tet topology classes
# ============================================================================

class Tet2Di4CageBuilder(CageBuilder):
    """Tet2Di4 — 2 tetratopic nodes + 4 ditopic linkers (TwoPlusFour).  D4h."""

    topology_name = "Tet2Di4"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.TwoPlusFour

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tet2di4_transformation_library(fg_anchors)


class Tet3_3Di3CageBuilder(CageBuilder):
    """Tet3-3Di3 — 3 tetratopic nodes + 6 ditopic linkers (_ThreePlusSix).  D3h."""

    topology_name = "Tet3-3Di3"
    default_node_fg_count = 4
    _stk_topology_cls = _ThreePlusSix

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tet3_3di3_transformation_library(fg_anchors)


class Tet4_4Di8CageBuilder(CageBuilder):
    """Tet4-4Di8 — 4 tetratopic nodes + 8 ditopic linkers (FourPlusEight).  D4h."""

    topology_name = "Tet4-4Di8"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.FourPlusEight

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tet4_4di8_transformation_library(fg_anchors)


class Tet5Di10CageBuilder(CageBuilder):
    """Tet5Di10 — 5 tetratopic nodes (pentagonal arrangement) + 10 ditopic linkers (FivePlusTen).  D5h."""

    topology_name = "Tet5Di10"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.FivePlusTen

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tet5di10_transformation_library(fg_anchors)


class Tet6Di12CageBuilder(CageBuilder):
    """Tet6Di12 — 6 tetratopic nodes + 12 ditopic linkers (SixPlusTwelve).  Oh."""

    topology_name = "Tet6Di12"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.SixPlusTwelve
    _use_collapser = False   # SixPlusTwelve converges without Collapser


class Tet8Di16CageBuilder(CageBuilder):
    """Tet8Di16 — 8 tetratopic nodes + 16 ditopic linkers (EightPlusSixteen).  D4d."""

    topology_name = "Tet8Di16"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.EightPlusSixteen

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tet8di16_transformation_library(fg_anchors)


class Tet16Di32CageBuilder(CageBuilder):
    """Tet16Di32 — M12L24, 12 tetratopic nodes + 24 ditopic linkers.  D4h."""

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
    """Tet24Di48 — M24L48, 24 tetratopic nodes + 48 ditopic linkers.  Oh."""

    topology_name = "Tet24Di48"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M24L48

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tet24di48_transformation_library(fg_anchors)


# ============================================================================
#  Tri topology classes
# ============================================================================

class Tri2Di3CageBuilder(CageBuilder):
    """Tri2Di3 — 2 tritopic (triamine) nodes + 3 ditopic linkers (_TwoPlusThree).  D3h."""

    topology_name = "Tri2Di3"
    default_node_fg_count = 3
    _stk_topology_cls = _TwoPlusThree

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tri2di3_transformation_library(fg_anchors)


class Tri4Di6CageBuilder(CageBuilder):
    """
    Tri4Di6 — 4 tritopic nodes + 6 ditopic linkers (FourPlusSix).  Td.

    Symmetry is derived from the 4 node CoMs via universal_transformation_library
    (base-class default), so multi-atom SBU nodes (e.g. Zr6 clusters) work
    correctly without the BFS-overlap problem that affects point-atom detection.
    """

    topology_name = "Tri4Di6"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.FourPlusSix
    _use_collapser = False


class Tri4_2Di6CageBuilder(CageBuilder):
    """Tri4-2Di6 — 4 tritopic nodes + 6 ditopic linkers (FourPlusSix2 isomeric variant).  D2h."""

    topology_name = "Tri4-2Di6"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.FourPlusSix2

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tri4_2di6_transformation_library(fg_anchors)


class Tri6Di9CageBuilder(CageBuilder):
    """Tri6Di9 — 6 tritopic nodes + 9 ditopic linkers (SixPlusNine).  D3h."""

    topology_name = "Tri6Di9"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.SixPlusNine

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tri6di9_transformation_library(fg_anchors)


class Tri8Di12CageBuilder(CageBuilder):
    """
    Tri8Di12 — 8 tritopic nodes + 12 ditopic linkers (EightPlusTwelve).  Oh.

    Symmetry derived from 8 node CoMs via universal_transformation_library.
    """

    topology_name = "Tri8Di12"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.EightPlusTwelve


class Tri20Di30CageBuilder(CageBuilder):
    """
    Tri20Di30 — 20 tritopic nodes + 30 ditopic linkers (TwentyPlusThirty).  Ih.

    Full enumeration of 4^30 isomers is impractical; use
    count_unique_isomers_burnside() instead.
    """

    topology_name = "Tri20Di30"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.TwentyPlusThirty

    def _get_point_group_transformation_library(self, fg_anchors):
        return symmetry.tri20di30_transformation_library(fg_anchors)


# ============================================================================
#  Tet+Tri topology classes
# ============================================================================

class Tet6Tri8CageBuilder(CageBuilder):
    """
    Tet6Tri8 — 6 tetratopic nodes + 8 tritopic linkers (SixPlusEight).  Oh.

    Isomer enumeration is not supported (tritopic linkers break the
    FG-per-linker assumption).
    """

    topology_name = "Tet6Tri8"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.SixPlusEight

    def _get_point_group_transformation_library(self, fg_anchors):
        raise NotImplementedError(
            "Isomer enumeration is not supported for Tet6Tri8 (tritopic linkers)."
        )


# ============================================================================
#  Tri+Tri topology classes
# ============================================================================

class Tri4Tri4CageBuilder(CageBuilder):
    """
    Tri4Tri4 — 4 + 4 tritopic building blocks (FourPlusFour).  Td.

    Isomer enumeration not yet supported.
    """

    topology_name = "Tri4Tri4"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.FourPlusFour

    def _get_point_group_transformation_library(self, fg_anchors):
        raise NotImplementedError(
            "Isomer enumeration is not supported for Tri4Tri4 (tritopic linkers)."
        )


class Tri1Tri1CageBuilder(CageBuilder):
    """
    Tri1Tri1 — 1 + 1 tritopic building blocks (OnePlusOne).  C3v.

    Isomer enumeration not yet supported.
    """

    topology_name = "Tri1Tri1"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.OnePlusOne

    def _get_point_group_transformation_library(self, fg_anchors):
        raise NotImplementedError(
            "Isomer enumeration is not supported for Tri1Tri1 (tritopic linkers)."
        )


class Tri2_2Tri2CageBuilder(CageBuilder):
    """
    Tri2²Tri2 — 2 + 2 tritopic building blocks (TwoPlusTwo).  D2h.

    Isomer enumeration not yet supported.
    """

    topology_name = "Tri2-2Tri2"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.TwoPlusTwo

    def _get_point_group_transformation_library(self, fg_anchors):
        raise NotImplementedError(
            "Isomer enumeration is not supported for Tri2-2Tri2 (tritopic linkers)."
        )


# ============================================================================
#  MxLy metal-organic cage classes
#
#  Node fg count = the metal vertex degree in the STK topology graph (not
#  always 4 - check before changing). Topologies with more than 8 metal
#  nodes can't use the base class's universal_transformation_library (SVD
#  over node permutations is only practical up to 8! = 40320), so isomer
#  enumeration is marked unsupported for those rather than left to fail
#  confusingly at call time.
# ============================================================================

class M2L4LanternCageBuilder(CageBuilder):
    """M2L4Lantern — 2 tetratopic nodes + 4 ditopic linkers.  D4h."""

    topology_name = "M2L4Lantern"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M2L4Lantern


class M3L6CageBuilder(CageBuilder):
    """M3L6 — 3 tetratopic nodes + 6 ditopic linkers.  D3h."""

    topology_name = "M3L6"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M3L6
    _use_collapser = False   # stk.Collapser hits an internal bug on this topology


class M4L6TetrahedronSpacerCageBuilder(CageBuilder):
    """M4L6TetrahedronSpacer — 4 tritopic nodes + 6 ditopic (spacered) linkers.  Td."""

    topology_name = "M4L6TetrahedronSpacer"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.M4L6TetrahedronSpacer
    _use_collapser = False   # stk.Collapser hits an internal bug on this topology


class M4L8CageBuilder(CageBuilder):
    """M4L8 — 4 tetratopic nodes + 8 ditopic linkers.  D2d."""

    topology_name = "M4L8"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M4L8


class M4L82CageBuilder(CageBuilder):
    """M4L82 — 4 tetratopic nodes + 8 ditopic linkers (M4L8 vertex-arrangement isomer).  D2d."""

    topology_name = "M4L8-2"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M4L82


class M4L8xCageBuilder(CageBuilder):
    """M4L8x — 4 tetratopic nodes + 8 ditopic linkers (M4L8 vertex-arrangement isomer).  D2d."""

    topology_name = "M4L8x"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M4L8x


class M6L12CubeCageBuilder(CageBuilder):
    """M6L12Cube — 6 tetratopic nodes + 12 ditopic linkers.  Oh."""

    topology_name = "M6L12Cube"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M6L12Cube


class M8L6CubeCageBuilder(CageBuilder):
    """
    M8L6Cube — 8 tritopic corner nodes + 6 tetratopic face linkers.  Oh.

    Unlike most other topologies here, ``linker`` must be tetratopic (4
    functional groups), not ditopic - it caps a cube face, not an edge.
    """

    topology_name = "M8L6Cube"
    default_node_fg_count = 3
    _stk_topology_cls = stk.cage.M8L6Cube


class M9L18CageBuilder(CageBuilder):
    """
    M9L18 — 9 tetratopic nodes + 18 ditopic linkers.

    Isomer enumeration is not supported: with 9 metal nodes,
    universal_transformation_library's node-permutation search
    (N! permutations) is past the practical limit (see its docstring).
    """

    topology_name = "M9L18"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.M9L18

    def _get_point_group_transformation_library(self, fg_anchors):
        raise NotImplementedError(
            "Isomer enumeration is not supported for M9L18 (9 node centres "
            "exceeds universal_transformation_library's 8-node limit)."
        )


class TenPlusTwentyCageBuilder(CageBuilder):
    """
    TenPlusTwenty — 10 tetratopic nodes + 20 ditopic linkers.

    Isomer enumeration not supported - see M9L18.
    """

    topology_name = "TenPlusTwenty"
    default_node_fg_count = 4
    _stk_topology_cls = stk.cage.TenPlusTwenty
    _use_collapser = False   # stk.Collapser hits an internal bug on this topology

    def _get_point_group_transformation_library(self, fg_anchors):
        raise NotImplementedError(
            "Isomer enumeration is not supported for TenPlusTwenty (10 node "
            "centres exceeds universal_transformation_library's 8-node limit)."
        )


class TwelvePlusThirtyCageBuilder(CageBuilder):
    """
    TwelvePlusThirty — 12 pentatopic nodes + 30 ditopic linkers.  Ih.

    Isomer enumeration not supported - see M9L18.
    """

    topology_name = "TwelvePlusThirty"
    default_node_fg_count = 5
    _stk_topology_cls = stk.cage.TwelvePlusThirty
    _use_collapser = False   # stk.Collapser hits an internal bug on this topology

    def _get_point_group_transformation_library(self, fg_anchors):
        raise NotImplementedError(
            "Isomer enumeration is not supported for TwelvePlusThirty (12 "
            "node centres exceeds universal_transformation_library's 8-node limit)."
        )
