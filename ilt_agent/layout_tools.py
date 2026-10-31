"""LangChain tools for the deterministic layout-reading / geometry toolkit (layout.LayoutModel).

Grouped by purpose; every tool returns compact JSON with nm coordinates.
"""
from __future__ import annotations

import json

from langchain_core.tools import tool

from .layout import CAPABILITIES, LayoutModel
from .tools import _dump


def _opt(v):
    """Tools take 0 / '' for 'not given' (simpler schemas for the model)."""
    return None if v in (0, 0.0, "", None) else v


def build_layout_tools(lm: LayoutModel, analyzer=None):
    findings = (lambda: analyzer.findings()) if analyzer is not None else (lambda: None)

    # ---------------------------------------------------------------- structure
    @tool
    def list_capabilities() -> str:
        """Overview of the layout/geometry tool groups and what each tool extracts."""
        return json.dumps(CAPABILITIES)

    @tool
    def layout_structure() -> str:
        """Read the target layout: polygon classes, dominant line direction, pitch, tracks used,
        line widths, number of line ends and how their gaps are typed. Good first call."""
        return _dump(lm.layout_structure())

    @tool
    def list_polygons(x0: float = 0.0, y0: float = 0.0, x1: float = 0.0, y1: float = 0.0,
                      polygon_class: str = "", limit: int = 30) -> str:
        """Target polygons (id, class, bbox, width, length, line-end count, frame cut). Optional
        region (x0,y0,x1,y1; all 0 = whole clip) and class filter: line | L-shape | T-shape | pad | island | complex."""
        reg = (x0, y0, x1, y1) if any((x0, y0, x1, y1)) else (None,) * 4
        return _dump(lm.list_polygons(*reg, cls=polygon_class, limit=limit))

    @tool
    def describe_polygon(poly_id: int) -> str:
        """Everything about one target polygon: class, track, every edge with its width/space and
        what it faces, neighbouring polygons with min spacing, its line ends."""
        return _dump(lm.describe_polygon(poly_id))

    @tool
    def list_line_ends(max_gap_nm: float = 0.0, gap_kind: str = "", limit: int = 30) -> str:
        """All line ends with facing direction, width, gap to what they face, gap type
        (end-end = tip-to-tip, end-side, none) and dense/semi-isolated/isolated context. Sorted by gap."""
        return _dump(lm.list_line_ends(_opt(max_gap_nm), gap_kind, limit))

    @tool
    def pattern_context(x_nm: float, y_nm: float) -> str:
        """Local context of a point: nearest polygon, inside/outside, target and mask density at
        several radii, track and whether the neighbouring tracks are occupied there."""
        return _dump(lm.pattern_context(x_nm, y_nm))

    # --------------------------------------------------------------------- drc
    @tool
    def width_space_statistics() -> str:
        """DRC-style edge-pair statistics of the target: line widths, line-end widths, and spaces
        split into side-side, end-end (tip-to-tip), end-side and notch."""
        return _dump(lm.width_space_statistics())

    @tool
    def design_rule_check(min_width: float = 0.0, min_space: float = 0.0, min_t2t: float = 0.0,
                          min_e2s: float = 0.0, min_corner_to_corner: float = 0.0, min_area: float = 0.0,
                          min_edge_length: float = 0.0, limit: int = 30) -> str:
        """Check the target against design rules (nm / nm2; 0 = rule not checked). Returns violations
        with location, measured value and limit."""
        return _dump(lm.design_rule_check(_opt(min_width), _opt(min_space), _opt(min_t2t), _opt(min_e2s),
                                          _opt(min_corner_to_corner), _opt(min_area), _opt(min_edge_length), limit))

    @tool
    def corner_to_corner_spacing(max_nm: float = 120.0, limit: int = 20) -> str:
        """Closest convex-corner-to-other-polygon distances (diagonal spacing, e.g. line end to the
        corner of a neighbour)."""
        return _dump(lm.corner_to_corner_spacing(max_nm, limit))

    # ---------------------------------------------------------------- patterns
    @tool
    def pattern_catalog(anchor: str = "line_end", radius_nm: float = 80.0, limit: int = 15) -> str:
        """Group all line ends (anchor='line_end') or convex corners (anchor='corner') by the target
        geometry around them (rotation/mirror invariant). Shows how many distinct contexts exist."""
        return _dump(lm.pattern_catalog(anchor, radius_nm, limit))

    @tool
    def find_similar_patterns(x_nm: float, y_nm: float, radius_nm: float = 60.0,
                              max_mismatch_frac: float = 0.03, limit: int = 20) -> str:
        """Find every place in the clip whose target geometry matches the window around (x, y)
        (any rotation/mirror, up to max_mismatch_frac differing pixels) -- e.g. all copies of a hotspot."""
        return _dump(lm.find_similar_patterns(x_nm, y_nm, radius_nm, max_mismatch_frac, limit))

    # -------------------------------------------------------------------- mask
    @tool
    def mask_target_correspondence() -> str:
        """Which mask shapes belong to which target polygons: detached assist features, mask shapes
        that join two different targets (with neck width and target gap), targets without mask."""
        return _dump(lm.mask_target_correspondence())

    @tool
    def mask_bias_statistics() -> str:
        """Mask edge displacement relative to the drawn target edges along their normals (bias
        distribution, largest-bias locations); corners and frame halo excluded."""
        return _dump(lm.mask_bias_statistics())

    @tool
    def line_end_treatment(limit: int = 30) -> str:
        """Per line end: how far the mask extends beyond the drawn end and how wide it is just inside,
        at and beyond the end (hammerhead ratio)."""
        return _dump(lm.line_end_treatment(limit))

    @tool
    def assist_features(band_nm: float = 15.0, limit: int = 20) -> str:
        """Assist features: detached mask shapes (size, width, distance to target), attached appendages
        reaching farther than band_nm from any target, and mask shapes joining two targets."""
        return _dump(lm.assist_features(band_nm, limit))

    @tool
    def mask_complexity() -> str:
        """Mask complexity: rectangle-decomposition count (VSB shot proxy), polygons, vertices,
        perimeter, curvature / smallest radii."""
        return _dump(lm.mask_complexity())

    # ---------------------------------------------------------------- contours
    @tool
    def gauge_cd_table(kind: str = "", limit: int = 25) -> str:
        """Automatic CD gauges on the target (width across lines, space between lines, tip-to-tip),
        measured on every process condition: nominal/min/max CD, worst condition, margin to spec,
        dose sensitivity, MEEF, focus tilt/curvature. Worst margin first. kind: '' | width | space | tip-to-tip."""
        return _dump(lm.gauge_cd_table(kind, True, limit))

    @tool
    def process_window(gauge_id: str) -> str:
        """Full CD-vs-condition table (dose, focus, mask bias) for one gauge id from gauge_cd_table."""
        return _dump(lm.process_window(gauge_id))

    @tool
    def line_end_pullback(limit: int = 25) -> str:
        """Printed line-end position vs. drawn end on every condition (pull-back > 0, overshoot < 0),
        with the worst condition per end."""
        return _dump(lm.line_end_pullback(limit))

    @tool
    def corner_rounding(condition: str = "", limit: int = 25) -> str:
        """Printed corner rounding at every target corner (inset along the bisector and the
        equivalent radius), convex and concave. condition '' = nominal."""
        return _dump(lm.corner_rounding(condition, limit))

    @tool
    def edge_epe(kind: str = "", limit: int = 25) -> str:
        """EPE per drawn target edge on every condition (nominal, min, max, spread). kind: '' | side | end."""
        return _dump(lm.edge_epe(kind, limit))

    # ------------------------------------------------------------------- query
    @tool
    def features_in_region(x0: float, y0: float, x1: float, y1: float) -> str:
        """Everything inside a rectangle: polygons, line ends, assists, mask joins, gauges, findings."""
        return _dump(lm.features_in_region(x0, y0, x1, y1, findings()))

    @tool
    def nearest_features(x_nm: float, y_nm: float, k: int = 5) -> str:
        """Nearest target polygons and line ends to a point, with distances."""
        return _dump(lm.nearest_features(x_nm, y_nm, k))

    @tool
    def measure_distance(x1: float = 0.0, y1: float = 0.0, x2: float = 0.0, y2: float = 0.0,
                         poly_a: int = -1, poly_b: int = -1) -> str:
        """Distance between two points, or (poly_a, poly_b >= 0) the minimum distance between two
        target polygons with the closest points."""
        if poly_a >= 0 and poly_b >= 0:
            return _dump(lm.measure_distance(poly_a=poly_a, poly_b=poly_b))
        return _dump(lm.measure_distance(x1, y1, x2, y2))

    @tool
    def cutline_profile(x1: float, y1: float, x2: float, y2: float, conditions: str = "") -> str:
        """Along a cutline: where it crosses target, mask and printed edges (distance from the start),
        plus nominal image min/max and image log-slope at the threshold crossings.
        conditions: comma-separated names ('' = nominal)."""
        conds = [c.strip() for c in conditions.split(",") if c.strip()] or None
        return _dump(lm.cutline_profile(x1, y1, x2, y2, conds))

    return [list_capabilities, layout_structure, list_polygons, describe_polygon, list_line_ends,
            pattern_context, width_space_statistics, design_rule_check, corner_to_corner_spacing,
            pattern_catalog, find_similar_patterns, mask_target_correspondence, mask_bias_statistics,
            line_end_treatment, assist_features, mask_complexity, gauge_cd_table, process_window,
            line_end_pullback, corner_rounding, edge_epe, features_in_region, nearest_features,
            measure_distance, cutline_profile]
