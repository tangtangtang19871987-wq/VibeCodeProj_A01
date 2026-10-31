"""LangChain tools over one Analyzer.  Each returns compact JSON text.

Coordinates are nm.  Tools are cheap to call repeatedly: results are cached.
"""
from __future__ import annotations

import json

from langchain_core.tools import tool

from .analysis import Analyzer

MAX_CHARS = 6000


def _dump(obj):
    s = json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if len(s) > MAX_CHARS:
        s = s[:MAX_CHARS] + ' ... [truncated: narrow the request, e.g. smaller region or limit]'
    return s


def build_tools(an: Analyzer, vision_chat=None):
    @tool
    def get_global_summary() -> str:
        """Whole-clip overview: extent, process conditions, target/mask/printed polygon statistics,
        assist-feature count, density, and finding counts by type. Call this first."""
        return _dump(an.global_summary())

    @tool
    def check_printability(level: str = "", limit: int = 30) -> str:
        """Defect candidates over ALL process conditions, merged by location: bridge / open
        (topology changes), pinch (printed width < spec), space (printed space < spec, bridge risk),
        epe (|EPE| > tolerance at nominal), sraf_print (printed shape touching no target).
        level: '' for all, or 'catastrophic' | 'violation' | 'marginal'. Returns ids and nm coordinates."""
        return _dump(an.printability_check(level or None, limit))

    @tool
    def analyze_epe(top_k: int = 10) -> str:
        """Edge placement error along all target edges: statistics per condition, PV-band width
        (EPE spread across conditions) and the worst locations."""
        return _dump(an.epe_report(top_k))

    @tool
    def analyze_pv_band() -> str:
        """Process-variation band (union minus intersection of printed patterns over all conditions)."""
        return _dump(an.pv_band_report())

    @tool
    def check_mask_rules(min_width_nm: float = 0.0, min_space_nm: float = 0.0, min_area_nm2: float = 0.0) -> str:
        """Mask rule check on the mask itself: width/space violations (morphological opening/closing),
        small islands, sharp corners. 0 = use the default rule value."""
        return _dump(an.mask_rule_check(min_width_nm or None, min_space_nm or None, min_area_nm2 or None))

    @tool
    def regional_statistics(tile_nm: float = 150.0) -> str:
        """Tile the clip and report per-tile features (densities, mask-target deviation, EPE, PV band,
        findings, mask complexity, edge image log-slope) with medians and the busiest tiles."""
        return _dump(an.regional_statistics(tile_nm))

    @tool
    def find_anomalies(tile_nm: float = 150.0, top_k: int = 6) -> str:
        """Unsupervised anomaly detection over tiles (IsolationForest + robust z-scores). Returns the most
        unusual tiles and which features make them unusual -- useful for problems no rule checks."""
        return _dump(an.anomalies(tile_nm, top_k))

    @tool
    def cluster_hotspots(radius_nm: float = 60.0) -> str:
        """Group findings by the target geometry around them (pattern fingerprint + DBSCAN), so one
        fix per pattern class can be proposed."""
        return _dump(an.hotspot_clusters(radius_nm))

    @tool
    def inspect_region(x_nm: float, y_nm: float, half_size_nm: float = 80.0) -> str:
        """Numeric close-up of a square region: findings inside, worst printed width/space and EPE per
        condition, mask deviation from target, edge image log-slope."""
        return _dump(an.region_detail(x_nm, y_nm, half_size_nm))

    @tool
    def render_region(x_nm: float = -1.0, y_nm: float = -1.0, half_size_nm: float = 100.0, heat: str = "") -> str:
        """Save a PNG overlay (mask, target, printed contours at nominal and extreme conditions, finding
        markers). x_nm < 0 renders the whole clip. heat: '' | 'pv_band' | 'ils'. Returns the file path."""
        if x_nm < 0:
            path = an.render_region(heat=heat or None, name=f"full_{heat or 'none'}.png")
        else:
            path = an.render_region(x_nm, y_nm, half_size_nm, heat=heat or None)
        return _dump({"png": path})

    tools = [get_global_summary, check_printability, analyze_epe, analyze_pv_band, check_mask_rules,
             regional_statistics, find_anomalies, cluster_hotspots, inspect_region, render_region]

    if vision_chat is not None:
        from .llm import describe_image

        @tool
        def visual_inspect_region(x_nm: float, y_nm: float, half_size_nm: float = 100.0,
                                  question: str = "What printability problems are visible?") -> str:
            """Render a region and have the multimodal model look at it. Use it to confirm or explain
            geometric findings, or to look at anomalous tiles that no rule flagged."""
            path = an.render_region(x_nm, y_nm, half_size_nm, heat=None)
            return _dump({"png": path, "visual_assessment": describe_image(vision_chat, path, question)})
        tools.append(visual_inspect_region)
    return tools
