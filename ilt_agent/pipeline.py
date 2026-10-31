"""Deterministic analysis pipeline (no LLM): the same tools in a fixed order, written
up as a Markdown report with figures.  Used offline, in CI, and as the evidence
pack the agent's conclusions can be checked against."""
from __future__ import annotations

import json
import os

from .analysis import Analyzer


def _table(rows, cols):
    if not rows:
        return "_none_\n"
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return out


def offline_report(an: Analyzer, title="ILT clip analysis", tile_nm=150.0):
    g = an.global_summary()
    pc = an.printability_check(limit=25)
    epe = an.epe_report(8)
    pvb = an.pv_band_report()
    mrc = an.mask_rule_check()
    reg = an.regional_statistics(tile_nm)
    ano = an.anomalies(tile_nm, 5)
    cl = an.hotspot_clusters()
    figs = {"overview": an.render_region(name="overview.png"),
            "pv_band": an.render_region(heat="pv_band", name="overview_pvband.png")}
    if an.b.intensity is not None:
        figs["ils"] = an.render_region(heat="ils", name="overview_ils.png")
    details = []
    worst = [f for f in an.findings() if f["level"] not in ("marginal", "info")][:4]
    for f in worst:
        d = an.region_detail(f["x_nm"], f["y_nm"], 70)
        d["png"] = an.render_region(f["x_nm"], f["y_nm"], 70, name=f"finding_{f['id']}.png")
        details.append((f, d))
    for t in ano["anomalous_tiles"][:2]:
        x, y = (t["x0_nm"] + t["x1_nm"]) / 2, (t["y0_nm"] + t["y1_nm"]) / 2
        d = an.region_detail(x, y, 80)
        d["png"] = an.render_region(x, y, 80, name=f"anomaly_{int(x)}_{int(y)}.png")
        details.append(({"id": "anomaly", "type": ", ".join(t["driving_features"][:2]), "x_nm": round(x, 1),
                         "y_nm": round(y, 1)}, d))

    rel = lambda p: os.path.relpath(p, an.outdir)
    md = [f"# {title}\n"]
    md.append(f"Extent {g['extent_nm'][0]:.0f} x {g['extent_nm'][1]:.0f} nm, pixel {g['pixel_nm']} nm, "
              f"{len(g['conditions'])} process conditions (nominal: `{g['nominal']}`).\n")
    md.append(f"Spec: {json.dumps(an.spec_dict())}\n")
    md.append("## 1. Global\n")
    md.append(_table([{"layer": "target", **g["target"]}, {"layer": "mask", **g["mask"]}],
                     ["layer", "count", "area_total_nm2", "area_min_nm2", "vertices_median", "compactness_median"]))
    md.append(f"\nDetached assist features on mask: {g['mask_detached_assist_features']}; mask area outside "
              f"target: {g['mask_area_outside_target_nm2']:.0f} nm2; printed/target area at nominal: "
              f"{g['printed_nominal_vs_target_area_ratio']:.3f}.\n")
    md.append(f"\n![overview]({rel(figs['overview'])})\n")
    md.append("## 2. Printability (all conditions)\n")
    md.append(f"Counts: `{json.dumps(pc['counts'])}`; conditions with most findings: {pc['worst_conditions']}\n\n")
    md.append(_table(pc["findings"], ["id", "type", "level", "x_nm", "y_nm", "value", "n_conditions"]))
    md.append("\n## 3. EPE and PV band\n")
    md.append(f"Nominal EPE: {epe['nominal']}; PV-band width along edges: {epe['pv_band_width']}; "
              f"PV-band area / target area: {pvb['pv_band_area_over_target_area']:.3f}.\n\n")
    md.append(_table(epe["worst_nominal_epe"], ["x_nm", "y_nm", "value_nm"]))
    md.append(f"\n![pv band]({rel(figs['pv_band'])})\n")
    if "ils" in figs:
        md.append(f"\n![ils]({rel(figs['ils'])})\n")
    md.append("## 4. Mask rules\n")
    md.append(f"Rules {mrc['rules']}: width violations {mrc['width']['count']}, space violations "
              f"{mrc['space']['count']}, small islands {mrc['small_islands']['count']}, sharp corners "
              f"(r<4 nm) {mrc['sharp_corners_radius_lt_4nm']}.\n")
    md.append("## 5. Regional view and anomalies\n")
    md.append(f"Tile {reg['tile_nm']} nm, {reg['n_tiles']} tiles. Feature medians: `{json.dumps(reg['feature_median'])}`\n\n")
    md.append(_table(ano["anomalous_tiles"], ["x0_nm", "y0_nm", "x1_nm", "y1_nm", "score", "n_findings", "driving_features"]))
    md.append("\n## 6. Hotspot pattern classes\n")
    md.append(_table([{"cluster": c["cluster"], "size": c["size"], "types": c["types"],
                       "representative": c["representative"]["id"], "members": ", ".join(c["members"])}
                      for c in cl["clusters"]], ["cluster", "size", "types", "representative", "members"]))
    md.append("\n## 7. Close-ups\n")
    for f, d in details:
        md.append(f"### {f['id']} {f['type']} at ({f['x_nm']}, {f['y_nm']}) nm\n")
        md.append(f"worst width {d['worst_width_nm']} nm, worst space {d['worst_space_nm']} nm "
                  f"(spec {d['spec']}); mask outside target {d['mask_outside_target_frac']}; "
                  f"edge ILS {d.get('edge_ils_per_nm', 'n/a')}\n\n![]({rel(d['png'])})\n")
    path = os.path.join(an.outdir, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(md))
    return path
