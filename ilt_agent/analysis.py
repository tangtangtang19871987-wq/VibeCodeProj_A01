"""Analyzer: global -> regional -> anomaly -> point analyses over a LayoutBundle.

Every public method returns a JSON-serialisable dict with coordinates in nm,
so the same results feed the LangChain tools, the offline pipeline and tests.
Heavy intermediate results are cached on the instance.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label

from . import geometry as geo
from . import vision as vis
from .bundle import LayoutBundle


@dataclass
class Spec:
    w_min_nm: float = 28.0          # printed line width below this = pinch
    s_min_nm: float = 28.0          # printed space below this = bridge risk
    margin_nm: float = 2.0          # within this of the spec = marginal
    epe_tol_nm: float = 5.0         # |EPE| above this at nominal = EPE note (level "info")
    mrc_width_nm: float = 10.0      # mask rules (wafer scale)
    mrc_space_nm: float = 10.0
    mrc_area_nm2: float = 150.0
    merge_nm: float = 12.0          # findings closer than this are merged
    border_nm: float = 30.0         # halo at the frame edge: shapes are cut there, findings ignored
    corner_nm: float = 12.0         # EPE within this of a target corner is corner rounding, not EPE


SEVERITY = {"bridge": 3, "open": 3, "sraf_print": 2, "pinch": 2, "space": 2, "epe": 1}


class Analyzer:
    def __init__(self, bundle: LayoutBundle, spec: Spec | None = None, outdir="ilt_agent_out"):
        self.b = bundle
        self.spec = spec or Spec()
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self._cache = {}

    # ------------------------------------------------------------ helpers
    def _nm(self, row, col):
        x, y = self.b.to_nm(row, col)
        return round(float(x), 1), round(float(y), 1)

    def _cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def _edge_samples(self):
        return self._cached("edge", lambda: geo.target_edge_samples(self.b.target, self.b.pixel_nm, 4.0))

    def _edge_valid(self):
        """Edge samples away from target corners and from the frame halo."""
        def compute():
            P, _ = self._edge_samples()
            C = geo.target_corners(self.b.target, self.b.pixel_nm)
            ok = np.ones(len(P), bool)
            if len(C):
                d = np.min(np.hypot(P[:, None, 0] - C[None, :, 0], P[:, None, 1] - C[None, :, 1]), axis=1)
                ok &= d * self.b.pixel_nm > self.spec.corner_nm
            ok &= self._in_core(P[:, 0], P[:, 1])
            return ok
        return self._cached("edge_ok", compute)

    def _in_core(self, rows, cols):
        h, w = self.b.shape
        m = self.spec.border_nm / self.b.pixel_nm
        rows, cols = np.asarray(rows), np.asarray(cols)
        return (rows >= m) & (cols >= m) & (rows < h - m) & (cols < w - m)

    def _epe(self, cond):
        return self._cached(("epe", cond), lambda: geo.epe(self.b.printed[cond], self.b.target,
                                                           self.b.pixel_nm, samples=self._edge_samples())[1])

    # ------------------------------------------------------------- global
    def global_summary(self):
        b, px = self.b, self.b.pixel_nm
        tp = geo.polygons(b.target, px)
        mbin = b.mask > 0.5
        mp = geo.polygons(mbin, px)
        lt = label(b.target, connectivity=1)
        lm = label(mbin, connectivity=1)
        touching = set(np.unique(lm[b.target])) - {0}
        sraf = [r for r in range(1, lm.max() + 1) if r not in touching]
        nom = b.printed[b.nominal]
        out = {"extent_nm": [round(v, 1) for v in b.extent_nm], "pixel_nm": round(px, 3),
               "conditions": b.conditions, "nominal": b.nominal,
               "target": geo.polygon_stats(tp), "mask": geo.polygon_stats(mp),
               "mask_detached_assist_features": len(sraf),
               "mask_area_outside_target_nm2": float((mbin & ~b.target).sum() * px ** 2),
               "printed_nominal_vs_target_area_ratio": float(nom.sum() / max(b.target.sum(), 1)),
               "target_density": float(b.target.mean()), "meta": b.meta}
        f = self.findings()
        out["findings"] = {k: sum(1 for x in f if x["type"] == k) for k in SEVERITY}
        out["findings"]["total"] = len(f)
        return out

    # -------------------------------------------------------- printability
    def findings(self):
        """All defect candidates over all conditions, merged by location."""
        return self._cached("findings", self._compute_findings)

    def _compute_findings(self):
        b, sp, px = self.b, self.spec, self.b.pixel_nm
        raw = []
        tdist_out = ndi.distance_transform_edt(~b.target) * px
        for cond, pr in b.printed.items():
            topo = geo.connectivity(pr, b.target)
            for d in topo["bridges"]:
                raw.append(("bridge", d["row"], d["col"], 0.0, cond))
            for d in topo["opens"]:
                raw.append(("open", d["row"], d["col"], 0.0, cond))
            for d in topo["extras"]:
                raw.append(("sraf_print", d["row"], d["col"], d["area_px"] * px * px, cond))
            for d in geo.violations_below(pr, px, sp.w_min_nm + sp.margin_nm):
                if b.target[d["row"], d["col"]]:
                    raw.append(("pinch", d["row"], d["col"], d["value_nm"], cond))
            # spaces: complement of printed, only between features (near target)
            comp = ~pr
            for d in geo.violations_below(comp, px, sp.s_min_nm + sp.margin_nm):
                if (tdist_out[d["row"], d["col"]] < 40 and not b.target[d["row"], d["col"]] and
                        geo.separates_features(pr, b.target, d["row"], d["col"], d["value_nm"] / 2 / px)):
                    raw.append(("space", d["row"], d["col"], d["value_nm"], cond))
        P, _ = self._edge_samples()
        e = self._epe(b.nominal)
        bad = (np.abs(e) > sp.epe_tol_nm) & self._edge_valid()
        if bad.any():
            img = np.zeros(b.shape, bool)
            rc = np.clip(np.round(P[bad]).astype(int), 0, np.array(b.shape) - 1)
            img[rc[:, 0], rc[:, 1]] = True
            lab = label(ndi.binary_dilation(img, iterations=2))
            for reg in range(1, lab.max() + 1):
                sel = lab[rc[:, 0], rc[:, 1]] == reg
                k = np.argmax(np.where(sel, np.abs(e[bad]), -1))
                raw.append(("epe", int(rc[k, 0]), int(rc[k, 1]), float(e[bad][k]), b.nominal))
        raw = [x for x in raw if self._in_core(x[1], x[2])]
        # merge by type + proximity
        merged = []
        rad = sp.merge_nm / px
        for t, r, c, v, cond in raw:
            for m in merged:
                if m["type"] == t and np.hypot(m["row"] - r, m["col"] - c) <= rad:
                    m["conditions"].add(cond)
                    if t in ("pinch", "space") and v < m["value"]:
                        m.update(row=r, col=c, value=v)
                    if t == "epe" and abs(v) > abs(m["value"]):
                        m.update(row=r, col=c, value=v)
                    break
            else:
                merged.append({"type": t, "row": r, "col": c, "value": v, "conditions": {cond}})
        out = []
        px = self.b.pixel_nm
        for i, m in enumerate(sorted(merged, key=lambda m: (-SEVERITY[m["type"]], m["value"]))):
            x, y = self._nm(m["row"], m["col"])
            if m["type"] in ("pinch", "space"):
                # raster measurements carry +-1 pixel uncertainty: only a value more than one
                # pixel below the spec is a violation, anything closer is marginal
                lim = sp.w_min_nm if m["type"] == "pinch" else sp.s_min_nm
                level = "violation" if m["value"] < lim - px else "marginal"
            elif m["type"] == "epe":
                level = "info"
            else:
                level = "catastrophic" if m["type"] in ("bridge", "open") else "violation"
            out.append({"id": f"F{i:03d}", "type": m["type"], "level": level, "x_nm": x, "y_nm": y,
                        "row": m["row"], "col": m["col"], "value": round(float(m["value"]), 2),
                        "n_conditions": len(m["conditions"]),
                        "conditions": sorted(m["conditions"])[:6]})
        return out

    def printability_check(self, level=None, limit=40):
        f = self.findings()
        if level:
            f = [x for x in f if x["level"] == level]
        counts = {}
        for x in f:
            counts.setdefault(x["type"], {}).setdefault(x["level"], 0)
            counts[x["type"]][x["level"]] += 1
        per_cond = {}
        for x in self.findings():
            if x["level"] == "info":
                continue
            for c in x["conditions"]:
                per_cond[c] = per_cond.get(c, 0) + 1
        worst = sorted(per_cond.items(), key=lambda kv: -kv[1])[:5]
        strip = [{k: v for k, v in x.items() if k not in ("row", "col")} for x in f[:limit]]
        return {"counts": counts, "total": len(f), "worst_conditions": worst, "findings": strip,
                "measurement_resolution_nm": round(self.b.pixel_nm, 2),
                "note": "value = local width/space in nm for pinch/space, EPE in nm for epe, area nm2 for "
                        "sraf_print. Width/space come from binary rasters: values within one pixel of the "
                        "spec are 'marginal', not violations. EPE notes are level 'info'."}

    # ---------------------------------------------------------------- EPE
    def epe_report(self, top_k=10):
        b = self.b
        P, N = self._edge_samples()
        if len(P) == 0:
            return {"error": "no target edges"}
        ok = self._edge_valid()
        Pall = P
        E_all = np.stack([self._epe(c) for c in b.conditions])     # [C, S]
        nom_all = self._epe(b.nominal)
        corner = ~ok & self._in_core(P[:, 0], P[:, 1])
        corner_stats = {"n": int(corner.sum()),
                        "nominal_min": round(float(nom_all[corner].min()), 2) if corner.any() else None,
                        "nominal_median": round(float(np.median(nom_all[corner])), 2) if corner.any() else None}
        P, E, nom = P[ok], E_all[:, ok], nom_all[ok]
        if len(P) == 0:
            return {"error": "no edge samples outside corners/halo"}
        pvb = E.max(0) - E.min(0)
        idx = np.argsort(-np.abs(nom))[:top_k]
        jdx = np.argsort(-pvb)[:top_k]

        def pt(i, val):
            x, y = self._nm(*P[i])
            return {"x_nm": x, "y_nm": y, "value_nm": round(float(val), 2)}
        per_cond = {c: {"mean": round(float(E[k].mean()), 2), "rms": round(float(np.sqrt((E[k] ** 2).mean())), 2),
                        "max_out": round(float(E[k].max()), 2), "max_in": round(float(E[k].min()), 2)}
                    for k, c in enumerate(b.conditions)}
        return {"n_edge_samples": int(len(P)), "n_edge_samples_total": int(len(Pall)),
                "nominal": {"mean": round(float(nom.mean()), 2), "rms": round(float(np.sqrt((nom ** 2).mean())), 2),
                            "p99_abs": round(float(np.percentile(np.abs(nom), 99)), 2),
                            "frac_above_tol": round(float((np.abs(nom) > self.spec.epe_tol_nm).mean()), 4)},
                "per_condition": per_cond,
                "pv_band_width": {"median": round(float(np.median(pvb)), 2), "p95": round(float(np.percentile(pvb, 95)), 2),
                                  "max": round(float(pvb.max()), 2)},
                "worst_nominal_epe": [pt(i, nom[i]) for i in idx],
                "widest_pv_band": [pt(i, pvb[i]) for i in jdx],
                "corner_samples_excluded": corner_stats,
                "sign": "EPE > 0: printed edge outside target (bloat); < 0: pull-back; samples within "
                        f"{self.spec.corner_nm} nm of a target corner and the {self.spec.border_nm} nm frame halo excluded"}

    # ------------------------------------------------------------- PV band
    def pv_band_report(self):
        band, union, inter = geo.pv_band(self.b.printed)
        px = self.b.pixel_nm
        lab = label(band)
        sizes = np.bincount(lab.ravel())[1:]
        return {"pv_band_area_nm2": float(band.sum() * px ** 2),
                "pv_band_area_over_target_area": float(band.sum() / max(self.b.target.sum(), 1)),
                "n_components": int(lab.max()),
                "largest_component_nm2": float(sizes.max() * px ** 2) if len(sizes) else 0.0}

    # ----------------------------------------------------------------- MRC
    def mask_rule_check(self, min_width_nm=None, min_space_nm=None, min_area_nm2=None, limit=20):
        sp = self.spec
        r = geo.mrc(self.b.mask > 0.5, self.b.pixel_nm, min_width_nm or sp.mrc_width_nm,
                    min_space_nm or sp.mrc_space_nm, min_area_nm2 or sp.mrc_area_nm2)
        pts, k = geo.contour_curvature(self.b.mask > 0.5, self.b.pixel_nm)
        sharp = np.abs(k) > 1 / 4.0                                       # radius < 4 nm
        out = {}
        for key, lst in r.items():
            out[key] = {"count": len(lst),
                        "items": [{**dict(zip(("x_nm", "y_nm"), self._nm(d["row"], d["col"]))),
                                   "area_nm2": round(d["area_nm2"], 1)} for d in lst[:limit]]}
        out["sharp_corners_radius_lt_4nm"] = int(sharp.sum())
        out["rules"] = {"min_width_nm": min_width_nm or sp.mrc_width_nm, "min_space_nm": min_space_nm or sp.mrc_space_nm,
                        "min_area_nm2": min_area_nm2 or sp.mrc_area_nm2}
        return out

    # ------------------------------------------------------------ regional
    FEATURES = ["target_density", "mask_density", "mask_target_xor", "epe_rms_nom", "epe_maxabs_nom",
                "pv_band_frac", "n_findings", "mask_perimeter_per_area", "min_edge_ils"]

    def tile_features(self, tile_nm=150.0):
        def compute():
            b, px = self.b, self.b.pixel_nm
            t, tiles = vis.tile_grid(b.shape, px, tile_nm)
            P, _ = self._edge_samples()
            nom = self._epe(b.nominal)
            band, _, _ = geo.pv_band(b.printed)
            mbin = b.mask > 0.5
            medge = vis.edge_band(mbin)
            ils = vis.ils_map(b.intensity, px) if b.intensity is not None else None
            tedge = vis.edge_band(b.target)
            F = np.array([[f["row"], f["col"]] for f in self.findings()]) if self.findings() else np.zeros((0, 2))
            rows = []
            for (r0, c0, r1, c1) in tiles:
                sl = (slice(r0, r1), slice(c0, c1))
                inp = (P[:, 0] >= r0) & (P[:, 0] < r1) & (P[:, 1] >= c0) & (P[:, 1] < c1)
                e = nom[inp]
                nf = int(((F[:, 0] >= r0) & (F[:, 0] < r1) & (F[:, 1] >= c0) & (F[:, 1] < c1)).sum()) if len(F) else 0
                te = tedge[sl]
                rows.append([b.target[sl].mean(), mbin[sl].mean(), (mbin[sl] ^ b.target[sl]).mean(),
                             float(np.sqrt((e ** 2).mean())) if len(e) else 0.0,
                             float(np.abs(e).max()) if len(e) else 0.0,
                             band[sl].mean(), nf,
                             medge[sl].sum() * px / max(mbin[sl].sum() * px * px, 1.0),
                             float(np.percentile(ils[sl][te], 5)) if (ils is not None and te.any()) else 0.0])
            return t, tiles, np.array(rows, float)
        return self._cached(("tiles", tile_nm), compute)

    def regional_statistics(self, tile_nm=150.0, top_k=6):
        t, tiles, X = self.tile_features(tile_nm)
        px = self.b.pixel_nm
        out = {"tile_nm": round(t * px, 1), "n_tiles": len(tiles), "features": self.FEATURES,
               "feature_median": dict(zip(self.FEATURES, np.round(np.median(X, 0), 4).tolist())),
               "feature_max": dict(zip(self.FEATURES, np.round(X.max(0), 4).tolist()))}
        order = np.argsort(-X[:, self.FEATURES.index("n_findings")])[:top_k]
        out["tiles_with_most_findings"] = [self._tile_dict(tiles[i], X[i]) for i in order if X[i, 6] > 0]
        return out

    def _tile_dict(self, tile, x, z=None):
        r0, c0, r1, c1 = tile
        px = self.b.pixel_nm
        d = {"x0_nm": round(c0 * px, 1), "y0_nm": round(r0 * px, 1), "x1_nm": round(c1 * px, 1),
             "y1_nm": round(r1 * px, 1), **{k: round(float(v), 4) for k, v in zip(self.FEATURES, x)}}
        if z is not None:
            top = np.argsort(-z)[:3]
            d["driving_features"] = [f"{self.FEATURES[j]} (|z|={z[j]:.1f})" for j in top]
        return d

    def anomalies(self, tile_nm=150.0, top_k=6):
        t, tiles, X = self.tile_features(tile_nm)
        score, Z = vis.anomaly_scores(X)
        order = np.argsort(-score)[:top_k]
        return {"method": "0.5*IsolationForest + 0.5*max robust |z| (median/MAD) over tile features",
                "tile_nm": round(t * self.b.pixel_nm, 1),
                "anomalous_tiles": [{**self._tile_dict(tiles[i], X[i], Z[i]), "score": round(float(score[i]), 3)}
                                    for i in order]}

    # ------------------------------------------------------- hotspot classes
    def hotspot_clusters(self, radius_nm=60.0, eps=0.15):
        f = [x for x in self.findings() if x["level"] not in ("marginal", "info")]
        if not f:
            return {"n_findings": 0, "clusters": []}
        rad = max(3, int(round(radius_nm / self.b.pixel_nm)))
        sigs = [vis.context_signature(self.b.target, x["row"], x["col"], rad) for x in f]
        lab = vis.cluster_signatures(sigs, eps)
        clusters = []
        for k in np.unique(lab):
            mem = [f[i] for i in np.nonzero(lab == k)[0]]
            types = {}
            for m in mem:
                types[m["type"]] = types.get(m["type"], 0) + 1
            clusters.append({"cluster": int(k), "size": len(mem), "types": types,
                             "members": [m["id"] for m in mem][:12],
                             "representative": {"id": mem[0]["id"], "x_nm": mem[0]["x_nm"], "y_nm": mem[0]["y_nm"]}})
        clusters.sort(key=lambda c: -c["size"])
        return {"n_findings": len(f), "context_radius_nm": radius_nm, "n_clusters": len(clusters),
                "clusters": clusters,
                "note": "findings with the same surrounding target geometry share a cluster -> fix once, apply to all"}

    # --------------------------------------------------------------- ROI
    def _roi(self, x_nm, y_nm, half_nm):
        px = self.b.pixel_nm
        h, w = self.b.shape
        r, c = self.b.to_px(x_nm, y_nm)
        hp = max(4, int(round(half_nm / px)))
        return max(0, r - hp), max(0, c - hp), min(h, r + hp), min(w, c + hp)

    def region_detail(self, x_nm, y_nm, half_size_nm=80.0):
        b, px, sp = self.b, self.b.pixel_nm, self.spec
        r0, c0, r1, c1 = self._roi(x_nm, y_nm, half_size_nm)
        sl = (slice(r0, r1), slice(c0, c1))
        inside = [x for x in self.findings() if r0 <= x["row"] < r1 and c0 <= x["col"] < c1]
        P, _ = self._edge_samples()
        inp = (P[:, 0] >= r0) & (P[:, 0] < r1) & (P[:, 1] >= c0) & (P[:, 1] < c1)
        cond = {}
        for name, pr in b.printed.items():
            _, _, wv = geo.local_width(pr[sl], px, prune_nm=sp.w_min_nm)
            _, _, sv = geo.local_width(~pr[sl], px, prune_nm=sp.s_min_nm)
            e = self._epe(name)[inp]
            cond[name] = {"min_width_nm": round(float(wv.min()), 1) if len(wv) else None,
                          "min_space_nm": round(float(sv.min()), 1) if len(sv) else None,
                          "epe_min": round(float(e.min()), 2) if len(e) else None,
                          "epe_max": round(float(e.max()), 2) if len(e) else None}
        worst_w = min((v["min_width_nm"] for v in cond.values() if v["min_width_nm"] is not None), default=None)
        worst_s = min((v["min_space_nm"] for v in cond.values() if v["min_space_nm"] is not None), default=None)
        out = {"roi_nm": [round(c0 * px, 1), round(r0 * px, 1), round(c1 * px, 1), round(r1 * px, 1)],
               "target_density": round(float(b.target[sl].mean()), 3),
               "mask_density": round(float((b.mask[sl] > 0.5).mean()), 3),
               "mask_outside_target_frac": round(float(((b.mask[sl] > 0.5) & ~b.target[sl]).mean()), 3),
               "worst_width_nm": worst_w, "worst_space_nm": worst_s,
               "spec": {"w_min_nm": sp.w_min_nm, "s_min_nm": sp.s_min_nm},
               "per_condition": cond,
               "findings": [{k: v for k, v in x.items() if k not in ("row", "col")} for x in inside][:15]}
        if b.intensity is not None:
            ils = vis.ils_map(b.intensity, px)[sl][vis.edge_band(b.target)[sl]]
            if len(ils):
                out["edge_ils_per_nm"] = {"p5": round(float(np.percentile(ils, 5)), 4),
                                          "median": round(float(np.median(ils)), 4)}
        return out

    def render_region(self, x_nm=None, y_nm=None, half_size_nm=None, conditions=None, heat=None, name=None):
        if x_nm is None:
            roi = None
        else:
            roi = self._roi(x_nm, y_nm, half_size_nm or 100.0)
        conds = conditions or self._extreme_conditions()
        markers = []
        col = {"bridge": "red", "open": "orange", "pinch": "magenta", "space": "cyan", "epe": "yellow",
               "sraf_print": "lime"}
        for f in self.findings():
            if f["level"] not in ("marginal", "info"):
                markers.append((f["row"], f["col"], f"{f['id']}:{f['type']}", col[f["type"]]))
        hm, hl = None, None
        if heat == "pv_band":
            hm, hl = geo.pv_band(self.b.printed)[0].astype(float), "PV band"
        elif heat == "ils" and self.b.intensity is not None:
            hm, hl = vis.ils_map(self.b.intensity, self.b.pixel_nm), "ILS (1/nm)"
        name = name or f"roi_{int(x_nm or 0)}_{int(y_nm or 0)}_{int(half_size_nm or 0)}_{heat or 'none'}.png"
        path = os.path.join(self.outdir, name)
        title = "full clip" if roi is None else f"ROI centre ({x_nm:.0f},{y_nm:.0f}) nm, half {half_size_nm:.0f} nm"
        vis.render(self.b, path, roi, conds, markers, title=title, heat=hm, heat_label=hl)
        return path

    def _extreme_conditions(self):
        """Nominal + the conditions printing the most and the least area."""
        areas = {k: v.sum() for k, v in self.b.printed.items()}
        hi = max(areas, key=areas.get)
        lo = min(areas, key=areas.get)
        return list(dict.fromkeys([self.b.nominal, hi, lo]))

    def spec_dict(self):
        return asdict(self.spec)
