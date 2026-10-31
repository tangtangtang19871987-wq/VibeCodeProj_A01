"""Ground-truth tests for the layout-reading / geometry toolkit (pixel = 1 nm)."""
import os
import sys

import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.morphology import disk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ilt_agent import LayoutBundle, Spec                       # noqa: E402
from ilt_agent.layout import LayoutModel                       # noqa: E402
from ilt_agent.layout import patterns as pt                    # noqa: E402

N = 300


def target():
    t = np.zeros((N, N), bool)
    t[20:280, 30:60] = True                 # track 0, x centre 45
    t[20:130, 90:120] = True                # track 1 lower  (end at y=130)
    t[170:280, 90:120] = True               # track 1 upper  (end at y=170) -> T2T gap 40 at (105, 150)
    t[20:280, 150:180] = True               # track 2, x centre 165
    t[100:200, 210:240] = True              # track 3 short piece, x centre 225
    return t


def grow(b, n):
    return ndi.binary_dilation(b, iterations=n) if n > 0 else (ndi.binary_erosion(b, iterations=-n) if n < 0 else b.copy())


def scene():
    t = target()
    m = grow(t, 2).astype(float)                                 # +2 nm bias everywhere
    m[126:136, 84:126] = 1; m[164:174, 84:126] = 1               # hammerheads on the T2T ends
    m[30:270, 268:274] = 1                                       # detached assist bar (6 nm)
    m[148:153, 180:210] = 1                                      # mask joins track 2 and track 3
    nominal = t.copy()
    nominal[125:130, 90:120] = False                             # 5 nm pull-back at the lower T2T end
    printed = {"d1.00_f+0_m+0": nominal, "d1.05_f+0_m+0": grow(t, 2), "d0.95_f+0_m+0": grow(t, -2),
               "d1.00_f+0_m+1": grow(t, 1), "d1.00_f+0_m-1": grow(t, -1)}
    return LayoutBundle(t, m, printed, 1.0, "d1.00_f+0_m+0")


@pytest.fixture(scope="module")
def lm():
    return LayoutModel(scene(), Spec(w_min_nm=25, s_min_nm=25))


def test_structure(lm):
    s = lm.layout_structure()
    assert s["orientation"] == "V" and abs(s["pitch_nm"] - 60) < 1
    assert s["n_polygons"] == 5 and s["classes"].get("line") == 5
    assert s["line_width_nm"]["median"] == pytest.approx(30, abs=0.5)
    assert set(s["tracks_used"]) == {0, 1, 2, 3} or len(s["tracks_used"]) == 4


def test_widths_spaces_and_t2t(lm):
    w = lm.width_space_statistics()
    assert w["width_side_edges"]["median"] == pytest.approx(30, abs=0.5)
    assert w["space_side-side"]["min"] == pytest.approx(30, abs=0.5)
    ends = lm.list_line_ends(gap_kind="end-end")["line_ends"]
    assert len(ends) == 2 and all(abs(e["gap_nm"] - 40) < 0.5 for e in ends)
    assert {e["facing"] for e in ends} == {"+y", "-y"}


def test_drc(lm):
    r = lm.design_rule_check(min_space=35, min_t2t=45)
    assert r["counts"].get("space:side-side", 0) >= 1 and r["counts"].get("space:end-end", 0) == 1


def test_mask_tools(lm):
    b = lm.mask_bias_statistics()
    assert b["bias_nm"]["median"] == pytest.approx(2, abs=0.6)
    a = lm.assist_features()
    assert a["detached_assists"]["count"] == 1
    assert a["detached_assists"]["items"][0]["width_nm"] == pytest.approx(6, abs=1.5)
    joins = a["mask_shapes_joining_targets"]["items"]
    assert len(joins) == 1 and 180 <= joins[0]["x_nm"] <= 210 and joins[0]["neck_width_nm"] == pytest.approx(5, abs=1.5)
    le = {(e["x_nm"], e["y_nm"]): e for e in lm.line_end_treatment()["line_ends"]}
    t2t_lower = le[(105.0, 130.0)]
    assert t2t_lower["mask_extension_nm"] >= 5 and t2t_lower["hammerhead_ratio"] > 1.3
    assert lm.mask_complexity()["rect_decomposition_count"] > 5


def test_gauges_process_window(lm):
    g = lm.gauge_cd_table(kind="width", worst_first=False)
    row = g["gauges"][0]
    assert row["nominal_cd"] == pytest.approx(30, abs=0.8)
    assert row["dCD_per_1pct_dose_nm"] == pytest.approx(0.8, abs=0.15)      # 26 -> 30 -> 34 over 0.95..1.05
    assert row["MEEF"] == pytest.approx(1.0, abs=0.15)
    pw = lm.process_window(row["id"])
    assert len(pw["table"]) == 5


def test_pullback(lm):
    rows = {(r["x_nm"], r["y_nm"]): r for r in lm.line_end_pullback()["line_ends"]}
    r = rows[(105.0, 130.0)]
    assert r["max_pullback_nm"] == pytest.approx(5, abs=1.0) and r["worst_condition"] in ("d1.00_f+0_m+0", "d0.95_f+0_m+0")


def test_corner_rounding():
    t = target()
    rounded = ndi.binary_opening(t, structure=disk(8))           # corners rounded, radius 8
    b = LayoutBundle(t, t.astype(float), {"nominal": rounded}, 1.0)
    r = LayoutModel(b).corner_rounding()
    assert r["convex_radius_nm"]["median"] == pytest.approx(8, abs=2.5)


def test_edge_epe(lm):
    rows = lm.edge_epe(kind="side")["edges"]
    big = [r for r in rows if r["pv_band_nm"] > 0]
    assert big and max(r["epe_max_nm"] for r in big) == pytest.approx(2, abs=0.8)


def test_patterns_rotation_invariant():
    img = np.zeros((200, 200), bool)
    img[30:60, 30:40] = True; img[50:60, 30:60] = True          # an L
    L = img[25:65, 25:65]
    img[120:160, 120:160] = np.rot90(L)                          # the same L rotated
    h1, _ = pt.canonical(pt.window(img, 45, 45, 20))
    h2, _ = pt.canonical(pt.window(img, 140, 140, 20))
    assert h1 == h2
    hits = pt.search(img, pt.window(img, 45, 45, 20), 0.0)
    assert len(hits) == 2


def test_catalog_and_similar(lm):
    c = lm.pattern_catalog("line_end", radius_nm=25)
    assert c["n_anchors"] == len(lm.ends) and c["n_unique_patterns"] <= c["n_anchors"]
    s = lm.find_similar_patterns(45, 150, radius_nm=20)          # middle of a long line
    assert s["n_matches"] >= 3


def test_queries(lm):
    q = lm.features_in_region(80, 120, 130, 180)
    assert len(q["line_ends"]) == 2
    d = lm.measure_distance(poly_a=0, poly_b=3)
    assert d["distance_nm"] == pytest.approx(90, abs=0.5)
    c = lm.cutline_profile(20, 150, 130, 150)
    assert c["target_edges_at_nm"][:2] == pytest.approx([10, 40], abs=1.0)
    n = lm.nearest_features(105, 150, k=2)
    assert n["line_ends"][0]["distance_nm"] == pytest.approx(20, abs=0.5)
    ctx = lm.pattern_context(105, 60)
    assert ctx["context"] == "dense"
