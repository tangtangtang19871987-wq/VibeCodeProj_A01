"""Synthetic ground-truth tests for the geometry / vision layers and the agent wiring.

Scene (pixel 2 nm, 160 x 200 px): two vertical target lines 36 nm wide, 40 nm
apart.  Printed nominal = target with a deliberate bridge, a pinch and a stray
printed blob, so every checker has a known answer at a known place.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ilt_agent import Analyzer, LayoutBundle, Spec          # noqa: E402
from ilt_agent import geometry as geo                       # noqa: E402

PX = 2.0
H, W = 160, 200
L1 = slice(60, 78)          # columns of line 1 (18 px = 36 nm)
L2 = slice(98, 116)         # line 2, 20 px (40 nm) to the right


def scene():
    t = np.zeros((H, W), bool)
    t[10:150, L1] = True
    t[10:150, L2] = True
    p = t.copy()
    p[118:124, 78:98] = True                     # bridge at rows 118-123 (y ~ 242 nm)
    p[40:46, 60:65] = False                      # pinch line 1: 18-5-5 = 8 px left (16 nm) at y ~ 86 nm
    p[40:46, 73:78] = False
    p[20:26, 170:176] = True                     # stray printed blob (no target)
    m = t.astype(np.float32)
    m[100:102, 150:152] = 1.0                    # tiny mask island (MRC small-area)
    return t, m, p


@pytest.fixture
def an(tmp_path):
    t, m, p = scene()
    b = LayoutBundle(t, m, {"nominal": p, "lowdose": t.copy()}, PX)
    return Analyzer(b, Spec(w_min_nm=28, s_min_nm=28, margin_nm=0.0), str(tmp_path))


def near(f, x, y, tol=12.0):
    return abs(f["x_nm"] - x) <= tol and abs(f["y_nm"] - y) <= tol


def test_connectivity_bridge_and_extra():
    t, m, p = scene()
    c = geo.connectivity(p, t)
    assert len(c["bridges"]) == 1 and len(c["extras"]) == 1
    br = c["bridges"][0]
    assert 118 <= br["row"] <= 123 and 78 <= br["col"] < 98


def test_findings_locate_bridge_pinch_and_sraf(an):
    f = an.findings()
    types = {x["type"] for x in f}
    assert {"bridge", "pinch", "sraf_print"} <= types
    bridge = [x for x in f if x["type"] == "bridge"][0]
    assert near(bridge, x=88 * PX, y=121 * PX, tol=25)
    pinch = [x for x in f if x["type"] == "pinch"][0]
    assert near(pinch, x=69 * PX, y=43 * PX, tol=15)
    assert 13 <= pinch["value"] <= 19                          # 8 px = 16 nm wide neck


def test_epe_sign_and_magnitude():
    t, _, _ = scene()
    from scipy import ndimage as ndi
    grown = ndi.binary_dilation(t, iterations=2)                # +2 px = +4 nm outward
    shrunk = ndi.binary_erosion(t, iterations=2)
    _, e_out = geo.epe(grown, t, PX)
    _, e_in = geo.epe(shrunk, t, PX)
    assert 2.5 < np.median(e_out) < 5.5
    assert -5.5 < np.median(e_in) < -2.5


def test_local_width_of_line():
    t, _, _ = scene()
    _, _, w = geo.local_width(t, PX)
    assert abs(np.median(w) - 36) < 3


def test_mrc_small_island(an):
    r = an.mask_rule_check(min_area_nm2=50.0)
    assert r["small_islands"]["count"] >= 1


def test_anomaly_finds_injected_tile(tmp_path):
    rng = np.random.default_rng(0)
    t = np.zeros((240, 240), bool)
    for c0 in range(10, 240, 40):
        t[:, c0:c0 + 18] = True                                 # regular grating
    p = t.copy()
    p[160:200, 160:200] = ~p[160:200, 160:200]                  # one corrupted tile region
    b = LayoutBundle(t, t.astype(float), {"nominal": p}, PX)
    an = Analyzer(b, Spec(), str(tmp_path))
    top = an.anomalies(tile_nm=80, top_k=3)["anomalous_tiles"][0]
    assert top["x0_nm"] <= 180 * PX <= top["x1_nm"] and top["y0_nm"] <= 180 * PX <= top["y1_nm"]


def test_bundle_roundtrip(tmp_path):
    t, m, p = scene()
    b = LayoutBundle(t, m, {"nominal": p, "x": t}, PX, intensity=np.ones((H, W)), meta={"a": 1})
    path = tmp_path / "b.npz"
    b.save(path)
    b2 = LayoutBundle.load(path)
    assert b2.conditions == ["nominal", "x"] and b2.meta == {"a": 1}
    assert (b2.printed["nominal"] == p).all() and b2.intensity is not None


def test_render_and_region_detail(an):
    d = an.region_detail(88 * PX, 121 * PX, 40)
    assert any(f["type"] == "bridge" for f in d["findings"])
    png = an.render_region(88 * PX, 121 * PX, 40)
    assert os.path.getsize(png) > 1000


def test_agent_wiring_with_fake_model(an):
    """The real agent graph with a scripted chat model: tool call -> tool result -> answer."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from ilt_agent.agent import build_agent, run_agent

    class ScriptedModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    script = iter([
        AIMessage(content="", tool_calls=[{"name": "get_global_summary", "args": {}, "id": "c1", "type": "tool_call"},
                                          {"name": "check_printability", "args": {"level": "catastrophic"},
                                           "id": "c2", "type": "tool_call"}]),
        AIMessage(content="", tool_calls=[{"name": "inspect_region", "args": {"x_nm": 176, "y_nm": 242,
                                                                               "half_size_nm": 40},
                                           "id": "c3", "type": "tool_call"}]),
        AIMessage(content="One bridge between the two lines near (176, 242) nm."),
    ])
    agent = build_agent(an, model=ScriptedModel(messages=script), use_vision=False)
    final, trace = run_agent(agent, "find bridges")
    assert "bridge" in final
    assert [t["tool"] for t in trace] == ["get_global_summary", "check_printability", "inspect_region"]
    assert all(t.get("result_chars", 0) > 10 for t in trace)


def test_chat_model_payload_offline():
    """Claude request shape (no network): model, adaptive thinking, effort, fallbacks, image block."""
    from langchain_core.messages import HumanMessage
    from ilt_agent.llm import make_chat_model
    m = make_chat_model()
    m.anthropic_api_key = type(m.anthropic_api_key)("sk-dummy") if m.anthropic_api_key else None
    p = m._get_request_payload([HumanMessage(content=[
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
        {"type": "text", "text": "x"}])])
    assert p["model"] == "claude-opus-5-5" and p["thinking"] == {"type": "adaptive"}
    assert p["output_config"]["effort"] == "high" and p["fallbacks"] == "default"
    assert p["messages"][0]["content"][0]["type"] == "image"
