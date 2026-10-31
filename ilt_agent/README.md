# ilt_agent — LangChain + Claude agent for ILT mask / resist / target analysis

An agent that inspects one ILT clip — the target layer, the optimized mask, and the
simulated resist contours at every process condition — and explains what is wrong,
where, and why. Measuring is done by deterministic computational-geometry and
computer-vision tools; Claude (via LangChain) decides what to look at, reads overlay
images, and writes the interpretation.

```
                     ┌──────────────── LangChain agent (create_agent) ────────────────┐
 question ──► Claude (claude-opus-5-5, adaptive thinking) ──tool calls──► 11 tools ──► JSON / PNG
                     └──────────── multimodal sub-call: overlay PNG → Claude vision ──┘
 tools ──► Analyzer (cached) ──► geometry.py (CG)  +  vision.py (CV)  ──► LayoutBundle
```

## Layers

| module | what it does |
|---|---|
| `bundle.py` | `LayoutBundle`: target (bool), mask (float), printed pattern per condition (bool), optional nominal resist image, pixel size; `.npz` save/load |
| `geometry.py` (computational geometry) | raster→shapely polygons; exact signed-distance fields; **EPE** at target-edge samples with outward normals; **local width/space** from the medial axis with corner-spur pruning; **connectivity** (bridge / open / printed-but-no-target) with frame-cut components excluded; PV band; **MRC** (opening/closing, small islands); contour curvature; target-corner extraction |
| `vision.py` (computer vision) | image-log-slope (ILS) map; tile grids; robust z-scores + **IsolationForest** anomaly scoring; target-context fingerprints + **DBSCAN** pattern clustering; overlay rendering |
| `analysis.py` | `Analyzer`: global summary → printability over all conditions (merged findings with severity) → EPE/PV band → MRC → regional statistics → anomalies → hotspot pattern classes → ROI detail / render. Handles frame halo, corner rounding and ±1-pixel raster resolution explicitly |
| `tools.py` | LangChain `@tool` wrappers returning compact JSON (+ `visual_inspect_region`, which renders an ROI and asks the multimodal model) |
| `llm.py` | `ChatAnthropic(model="claude-opus-5-5", thinking={"type":"adaptive"}, output_config={"effort":...})`, server-side refusal fallback on by default (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`; `--no-fallbacks` turns it off); image messages |
| `agent.py` | system prompt with the analysis protocol (global → regional → anomalies → pattern classes → close-ups → mask rules → report, facts separated from interpretation) and `run_agent()` returning the answer and the tool trace |
| `pipeline.py` | the same analyses in a fixed order without any LLM → `report.md` + figures (offline / CI / evidence pack to check the agent against) |

## Deterministic layout-reading and geometry toolkit (`layout/`, 25 tools)

Vector polygon layers (exact pixel-edge polygonization, or the drawn polygons from
`bundle.meta["target_polygons"]`) with an STRtree index, edges decomposed per ring, and
DRC-style edge-pair measurement by ray casting. Every tool returns JSON in nm.

| group | tools | what they extract |
|---|---|---|
| structure | `layout_structure`, `list_polygons`, `describe_polygon`, `list_line_ends`, `pattern_context` | polygon classes (line / L / T / cross / pad / island / complex), dominant direction, pitch (robust to empty tracks), track index, widths/lengths, every line end with facing direction, gap and gap type, dense / semi-isolated / isolated context, multi-radius density |
| drc | `width_space_statistics`, `design_rule_check`, `corner_to_corner_spacing` | width per edge, spaces typed side-side / end-end (tip-to-tip) / end-side / notch, corner-to-corner diagonal spacing, rule violations (width, space, T2T, E2S, C2C, area, edge length) |
| patterns | `pattern_catalog`, `find_similar_patterns` | D4-invariant (rotation + mirror) fingerprints of line-end / corner contexts, number of distinct contexts, FFT search for every copy of a window |
| mask | `mask_target_correspondence`, `mask_bias_statistics`, `line_end_treatment`, `assist_features`, `mask_complexity` | mask<->target mapping, detached assists (size, width, distance), attached appendages beyond a band, mask shapes joining different targets (neck width, target gap), bias along edge normals, hammerhead extension/width ratio per line end, VSB shot proxy, vertices, curvature |
| contours | `gauge_cd_table`, `process_window`, `line_end_pullback`, `corner_rounding`, `edge_epe` | automatic width / space / tip-to-tip gauges measured on every condition (condition names parsed into dose / focus / mask bias): nominal, min, worst condition, margin to spec, dCD per 1 % dose, MEEF, focus tilt/curvature; pull-back per end and condition; corner rounding radius; EPE per drawn edge |
| query | `features_in_region`, `nearest_features`, `measure_distance`, `cutline_profile` | everything in a box, nearest polygons / ends, point or polygon-polygon distance with closest points, target / mask / printed edge crossings and image log-slope along any cutline |

Validated by `tests/test_layout_tools.py` on a scene with known geometry (pitch 60, width 30,
space 30, tip-to-tip 40, mask bias +2, hammerheads, a 6 nm assist bar, a 5 nm mask join, 5 nm
pull-back, dose/mask CD steps giving 0.8 nm per 1 % dose and MEEF 1, 8 nm corner rounding,
rotated-L pattern identity). On the ILT demo clip it reads pitch 74.51 nm, widths 30/37/44 nm,
tip-to-tip gaps 50/60/140 nm (the generator's values), 3 mask shapes joining different targets,
and the 17.6 nm worst line-end pull-back (spec 15 nm).

## Use

```bash
pip install langchain langchain-anthropic langgraph shapely scikit-image opencv-python-headless scikit-learn scipy matplotlib
export ANTHROPIC_API_KEY=...            # or `ant auth login`

python -m ilt_agent ilt_agent/demo/rand2_good.npz --out out_good                # report + agent answer
python -m ilt_agent ilt_agent/demo/rand2_good.npz --offline --out out_good      # no LLM
python -m ilt_agent bundle.npz --question "Why does the tip-to-tip gap near (372, 74) nm fail at high dose?"
```

From Python:

```python
from ilt_agent import LayoutBundle, Analyzer, Spec
from ilt_agent.agent import build_agent, run_agent

an = Analyzer(LayoutBundle.load("bundle.npz"), Spec(w_min_nm=28, s_min_nm=28), "out")
answer, trace = run_agent(build_agent(an), "Find and explain all bridge/pinch risks.")
```

Your own data: build a `LayoutBundle` from rasterized target / mask / resist contours
(any pixel size; give it in nm). `scripts/build_demo_bundles.py` shows how the ILT study
simulator is converted.

## Validation

* `tests/` (21 tests, `cd ilt_agent && pytest`): synthetic ground truth for bridge / pinch /
  printed-without-target location and size, EPE sign and magnitude, local width, MRC
  islands, anomaly localisation, bundle round-trip, rendering; the real `create_agent`
  graph driven by a scripted chat model (tool calls → tool results → answer); and the
  Claude request payload (model, adaptive thinking, effort, fallbacks, image block) built
  offline.
* Demo on the ILT study (`demo/`): the agent's checker agrees with the study's own
  cutline+topology checker on the catastrophic problems — reference recipe: 2 bridges
  (study: 2 topological bridges); fidelity-only recipe: 7 bridges (study: 6) plus dozens of
  pinch/space violations. Counting differs by design (merged locations vs. per cutline).
  The anomaly detector flagged a tile where the mask joins a tip-to-tip pair across the
  design gap through a thin neck — not printing at nominal, but leaving a 23 nm space at
  high dose; no rule is written for that pattern.

## Not tested here

No API credentials were available in the build environment, so the live Claude calls
(agent reasoning and image inspection) have not been run end to end; the wiring and the
exact request payload are tested offline. Width/space are measured on binary rasters
(±1 pixel); bundles with finer pixels or contour polygons give finer numbers.
