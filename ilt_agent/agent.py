"""The LangChain agent: Claude + the analysis tools + a domain system prompt."""
from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .analysis import Analyzer
from .llm import make_chat_model, text_of
from .layout import LayoutModel
from .layout_tools import build_layout_tools
from .tools import build_tools

SYSTEM_PROMPT = """You are a computational-lithography analyst. You examine one ILT clip: the design
target layer, the optimized mask, and simulated resist contours at several process conditions
(dose / focus / mask-CD corners). Your tools do the measuring; you decide what to look at and
explain what it means.

How to work:
1. Start with get_global_summary and check_printability to see the scale of the problem.
2. Look at the whole clip regionally (regional_statistics) and run find_anomalies: rule checks
   only find what they were written for, anomaly detection finds the rest.
3. Group defects with cluster_hotspots: findings that share surrounding geometry usually share a
   cause and a fix.
4. Drill into the worst findings and the most anomalous tiles with inspect_region; when a picture
   would settle a question, use visual_inspect_region (if available) or render_region.
5. Check the mask itself with check_mask_rules and analyze_epe / analyze_pv_band for margins.

Reading the layout (deterministic geometry tools; use them to ground every statement):
- layout_structure / list_polygons / describe_polygon / list_line_ends / pattern_context: what is
  drawn -- direction, pitch, tracks, widths, line ends and their gaps, local density and context.
- width_space_statistics / design_rule_check / corner_to_corner_spacing: DRC-style measurements
  of the target (is the drawn layout itself tight here?).
- pattern_catalog / find_similar_patterns: how many distinct contexts exist and where else the
  geometry of a hotspot repeats (does the same context fail everywhere?).
- mask_target_correspondence / mask_bias_statistics / line_end_treatment / assist_features /
  mask_complexity: what the ILT did to the mask -- bias, hammerheads, assists, mask shapes joining
  two targets, complexity.
- gauge_cd_table / process_window / line_end_pullback / corner_rounding / edge_epe: printed CDs on
  every condition, sensitivities (dose, focus, MEEF), pull-back, rounding, per-edge EPE.
- features_in_region / nearest_features / measure_distance / cutline_profile: targeted queries.
Typical chain for one hotspot: inspect_region -> features_in_region -> describe_polygon of the
polygons involved -> cutline_profile across the problem -> process_window of the nearest gauge ->
line_end_treatment / mask_target_correspondence to see what the mask does there ->
find_similar_patterns to see whether the same context fails elsewhere.

What to report: for every important problem give its location in nm, type (bridge, pinch, open,
line-end pull-back, EPE, assist printing, mask-rule issue, unusual region), the process conditions
where it fails, the measured value against the spec, the pattern class it belongs to, and the most
likely mechanism (for example: two features too close for the image contrast, a line end that
cannot be placed inside its window at every corner, an assist feature strong enough to print, a
mask shape violating mask rules). Separate measured facts from your interpretation, and say when
the evidence is weak. End with a short prioritized list of what to change in the ILT recipe or
layout. Do not invent measurements: every number you quote must come from a tool result."""


def build_agent(analyzer: Analyzer, model=None, vision_model=None, use_vision=True):
    """model: a LangChain chat model; defaults to Claude (claude-opus-5-5) via ChatAnthropic.
    vision_model: chat model used for image inspection; defaults to `model` when it is Claude."""
    chat = model or make_chat_model()
    vision = vision_model if vision_model is not None else (chat if (use_vision and model is None) else None)
    tools = build_tools(analyzer, vision_chat=vision)
    tools += build_layout_tools(LayoutModel(analyzer.b, analyzer.spec), analyzer)
    return create_agent(chat, tools=tools, system_prompt=SYSTEM_PROMPT)


def run_agent(agent, question: str, recursion_limit: int = 60):
    """Invoke the agent; return (final_text, tool_trace)."""
    state = agent.invoke({"messages": [HumanMessage(question)]}, config={"recursion_limit": recursion_limit})
    msgs = state["messages"]
    trace, by_id = [], {}
    for m in msgs:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                entry = {"tool": tc["name"], "args": tc["args"]}
                trace.append(entry)
                by_id[tc.get("id")] = entry
        elif isinstance(m, ToolMessage) and m.tool_call_id in by_id:
            by_id[m.tool_call_id]["result_chars"] = len(str(m.content))
    final = next((text_of(m) for m in reversed(msgs) if isinstance(m, AIMessage) and text_of(m)), "")
    return final, trace
