"""CLI:  python -m ilt_agent BUNDLE.npz [--question "..."] [--offline] [--out DIR]"""
import argparse
import json

from .analysis import Analyzer, Spec
from .bundle import LayoutBundle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle")
    ap.add_argument("--out", default="ilt_agent_out")
    ap.add_argument("--offline", action="store_true", help="deterministic pipeline only, no LLM")
    ap.add_argument("--question", default="Analyse this ILT result: find and explain all bridge/pinch "
                                          "risks and anything unusual, then recommend recipe changes.")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--no-fallbacks", action="store_true")
    a = ap.parse_args()
    b = LayoutBundle.load(a.bundle)
    spec = Spec(**{k: v for k, v in b.meta.get("spec", {}).items() if k in Spec.__dataclass_fields__})
    an = Analyzer(b, spec, a.out)
    from .pipeline import offline_report
    print("offline report:", offline_report(an))
    if a.offline:
        return
    from .agent import build_agent, run_agent
    from .llm import make_chat_model
    agent = build_agent(an, model=make_chat_model(effort=a.effort, fallbacks=not a.no_fallbacks), use_vision=True)
    final, trace = run_agent(agent, a.question)
    with open(f"{a.out}/agent_answer.md", "w") as f:
        f.write(final)
    with open(f"{a.out}/agent_trace.json", "w") as f:
        json.dump(trace, f, indent=1)
    print(final)


if __name__ == "__main__":
    main()
