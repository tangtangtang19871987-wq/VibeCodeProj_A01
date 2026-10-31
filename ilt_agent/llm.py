"""Claude via LangChain (langchain-anthropic, which wraps the official anthropic SDK).

Defaults: claude-opus-5-5 with adaptive thinking and explicit effort; the
server-side refusal fallback is enabled (fallbacks="default") -- pass
fallbacks=False to turn it off.  Credentials resolve the usual way
(ANTHROPIC_API_KEY, or an `ant auth login` profile).
"""
from __future__ import annotations

import base64

from langchain_core.messages import HumanMessage, SystemMessage

MODEL = "claude-opus-5-5"


def make_chat_model(model: str = MODEL, effort: str = "high", fallbacks: bool = True, max_tokens: int = 16000):
    from langchain_anthropic import ChatAnthropic
    kw = {}
    if fallbacks:
        kw["betas"] = ["server-side-fallback-2026-07-01"]
        kw["model_kwargs"] = {"fallbacks": "default"}
    return ChatAnthropic(model=model, max_tokens=max_tokens, thinking={"type": "adaptive"},
                         output_config={"effort": effort}, max_retries=3, **kw)


VISION_SYSTEM = (
    "You inspect lithography simulation overlays. Legend: grey = mask (darker = opaque background, "
    "lighter = mask opening), green outline = design target, coloured outlines = printed resist "
    "contours at the listed process conditions, circles = numbered findings from the geometric "
    "checker. Describe only what is visible: where printed contours leave the target, where two "
    "printed shapes approach or merge (bridge risk), where a line narrows or breaks (pinch), line-end "
    "pull-back, assist shapes printing, and mask shapes that look unusual. Give coordinates from the "
    "axes. Say when the picture is ambiguous instead of guessing."
)


def text_of(msg) -> str:
    """Concatenate text blocks of an AIMessage (thinking blocks are skipped)."""
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")


def describe_image(chat, png_path: str, question: str) -> str:
    with open(png_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    msg = HumanMessage(content=[
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
        {"type": "text", "text": question},
    ])
    resp = chat.invoke([SystemMessage(VISION_SYSTEM), msg])
    stop = (resp.response_metadata or {}).get("stop_reason")
    if stop == "refusal":
        return "[vision model declined this request]"
    return text_of(resp)
