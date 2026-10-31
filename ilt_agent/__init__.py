"""ILT result analysis agent: computational geometry + computer vision + a LangChain/Claude agent."""
from .analysis import Analyzer, Spec
from .bundle import LayoutBundle

__all__ = ["Analyzer", "Spec", "LayoutBundle"]
