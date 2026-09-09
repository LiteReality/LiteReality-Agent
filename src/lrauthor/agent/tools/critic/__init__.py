"""critic — PRIMITIVE: grade render|photo against a goal → {pass, score, issues}."""

from .tool import CriticInvocation, CriticParams, CriticTool

__all__ = ["CriticTool", "CriticInvocation", "CriticParams"]
