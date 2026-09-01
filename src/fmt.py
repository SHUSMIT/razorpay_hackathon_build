"""Display formatting.

Scores are reported as percentages everywhere a human reads them. A PR-AUC of
0.0466 and a fraud rate of 0.00163 are hard to compare at a glance; 4.66% and
0.16% are not. The underlying values stay as floats in the JSON artefacts --
this module only changes how they are shown.
"""
from __future__ import annotations

import math


def pct(x, dp: int = 2, dash: str = "n/a") -> str:
    """0.0466 -> '4.66%'. Handles None and NaN without blowing up."""
    if x is None:
        return dash
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    if math.isnan(v):
        return dash
    return f"{v * 100:.{dp}f}%"


def signed_pct(x, dp: int = 2, dash: str = "n/a") -> str:
    """0.0123 -> '+1.23%'. For deltas, where the sign carries the meaning."""
    if x is None:
        return dash
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    if math.isnan(v):
        return dash
    return f"{v * 100:+.{dp}f}%"


def money(x, dp: int = 0, dash: str = "n/a") -> str:
    if x is None:
        return dash
    try:
        return f"{float(x):,.{dp}f}"
    except (TypeError, ValueError):
        return dash


def lift(x, dp: int = 1, dash: str = "n/a") -> str:
    """Multiplier against a baseline, e.g. '76.9x'."""
    if x is None:
        return dash
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    if math.isnan(v) or math.isinf(v):
        return dash
    return f"{v:,.{dp}f}x"
