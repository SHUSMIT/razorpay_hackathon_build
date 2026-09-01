"""Probability calibration for the served score.

Why this matters here specifically: the cost model multiplies the score by
money. Each model is trained with a heavy positive class weight (~1:613), so
its raw output is a probability for a reweighted world, not for the merchant's.
Ranking survives that; the numbers do not. A "0.4" that actually means 4% fraud
would put the cost-optimal threshold in the wrong place.

Isotonic regression is used rather than Platt scaling because the raw scores
here are not sigmoid-shaped -- they are bimodal, with a heavy pile near zero.
Isotonic is monotonic, so it cannot change the ranking (PR-AUC is unchanged by
construction); it only re-maps the numbers onto observed frequencies.

FITTED ON A SLICE OF VALIDATION THE THRESHOLD IS NOT SELECTED ON. Validation is
split by time: the earlier part fits the blend weights and this calibrator, the
later part selects the operating threshold. Reusing one slice for both would
let the threshold be tuned to the calibrator's own fitting noise.
"""
from __future__ import annotations

import json

import numpy as np

from src.config import MODELS

CALIBRATOR_FILE = MODELS / "calibrator.json"


class Calibration:
    """Monotone piecewise-linear map from raw score to calibrated probability."""

    def __init__(self, x: list[float] | None = None, y: list[float] | None = None,
                 method: str = "none"):
        self.x = list(x or [])
        self.y = list(y or [])
        self.method = method

    @property
    def fitted(self) -> bool:
        return self.method != "none" and len(self.x) >= 2

    def predict(self, scores) -> np.ndarray:
        s = np.asarray(scores, dtype="float64")
        if not self.fitted:
            return s
        return np.interp(s, np.asarray(self.x), np.asarray(self.y))

    # ------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, scores, labels, max_knots: int = 512) -> "Calibration":
        from sklearn.isotonic import IsotonicRegression

        s = np.asarray(scores, dtype="float64")
        y = np.asarray(labels, dtype="float64")
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(s, y)

        # Store a thinned curve rather than the full step function: at 1.3M
        # points the JSON would be tens of MB and add nothing.
        grid = np.unique(np.quantile(s, np.linspace(0, 1, max_knots)))
        mapped = iso.predict(grid)
        return cls(grid.tolist(), mapped.tolist(), "isotonic")

    # --------------------------------------------------------- persistence
    def save(self) -> None:
        CALIBRATOR_FILE.write_text(
            json.dumps({"method": self.method, "x": self.x, "y": self.y}),
            encoding="utf-8")

    @classmethod
    def load(cls) -> "Calibration":
        if CALIBRATOR_FILE.exists():
            try:
                d = json.loads(CALIBRATOR_FILE.read_text(encoding="utf-8"))
                return cls(d.get("x"), d.get("y"), d.get("method", "none"))
            except (OSError, json.JSONDecodeError):
                pass
        return cls()


def expected_calibration_error(scores, labels, bins: int = 20) -> float:
    """Mean |predicted - observed| across equal-count bins. Lower is better."""
    s = np.asarray(scores, dtype="float64")
    y = np.asarray(labels, dtype="float64")
    if len(s) == 0:
        return float("nan")
    edges = np.unique(np.quantile(s, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return float("nan")
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, len(edges) - 2)
    total, err = 0, 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n:
            err += n * abs(s[m].mean() - y[m].mean())
            total += n
    return float(err / max(total, 1))
