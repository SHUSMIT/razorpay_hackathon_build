"""The ensemble weight search.

These exist because the original implementation was broken in the worst
possible way: it ran, printed a weight vector, and reported a result -- while
having evaluated exactly ONE candidate. A recursion bug meant it could only
ever emit the corner (0, ..., 0, 1), so "weights chosen by search" was false
and the ensemble was silently just whichever model happened to be last.

Nothing failed. The output looked plausible. That is the class of bug worth
spending tests on.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.models import ensemble_predict, fit_ensemble_weights, simplex


def test_simplex_enumerates_the_whole_grid():
    """3 members at 0.1 resolution is C(12,2) = 66 compositions, not 1."""
    combos = list(simplex(3))
    assert len(combos) == 66, f"searched {len(combos)} candidates, expected 66"
    assert len(set(combos)) == 66, "duplicate candidates"


def test_every_simplex_point_is_a_valid_weight_vector():
    for w in simplex(3):
        assert abs(sum(w) - 1.0) < 1e-9, f"{w} does not sum to 1"
        assert all(x >= 0 for x in w), f"{w} has a negative weight"


def test_simplex_includes_corners_and_genuine_blends():
    combos = set(simplex(3))
    for corner in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
        assert corner in combos, f"corner {corner} missing"
    blends = [w for w in combos if all(0 < x < 1 for x in w)]
    assert len(blends) > 20, "the search space contains almost no real blends"


@pytest.mark.parametrize("k,expected", [(1, 1), (2, 11), (3, 66)])
def test_simplex_size_matches_the_combinatorics(k, expected):
    assert len(list(simplex(k))) == expected


def test_search_finds_a_model_that_is_obviously_best():
    """One member is informative, two are noise. The search must not blend
    them evenly, and must not silently return a fixed corner."""
    rng = np.random.default_rng(0)
    n = 20000
    y = (rng.random(n) < 0.02).astype(int)
    good = np.clip(y * 0.8 + rng.random(n) * 0.2, 0, 1)
    noise1 = rng.random(n)
    noise2 = rng.random(n)

    weights = fit_ensemble_weights(
        {"xgboost": good, "catboost": noise1, "histgb": noise2}, y)
    assert weights["xgboost"] >= 0.7, (
        f"search failed to favour the informative member: {weights}")
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_search_beats_or_matches_the_best_single_member_on_its_own_data():
    """A weighted blend chosen on some data cannot be worse than the best
    single member on that same data -- the corners are in the search space."""
    rng = np.random.default_rng(1)
    n = 20000
    y = (rng.random(n) < 0.03).astype(int)
    preds = {
        "xgboost": np.clip(y * 0.6 + rng.random(n) * 0.4, 0, 1),
        "catboost": np.clip(y * 0.4 + rng.random(n) * 0.6, 0, 1),
        "histgb": np.clip(y * 0.5 + rng.random(n) * 0.5, 0, 1),
    }
    weights = fit_ensemble_weights(preds, y)
    blend = ensemble_predict(preds, weights)
    best_single = max(average_precision_score(y, p) for p in preds.values())
    assert average_precision_score(y, blend) >= best_single - 1e-9


def test_ensemble_predict_ignores_zero_weighted_members():
    p = {"xgboost": np.array([0.9, 0.1]), "catboost": np.array([0.0, 0.0])}
    out = ensemble_predict(p, {"xgboost": 1.0, "catboost": 0.0})
    assert np.allclose(out, [0.9, 0.1])


def test_ensemble_predict_falls_back_to_the_mean_when_weights_are_empty():
    p = {"a": np.array([0.2, 0.4]), "b": np.array([0.4, 0.8])}
    assert np.allclose(ensemble_predict(p, {}), [0.3, 0.6])
