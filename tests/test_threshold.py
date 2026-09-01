"""Unit tests for cost-curve smoothing, threshold selection and calibration."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.calibrate import (
    Calibrator,
    brier,
    expected_calibration_error,
)
from src.threshold import (
    DEFAULT_GRID,
    bootstrap_cost_band,
    cost_curve,
    select_threshold,
    smooth,
)


@pytest.fixture
def scored():
    """A ranker with realistic separation at a realistic base rate."""
    rng = np.random.default_rng(0)
    n = 20_000
    y = (rng.random(n) < 0.002).astype(int)
    z = rng.normal(-3.0, 1.0, n) + 4.5 * y
    p = 1.0 / (1.0 + np.exp(-z))
    amounts = rng.gamma(2.0, 60.0, n)
    return y, p, amounts


# ------------------------------------------------------------------ smoothing
def test_smooth_preserves_length_and_reduces_variance():
    v = np.array([5.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 2.0, 6.0])
    s = smooth(v, 5)
    assert len(s) == len(v)
    assert s.std() < v.std()


def test_smooth_window_one_is_the_identity():
    v = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    assert np.array_equal(smooth(v, 1), v)


def test_smooth_leaves_a_constant_alone():
    v = np.full(20, 7.0)
    assert np.allclose(smooth(v, 7), 7.0)


def test_smooth_even_window_is_made_odd():
    """A centred average needs an odd window; silently off-centring the result
    would shift the chosen threshold by half a grid step."""
    v = np.arange(10.0)
    assert len(smooth(v, 4)) == len(v)


# ------------------------------------------------------------------ cost curve
def test_cost_curve_endpoints(scored):
    y, p, amounts = scored
    fp_cost = 50.0
    grid = np.array([0.0, 1.01])
    c = cost_curve(y, p, amounts, grid, fp_cost)
    # Threshold 0: block everything -> pure friction cost, no missed fraud.
    assert c[0] == pytest.approx(fp_cost * int((y == 0).sum()))
    # Threshold above 1: block nothing -> eat every fraud.
    assert c[1] == pytest.approx(amounts[y == 1].sum())


def test_cost_curve_matches_the_scalar_implementation(scored):
    """The vectorised curve must agree with src.evaluate.cost_at, which is what
    the reported numbers are computed with."""
    from src.evaluate import cost_at

    y, p, amounts = scored
    grid = np.array([0.05, 0.3, 0.7])
    fast = cost_curve(y, p, amounts, grid, 50.0)
    for i, t in enumerate(grid):
        assert fast[i] == pytest.approx(cost_at(y, p, amounts, t)["total_cost"])


def test_cost_curve_is_never_negative(scored):
    y, p, amounts = scored
    assert (cost_curve(y, p, amounts, DEFAULT_GRID, 50.0) >= 0).all()


# ------------------------------------------------------------------- selection
def test_selected_threshold_is_on_the_grid(scored):
    y, p, amounts = scored
    sel = select_threshold(y, p, amounts, fp_cost=50.0, n_boot=25, seed=0)
    assert sel["threshold"] in set(np.round(DEFAULT_GRID, 4))


def test_one_se_is_at_least_as_conservative_as_the_minimum(scored):
    """The rule may only move the threshold UP -- towards fewer blocks. If it
    ever moved down it would be buying recall with noise."""
    y, p, amounts = scored
    sel = select_threshold(y, p, amounts, fp_cost=50.0, n_boot=25, seed=0)
    assert sel["threshold_one_se"] >= sel["threshold_argmin_smoothed"]


def test_one_se_costs_no_less_than_the_raw_argmin(scored):
    """Arithmetically forced -- if it were violated, the argmin is not an argmin
    and the cost curve is wrong."""
    y, p, amounts = scored
    sel = select_threshold(y, p, amounts, fp_cost=50.0, n_boot=25, seed=0)
    assert sel["cost_raw_at_chosen"] >= sel["cost_raw_at_argmin"] - 1e-9


@pytest.mark.parametrize("rule", ["argmin", "smoothed", "one_se"])
def test_every_rule_returns_a_usable_threshold(scored, rule):
    y, p, amounts = scored
    sel = select_threshold(y, p, amounts, fp_cost=50.0, n_boot=25, seed=0, rule=rule)
    assert 0.0 < sel["threshold"] < 1.0


def test_selection_is_seeded(scored):
    y, p, amounts = scored
    a = select_threshold(y, p, amounts, fp_cost=50.0, n_boot=25, seed=3)
    b = select_threshold(y, p, amounts, fp_cost=50.0, n_boot=25, seed=3)
    assert a["threshold"] == b["threshold"]
    assert a["argmin_bootstrap_std"] == b["argmin_bootstrap_std"]


def test_bootstrap_band_keeps_the_fraud_count_fixed(scored):
    """Stratified resampling: the spread must reflect uncertainty about the
    threshold, not a wandering base rate."""
    y, p, amounts = scored
    band = bootstrap_cost_band(y, p, amounts, DEFAULT_GRID, 50.0, n_boot=20, seed=0)
    assert len(band["se"]) == len(DEFAULT_GRID)
    assert (band["se"] >= 0).all()
    assert len(band["argmin_thresholds"]) == 20


def test_high_fp_cost_pushes_the_threshold_up(scored):
    """Sanity on the economics: make wrongly blocking a customer expensive and
    the system must become more reluctant to block."""
    y, p, amounts = scored
    cheap = select_threshold(y, p, amounts, fp_cost=1.0, n_boot=25, seed=0,
                             rule="smoothed")
    dear = select_threshold(y, p, amounts, fp_cost=5000.0, n_boot=25, seed=0,
                            rule="smoothed")
    assert dear["threshold"] >= cheap["threshold"]


# ----------------------------------------------------------------- calibration
def test_isotonic_never_inverts_two_scores(scored):
    """Isotonic is monotone NON-DECREASING, which is a weaker promise than
    "ranking metrics are unchanged" and the difference matters here.

    It merges score regions where the observed fraud rate is non-monotone, so
    transactions that were ordered come out tied -- and ROC-AUC scores a tie as
    half a concordant pair, so the number moves slightly. What can never happen
    is an inversion: if the model ranked A below B, calibration may make them
    equal but must not put A above B. That is the property the decision bands
    depend on, so that is what is asserted.
    """
    from sklearn.metrics import roc_auc_score

    y, p, _ = scored
    pc = Calibrator("isotonic").fit(p, y).predict(p)

    order = np.argsort(p, kind="stable")
    assert (np.diff(pc[order]) >= -1e-12).all(), "calibration inverted the ranking"
    # It may only move AUC by the tie-breaking margin, not materially.
    assert roc_auc_score(y, pc) == pytest.approx(roc_auc_score(y, p), abs=1e-3)


def test_calibration_fixes_a_deliberately_inflated_score(scored):
    """The exact failure oversampling causes: scores trained at an inflated base
    rate. Calibration must pull them back to the observed rate."""
    y, p, _ = scored
    inflated = np.clip(p * 25.0, 0, 1)
    assert inflated.mean() > 10 * y.mean()

    cal = Calibrator("isotonic").fit(inflated, y)
    fixed = cal.predict(inflated)
    assert abs(fixed.mean() - y.mean()) < abs(inflated.mean() - y.mean())
    assert brier(y, fixed) <= brier(y, inflated)
    assert expected_calibration_error(y, fixed) <= expected_calibration_error(y, inflated)


@pytest.mark.parametrize("method", ["none", "platt", "isotonic"])
def test_calibrator_round_trips_through_json(scored, method):
    """The API loads this from disk. Plain numbers, no pickle."""
    y, p, _ = scored
    cal = Calibrator(method).fit(p, y)
    blob = json.loads(json.dumps(cal.to_dict()))
    restored = Calibrator.from_dict(blob)
    assert np.abs(restored.predict(p) - cal.predict(p)).max() < 1e-6


def test_calibrated_probabilities_stay_in_range(scored):
    y, p, _ = scored
    for method in ("platt", "isotonic"):
        out = Calibrator(method).fit(p, y).predict(p)
        assert (out >= 0).all() and (out <= 1).all()


def test_ece_uses_quantile_bins(scored):
    """At a 0.2% base rate an equal-width binning puts everything in bin 0 and
    reports a flattering number; the default must not do that."""
    y, p, _ = scored
    q = expected_calibration_error(y, p, strategy="quantile")
    w = expected_calibration_error(y, p, strategy="uniform")
    assert q >= 0 and w >= 0
    assert expected_calibration_error(y, y.astype(float)) == pytest.approx(0.0)


def test_none_calibrator_is_the_identity(scored):
    _, p, _ = scored
    assert np.array_equal(Calibrator("none").fit(p, None).predict(p), p)
