"""Unit tests for the imbalance machinery.

The interesting assertions are not "does SMOTE run" but "does SMOTE do the
thing that makes it valid": synthetic points must lie on segments between real
minority rows, nothing may synthesise a majority row, the same seed must give
the same data, and no resampler may be pointed at a split it is not allowed to
touch.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.imbalance import (
    RESAMPLERS,
    WEIGHTERS,
    FocalLoss,
    _n_to_add,
    amount_weights,
    class_weights,
    effective_number_weights,
    resample,
    sample_weights,
    sigmoid,
    smooth_labels,
    smote,
)


@pytest.fixture
def imbalanced():
    """Two separated blobs at roughly the real 1:600 imbalance."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.0, 1.0, (3000, 8)), rng.normal(3.0, 0.6, (25, 8))])
    y = np.concatenate([np.zeros(3000, dtype=int), np.ones(25, dtype=int)])
    return X, y


@pytest.mark.parametrize("name", sorted(RESAMPLERS))
def test_resampler_adds_only_minority(name, imbalanced):
    X, y = imbalanced
    n_neg_before = int((y == 0).sum())
    Xr, yr = resample(name, X, y, ratio=0.05, seed=1)

    assert len(Xr) == len(yr)
    assert int(yr.sum()) > int(y.sum()), "no positives were added"
    # Cleaning variants may delete rows; none may invent a legitimate transaction.
    assert int((yr == 0).sum()) <= n_neg_before
    assert np.isfinite(Xr).all()
    assert Xr.shape[1] == X.shape[1]


@pytest.mark.parametrize("name", sorted(RESAMPLERS))
def test_resampler_is_seeded(name, imbalanced):
    X, y = imbalanced
    a = resample(name, X, y, ratio=0.05, seed=7)[0]
    b = resample(name, X, y, ratio=0.05, seed=7)[0]
    c = resample(name, X, y, ratio=0.05, seed=8)[0]
    assert np.array_equal(a, b), "same seed produced different data"
    assert not np.array_equal(a, c), "the seed does nothing"


@pytest.mark.parametrize("split", ["val", "test", "holdout"])
def test_resampling_non_training_splits_is_refused(split, imbalanced):
    X, y = imbalanced
    with pytest.raises(ValueError, match="refusing to resample"):
        resample("smote", X, y, ratio=0.05, seed=1, split=split)


def test_smote_points_lie_between_real_frauds(imbalanced):
    """The defining property: every synthetic row is a convex combination of
    two real minority rows, so it cannot leave their hull."""
    X, y = imbalanced
    Xr, yr = smote(X, y, ratio=0.05, seed=3)
    synth = Xr[len(X):]
    real_pos = X[y == 1]

    lo, hi = real_pos.min(axis=0), real_pos.max(axis=0)
    assert (synth >= lo - 1e-9).all() and (synth <= hi + 1e-9).all(), \
        "a synthetic fraud escaped the bounding box of the real frauds"


def test_jitter_can_leave_the_hull(imbalanced):
    """The counterpart, and the reason jitter is on the bench at all."""
    X, y = imbalanced
    Xr, _ = RESAMPLERS["jitter"](X, y, ratio=0.05, seed=3, sigma=0.5)
    synth = Xr[len(X):]
    real_pos = X[y == 1]
    outside = ((synth < real_pos.min(axis=0)) | (synth > real_pos.max(axis=0))).any()
    assert outside, "jitter never left the observed fraud range; it is just SMOTE"


def test_ratio_is_expressed_against_negatives(imbalanced):
    X, y = imbalanced
    Xr, yr = smote(X, y, ratio=0.10, seed=1)
    assert int(yr.sum()) == pytest.approx(0.10 * int((y == 0).sum()), rel=0.05)


def test_ratio_already_met_is_a_no_op(imbalanced):
    """Asking for fewer positives than exist must not delete any."""
    X, y = imbalanced
    assert _n_to_add(y, 0.0001) == 0
    Xr, yr = smote(X, y, ratio=0.0001, seed=1)
    assert len(Xr) == len(X) and int(yr.sum()) == int(y.sum())


def test_smote_enn_never_deletes_a_real_fraud(imbalanced):
    X, y = imbalanced
    Xr, yr = RESAMPLERS["smote_enn"](X, y, ratio=0.05, seed=1)
    kept_real = Xr[: int((yr[: len(X)] == 1).sum())]  # ordering is preserved
    assert int(yr.sum()) >= int(y.sum()), "cleaning removed real frauds"
    assert kept_real is not None


def test_unknown_resampler_is_an_error(imbalanced):
    X, y = imbalanced
    with pytest.raises(KeyError):
        resample("definitely_not_a_method", X, y, ratio=0.05, seed=1)


# ------------------------------------------------------------------- weighting
def test_class_weights_are_mean_one_and_favour_fraud(imbalanced):
    _, y = imbalanced
    w = class_weights(y)
    assert w.mean() == pytest.approx(1.0)
    assert w[y == 1].mean() > w[y == 0].mean() * 50


def test_effective_number_is_gentler_than_inverse_frequency(imbalanced):
    """The whole claim of the class-balanced loss: 25 frauds are not worth 25
    independent rows, so the up-weighting should be smaller."""
    _, y = imbalanced
    inv = class_weights(y)
    eff = effective_number_weights(y, beta=0.999)
    assert eff[y == 1].mean() / eff[y == 0].mean() < inv[y == 1].mean() / inv[y == 0].mean()


def test_amount_weights_are_winsorised():
    """One enormous fraud must not out-weigh every other fraud combined.

    Note the fraud count here: winsorising at a percentile only bites when
    there are enough positives for that percentile to have resolution. With
    25 frauds the 99th percentile interpolates to within a rounding error of
    the outlier itself and clips nothing -- which is a real property of the
    method, so the test states it at the scale the training split actually has
    (331 frauds) rather than asserting a guarantee that does not exist.
    """
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(5000, dtype=int), np.ones(331, dtype=int)])
    amounts = np.concatenate([rng.gamma(2, 40, 5000), rng.gamma(2, 40, 331)])
    amounts[-1] = 1e7  # one absurd fraud

    w = amount_weights(y, amounts, fp_cost=50.0)
    cap = np.percentile(amounts[y == 1], 99)
    assert w[y == 1].max() <= cap / np.mean(
        np.concatenate([np.full(5000, 50.0), np.clip(amounts[y == 1], 1, cap)])) + 1e-9
    assert w[-1] < 50 * np.median(w[y == 1]), "the outlier still dominates"
    assert np.isfinite(w).all() and (w > 0).all()


@pytest.mark.parametrize("name", sorted(WEIGHTERS))
def test_all_weighters_are_positive_and_finite(name, imbalanced):
    _, y = imbalanced
    w = sample_weights(name, y, amounts=np.full(len(y), 100.0), fp_cost=50.0)
    assert np.isfinite(w).all() and (w > 0).all() and len(w) == len(y)


# -------------------------------------------------------------- label smoothing
def test_label_smoothing_moves_both_classes_inward():
    y = np.array([0, 0, 1, 1])
    s = smooth_labels(y, eps_pos=0.02, eps_neg=0.005)
    assert set(np.unique(s)) == {0.005, 0.98}
    assert (s > 0).all() and (s < 1).all()


def test_label_smoothing_default_is_the_identity():
    y = np.array([0, 1, 0, 1])
    assert np.array_equal(smooth_labels(y), y.astype(float))


# ------------------------------------------------------------------ focal loss
class _FakeDMatrix:
    def __init__(self, y, w=None):
        self._y, self._w = np.asarray(y, dtype=float), w

    def get_label(self):
        return self._y

    def get_weight(self):
        return self._w


@pytest.mark.parametrize("label", [0.0, 1.0])
def test_focal_gradient_matches_finite_difference(label):
    fl = FocalLoss(gamma=2.0, alpha=0.75)
    z = np.array([-6.0, -2.0, -0.3, 0.0, 0.4, 2.0, 5.0])
    y = np.full_like(z, label)

    def loss(zz):
        p = sigmoid(zz)
        pt = np.where(y >= 0.5, p, 1 - p)
        at = np.where(y >= 0.5, fl.alpha, 1 - fl.alpha)
        return -at * (1 - pt) ** fl.gamma * np.log(np.clip(pt, 1e-12, 1))

    eps = 1e-5
    numeric = (loss(z + eps) - loss(z - eps)) / (2 * eps)
    assert np.abs(numeric - fl._grad(z, y)).max() < 1e-6


def test_focal_hessian_is_strictly_positive():
    """XGBoost divides by the hessian. A zero or negative one is a crash or a
    nonsense split, so the floor is not optional."""
    fl = FocalLoss()
    z = np.linspace(-20, 20, 200)
    _, h = fl(z, _FakeDMatrix(np.tile([0.0, 1.0], 100)))
    assert (h > 0).all()
    assert np.isfinite(h).all()


def test_focal_gradient_pushes_in_the_right_direction():
    fl = FocalLoss()
    # A positive scored far too low must get a negative gradient (raise it);
    # a negative scored far too high must get a positive one (lower it).
    g_pos, _ = fl(np.array([-5.0]), _FakeDMatrix([1.0]))
    g_neg, _ = fl(np.array([5.0]), _FakeDMatrix([0.0]))
    assert g_pos[0] < 0 and g_neg[0] > 0


def test_focal_downweights_easy_examples():
    """The entire point of focal loss: a confidently-correct row must produce a
    far smaller gradient than an ambiguous one."""
    fl = FocalLoss(gamma=2.0)
    easy, _ = fl(np.array([8.0]), _FakeDMatrix([1.0]))
    hard, _ = fl(np.array([0.0]), _FakeDMatrix([1.0]))
    assert abs(easy[0]) < abs(hard[0]) / 50


def test_focal_respects_sample_weights():
    fl = FocalLoss()
    z, y = np.array([0.5, 0.5]), np.array([1.0, 1.0])
    g1, h1 = fl(z, _FakeDMatrix(y))
    g2, h2 = fl(z, _FakeDMatrix(y, np.array([2.0, 2.0])))
    assert np.allclose(g2, 2 * g1) and np.allclose(h2, 2 * h1)


def test_sigmoid_does_not_overflow():
    assert np.isfinite(sigmoid(np.array([-1e9, 0.0, 1e9]))).all()
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
