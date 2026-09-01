"""Techniques for the 579:1 class imbalance, implemented so each one can be
switched on independently and measured against the others.

Four families live here, because they attack the problem in different places:

  1. RESAMPLING      - change the training set     (oversample the minority)
  2. REWEIGHTING     - change the loss per row     (weight the minority up)
  3. LOSS SHAPING    - change the loss function    (focal loss)
  4. TARGET SHAPING  - change the labels           (label smoothing)

Everything is dependency-free (numpy + sklearn's NearestNeighbors) and seeded.
imbalanced-learn is not a dependency: the interpolation schemes below are 40
lines each, and writing them out means the amount-aware and cost-aware variants
this project actually needs are not bolted onto someone else's API.

TWO RULES ARE ENFORCED, NOT ASSUMED:

  * Resampling touches the TRAINING split only. Synthesising validation or test
    positives would invent the very frauds we claim to detect, and every metric
    downstream would be measuring our own interpolation. `resample()` refuses to
    run on anything not labelled "train".
  * Neighbour search happens in z-scored space. V1..V28 are PCA components with
    unit-ish scale, Amount is in the thousands; on raw columns every Euclidean
    neighbour is decided by Amount alone and SMOTE degenerates into
    "interpolate between two transactions of similar size".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12


# ---------------------------------------------------------------- scaling util
def _zscore_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < EPS] = 1.0
    return mu, sd


def _knn_indices(query: np.ndarray, reference: np.ndarray, k: int,
                 drop_self: bool) -> np.ndarray:
    """k nearest neighbours of every row of `query` within `reference`."""
    k_eff = min(k + (1 if drop_self else 0), len(reference))
    nn = NearestNeighbors(n_neighbors=k_eff).fit(reference)
    idx = nn.kneighbors(query, return_distance=False)
    return idx[:, 1:] if drop_self and idx.shape[1] > 1 else idx


def _interpolate(rng: np.random.Generator, base: np.ndarray, partner: np.ndarray,
                 hi: float = 1.0) -> np.ndarray:
    """A point on the segment between two real minority transactions.

    `hi < 1` keeps the synthetic point nearer the seed row, which matters for
    borderline variants where the partner may sit close to the decision surface.
    """
    lam = rng.uniform(0.0, hi, size=(len(base), 1))
    return base + lam * (partner - base)


def _n_to_add(y: np.ndarray, ratio: float) -> int:
    """How many synthetic positives to reach `ratio` = positives / negatives.

    Expressed against the negative count rather than as a multiplier, so
    `ratio=0.05` means the same thing on any split.
    """
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    return max(int(round(ratio * neg)) - pos, 0)


# ------------------------------------------------------------------ resamplers
def random_oversample(X, y, *, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Duplicate existing minority rows. The honest floor for every other method:
    it adds no information, so anything that cannot beat it is not worth its
    cost."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    n_new = _n_to_add(y, ratio)
    if n_new <= 0 or len(pos) == 0:
        return X, y
    pick = rng.choice(pos, n_new, replace=True)
    return np.vstack([X, X[pick]]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])


def smote(X, y, *, ratio: float, seed: int, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Chawla et al. 2002. Each synthetic fraud is a random point on the segment
    between a real fraud and one of its k nearest fraud neighbours."""
    rng = np.random.default_rng(seed)
    n_new = _n_to_add(y, ratio)
    Xp = X[y == 1]
    if n_new <= 0 or len(Xp) < 2:
        return X, y
    mu, sd = _zscore_fit(Xp)
    Xpz = (Xp - mu) / sd
    nbr = _knn_indices(Xpz, Xpz, k, drop_self=True)

    seeds = rng.integers(0, len(Xp), n_new)
    partners = nbr[seeds, rng.integers(0, nbr.shape[1], n_new)]
    synth = _interpolate(rng, Xp[seeds], Xp[partners])
    return np.vstack([X, synth]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])


def borderline_smote(X, y, *, ratio: float, seed: int, k: int = 5,
                     m: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Han et al. 2005 (BorderlineSMOTE-1).

    Only frauds sitting in "danger" -- more than half of their m nearest
    neighbours in the FULL training set are legitimate -- get synthesised from.
    Interior frauds are already learned; the model's mistakes live on the
    boundary, so that is where the extra density should go.
    """
    rng = np.random.default_rng(seed)
    n_new = _n_to_add(y, ratio)
    Xp = X[y == 1]
    if n_new <= 0 or len(Xp) < 2:
        return X, y

    mu, sd = _zscore_fit(X)
    Xz, Xpz = (X - mu) / sd, (Xp - mu) / sd
    nbr_all = _knn_indices(Xpz, Xz, m, drop_self=True)
    n_majority_around = (y[nbr_all] == 0).sum(axis=1)

    # "danger": at least half but not all neighbours are majority. All-majority
    # is treated as noise and excluded -- amplifying a mislabelled or freak row
    # into hundreds of copies is how SMOTE earns its bad reputation.
    danger = (n_majority_around >= nbr_all.shape[1] / 2) & (
        n_majority_around < nbr_all.shape[1])
    if danger.sum() < 1:
        return smote(X, y, ratio=ratio, seed=seed, k=k)

    Xd = Xp[danger]
    nbr_pos = _knn_indices((Xd - mu) / sd, Xpz, k, drop_self=False)
    seeds = rng.integers(0, len(Xd), n_new)
    partners = nbr_pos[seeds, rng.integers(0, nbr_pos.shape[1], n_new)]
    synth = _interpolate(rng, Xd[seeds], Xp[partners], hi=0.5)
    return np.vstack([X, synth]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])


def adasyn(X, y, *, ratio: float, seed: int, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """He et al. 2008. Same interpolation as SMOTE, but the number of synthetic
    points per seed is proportional to how many legitimate rows surround it --
    density adapts to where the classifier is struggling instead of spreading
    the budget evenly."""
    rng = np.random.default_rng(seed)
    n_new = _n_to_add(y, ratio)
    Xp = X[y == 1]
    if n_new <= 0 or len(Xp) < 2:
        return X, y

    mu, sd = _zscore_fit(X)
    Xz, Xpz = (X - mu) / sd, (Xp - mu) / sd
    nbr_all = _knn_indices(Xpz, Xz, k, drop_self=True)
    r = (y[nbr_all] == 0).mean(axis=1)
    if r.sum() < EPS:
        return smote(X, y, ratio=ratio, seed=seed, k=k)
    r = r / r.sum()

    nbr_pos = _knn_indices(Xpz, Xpz, k, drop_self=True)
    seeds = rng.choice(len(Xp), n_new, replace=True, p=r)
    partners = nbr_pos[seeds, rng.integers(0, nbr_pos.shape[1], n_new)]
    synth = _interpolate(rng, Xp[seeds], Xp[partners])
    return np.vstack([X, synth]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])


def gaussian_jitter(X, y, *, ratio: float, seed: int,
                    sigma: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Additive noise scaled to each feature's own minority-class spread.

    Unlike SMOTE this can leave the convex hull of the observed frauds, which is
    the point: interpolation can only fill gaps between frauds we have already
    seen, and the frauds we have not seen are not all inside that hull.
    """
    rng = np.random.default_rng(seed)
    n_new = _n_to_add(y, ratio)
    Xp = X[y == 1]
    if n_new <= 0 or len(Xp) < 1:
        return X, y
    spread = Xp.std(axis=0)
    spread[spread < EPS] = 0.0
    pick = rng.integers(0, len(Xp), n_new)
    synth = Xp[pick] + rng.normal(0.0, 1.0, size=(n_new, X.shape[1])) * (sigma * spread)
    return np.vstack([X, synth]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])


def smote_enn(X, y, *, ratio: float, seed: int, k: int = 5, k_clean: int = 3,
              candidate_factor: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE, then Edited Nearest Neighbours cleaning.

    Oversampling pushes synthetic frauds into legitimate territory and leaves
    the two classes interleaved there. ENN then deletes any row whose own class
    disagrees with the majority of its neighbours, which removes both the worst
    synthetic points and the genuinely ambiguous real ones. Cleaning can never
    delete a REAL fraud -- there are only ~330 of them in the training split
    and none can be spared to a heuristic.

    THE CANDIDATE SET IS BOUNDED, and it has to be. Textbook ENN asks for the
    k nearest neighbours of every row, which on 200,000 rows in 33 dimensions
    is a 200k x 200k search: measured here it ran for over ten minutes and made
    this arm alone cost more than the other sixteen combined.

    The bound is not a shortcut, it follows from the rule. A row is deleted only
    when most of its neighbours belong to the OTHER class, so a legitimate
    transaction sitting far from every fraud is unconditionally safe: all of its
    neighbours are legitimate too. Only rows near the minority class can qualify.
    So the neighbour search is run over the whole dataset -- exact distances --
    but only for the frauds plus the `candidate_factor * k_clean * n_fraud`
    legitimate rows closest to them. Everything else is kept without asking.
    Rank the majority by distance to the minority set first, which is a query
    against a few thousand reference points instead of two hundred thousand.
    """
    Xr, yr = smote(X, y, ratio=ratio, seed=seed, k=k)
    n_orig = len(X)
    mu, sd = _zscore_fit(Xr)
    Xz = (Xr - mu) / sd

    pos = np.flatnonzero(yr == 1)
    if len(pos) == 0:
        return Xr, yr
    nn_min = NearestNeighbors(n_neighbors=1).fit(Xz[pos])
    dist_to_fraud = nn_min.kneighbors(Xz, return_distance=True)[0][:, 0]

    budget = min(len(yr), candidate_factor * k_clean * len(pos))
    candidates = np.union1d(pos, np.argsort(dist_to_fraud)[:budget])

    nbr = _knn_indices(Xz[candidates], Xz, k_clean, drop_self=True)
    disagree = (yr[nbr] != yr[candidates][:, None]).mean(axis=1) > 0.5

    protect_real_fraud = (candidates < n_orig) & (yr[candidates] == 1)
    drop = np.zeros(len(yr), dtype=bool)
    drop[candidates[disagree & ~protect_real_fraud]] = True
    return Xr[~drop], yr[~drop]


RESAMPLERS = {
    "random_oversample": random_oversample,
    "smote": smote,
    "borderline_smote": borderline_smote,
    "adasyn": adasyn,
    "jitter": gaussian_jitter,
    "smote_enn": smote_enn,
}


def resample(name: str | None, X, y, *, ratio: float, seed: int,
             split: str = "train", **kw) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch, with the train-only rule enforced rather than documented."""
    if split != "train":
        raise ValueError(
            f"refusing to resample the {split!r} split -- synthetic positives may "
            "only ever be added to training data"
        )
    if name in (None, "none"):
        return X, y
    if name not in RESAMPLERS:
        raise KeyError(f"unknown resampler {name!r}; have {sorted(RESAMPLERS)}")
    return RESAMPLERS[name](X, y, ratio=ratio, seed=seed, **kw)


# ----------------------------------------------------------------- reweighting
def class_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights, normalised to mean 1 so the effective learning
    rate does not change when the scheme does."""
    w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0).astype(float)
    return w / w.mean()


def effective_number_weights(y: np.ndarray, beta: float = 0.9999) -> np.ndarray:
    """Class-balanced loss (Cui et al. 2019).

    Inverse frequency assumes 350 frauds carry 350 rows' worth of information.
    They do not -- frauds overlap heavily, so the marginal 300th is worth less
    than the 3rd. The effective number (1-beta^n)/(1-beta) discounts that
    redundancy, landing between 1x and the full 579x depending on beta.
    """
    w = np.ones(len(y), dtype=float)
    for cls in (0, 1):
        n = int((y == cls).sum())
        eff = (1.0 - beta ** n) / (1.0 - beta) if n else 1.0
        w[y == cls] = 1.0 / max(eff, EPS)
    return w / w.mean()


def amount_weights(y: np.ndarray, amounts: np.ndarray, *, fp_cost: float,
                   cap_percentile: float = 99.0) -> np.ndarray:
    """Cost-sensitive weighting: train on the money, not on the counts.

    The deployed cost model already says a missed fraud costs its transaction
    amount and a wrong block costs a flat FP_COST. Weighting each row by the
    money it puts at risk makes the training objective agree with the objective
    the merchant is judged on, instead of leaving the two to be reconciled by
    the threshold alone.

    Amounts are winsorised at the 99th percentile of the frauds: one
    25,691-unit transaction would otherwise carry more weight than 500 typical
    frauds and the model would fit that single row.
    """
    w = np.full(len(y), float(fp_cost))
    pos = y == 1
    if pos.any():
        cap = float(np.percentile(amounts[pos], cap_percentile))
        w[pos] = np.clip(amounts[pos], 1.0, max(cap, 1.0))
    return w / w.mean()


WEIGHTERS = {
    "none": lambda y, **kw: np.ones(len(y)),
    "class_weight": lambda y, **kw: class_weights(y),
    "effective_number": lambda y, **kw: effective_number_weights(y),
    "amount_weight": lambda y, amounts=None, fp_cost=50.0, **kw: amount_weights(
        y, amounts, fp_cost=fp_cost),
}


def sample_weights(name: str, y, *, amounts=None, fp_cost: float = 50.0) -> np.ndarray:
    if name not in WEIGHTERS:
        raise KeyError(f"unknown weighting {name!r}; have {sorted(WEIGHTERS)}")
    return WEIGHTERS[name](y, amounts=amounts, fp_cost=fp_cost)


# --------------------------------------------------------------- target shaping
def smooth_labels(y: np.ndarray, *, eps_pos: float = 0.0,
                  eps_neg: float = 0.0) -> np.ndarray:
    """Label smoothing, asymmetric on purpose.

    A hard 0 on 245,000 legitimate rows asserts that none of them was fraud.
    Some were -- this dataset's labels come from disputes, and fraud that was
    never disputed is silently labelled legitimate. Nudging negatives to
    `eps_neg` states that uncertainty instead of training the model to be
    infinitely confident about it, which also stops boosting from driving
    margins to +/-inf on the easy majority and spending its capacity there.

    XGBoost's logistic objective accepts fractional labels directly, so this
    needs no custom loss.
    """
    out = np.asarray(y, dtype=float).copy()
    out[np.asarray(y) == 1] = 1.0 - eps_pos
    out[np.asarray(y) == 0] = eps_neg
    return out


# ------------------------------------------------------------------ focal loss
@dataclass
class FocalLoss:
    """Lin et al. 2017, as an XGBoost custom objective.

    scale_pos_weight applies one constant multiplier to every fraud, including
    the thousands the model already gets right. Focal loss instead multiplies
    each row's loss by (1 - p_t)^gamma, so a confidently-correct row contributes
    almost nothing and the gradient concentrates on the hard, ambiguous
    transactions near the boundary -- exactly the population that decides
    precision at the operating threshold.

    The hessian is taken by central difference on the analytic gradient. The
    closed form exists but is long, easy to get subtly wrong, and a subtly wrong
    hessian shows up as a slightly worse model rather than as an error -- the
    worst failure mode available. The finite difference costs two extra sigmoid
    evaluations per boosting round and is verifiably right.
    """

    gamma: float = 2.0
    alpha: float = 0.75
    h_floor: float = 1e-6
    fd_step: float = 1e-4
    name: str = field(default="focal", init=False)

    def _grad(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = sigmoid(z)
        pt = np.where(y >= 0.5, p, 1.0 - p)
        pt = np.clip(pt, 1e-9, 1.0 - 1e-9)
        at = np.where(y >= 0.5, self.alpha, 1.0 - self.alpha)
        dl_dpt = at * (1.0 - pt) ** (self.gamma - 1.0) * (
            self.gamma * np.log(pt) - (1.0 - pt) / pt
        )
        return dl_dpt * np.where(y >= 0.5, 1.0, -1.0) * p * (1.0 - p)

    def __call__(self, z: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        y = dtrain.get_label()
        g = self._grad(z, y)
        h = (self._grad(z + self.fd_step, y) - self._grad(z - self.fd_step, y)) / (
            2 * self.fd_step
        )
        h = np.maximum(h, self.h_floor)
        w = dtrain.get_weight()
        if w is not None and len(w) == len(y):
            g, h = g * w, h * w
        return g, h


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=float), -50, 50)))
