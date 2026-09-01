# 🛡️ Fraud Risk Manager

**Razorpay AI Buildathon — Track 02, AI Risk Manager**

A merchant-facing transaction risk service. It scores a payment, explains the
score, makes a **bounded** decision (`allow` / `review` / `block`), refuses to
block more of the merchant's traffic than a hard cap allows, and writes every
decision to an append-only audit log.

Strictly defense-only. Nothing in this repository helps anyone evade fraud
detection.

---

## The problem

Fraud and chargebacks eat merchant margin, but the naive fix eats more of it.
A model tuned to catch every fraud declines real customers at checkout, and on
this dataset legitimate transactions outnumber fraud **579 to 1** — so a
model that is slightly too aggressive costs the merchant more in lost good
customers than the fraud ever did.

The interesting question is therefore not "can you detect fraud" but **"where
do you put the threshold, and what stops the system when it goes wrong."**
This project answers both explicitly and in money.

---

## Architecture

```
  Hugging Face Hub                 ┌──────────── offline / training ──────────────┐
  TekMonger/creditcard-fraud       │                                              │
  284,807 real transactions        │  data_prep.py    dedupe → stratified 70/15/15│
          │                        │       │          → LEAKAGE ASSERTION (hash)  │
          ▼                        │       ▼                                      │
  data/raw/creditcard.csv ─────────┼──► features.py   V1..V28 + amount + cyclical │
                                   │       │                        hour-of-day   │
                                   │       ▼                                      │
                                   │  train.py        XGBoost baseline            │
                                   │       │          scale_pos_weight = 599      │
                                   │       ▼                                      │
                                   │  tune_optuna.py  50 TPE trials, PR-AUC on    │
                                   │       │          VALIDATION ONLY             │
                                   │       │          refit on train+val          │
                                   │       ▼                                      │
                                   │  evaluate.py     ONE look at the test split: │
                                   │       │          PR-AUC + cost curve         │
                                   │       │          → cost-optimal threshold    │
                                   │       ▼                                      │
                                   │  explain.py      TreeSHAP global + per-row   │
                                   └───────┬──────────────────────────────────────┘
                                           │  models/tuned.json
                                           │  models/serving_config.json  ← thresholds
                                           ▼
  ┌──────────────────────── online / serving ─────────────────────────────────┐
  │                                                                           │
  │   POST /score ──► score ──► BAND         ──► GATE            ──► AUDIT    │
  │                             allow/review/    block-rate cap      append-   │
  │                             block from the   5% of last 200      only      │
  │                             cost curve       else → review       .jsonl    │
  │                                │                    │               │      │
  │                                └──► top-3 TreeSHAP factors ─────────┘      │
  │                                                                           │
  │   GET /audit   ──► last N decisions      GET /gate ──► live block rate     │
  └───────────────────────────┬───────────────────────────────────────────────┘
                              ▼
                   app/streamlit_app.py
                   Live Score │ Model Report │ Audit Trail
```

---

## Run it locally

```bash
pip install -r requirements.txt        # or: python run.py setup

python run.py data       # download from HF, split, assert no leakage   (~1 min)
python run.py train      # baseline XGBoost                             (~30 s)
python run.py tune       # 50 Optuna trials, refit, evaluate both       (~20 min)
python run.py explain    # TreeSHAP summary                             (~20 s)
python run.py select     # paired-bootstrap model selection report      (~1 min)

python run.py api        # terminal 1 → http://127.0.0.1:8000/docs
python run.py app        # terminal 2 → http://localhost:8501
```

`make <target>` works identically on Linux/macOS. `run.py` exists because
`make` is not installed on most Windows machines and a demo that cannot start
is not a demo.

To skip the 20-minute search and reproduce the shipped numbers directly:
`python run.py data && python run.py train && python -m src.evaluate --model tuned`
(after `models/tuned.json` exists).

---

## Honest metrics

Everything below is measured **once**, on a 42,559-transaction test split that
no training or tuning step ever touched.

<!-- METRICS_TABLE_START -->
Held-out test split: **42,559 transactions, 71 frauds**. Optuna: 50 TPE trials, best validation PR-AUC 0.8532.

| metric | baseline | tuned | note |
|---|---:|---:|---|
| PR-AUC | 0.8150 | **0.8061** | headline metric; random baseline 0.0017 |
| ROC-AUC | 0.9681 | 0.9732 | reported for comparability, not relied on |
| Precision @ 0.50 | 0.8769 | 0.9016 | share of blocks that were really fraud |
| Recall @ 0.50 | 0.8028 | 0.7746 | share of fraud caught |
| F1 @ 0.50 | 0.8382 | 0.8333 |  |
| **Operating threshold** | 0.72 | **0.09** | min of the cost curve **on validation**, not 0.5, not test |
| Precision at that threshold | 0.9048 | **0.7808** |  |
| Recall at that threshold | 0.8028 | **0.8028** |  |
| False positives there | 6 | **16** | good customers wrongly declined |
| Missed frauds there | 14 | **14** | out of 71 in the split |
| Cost at that threshold | 3,535 | **4,038** | vs 9,241 doing nothing |

**Bottom line.** At the operating threshold the tuned model catches 80.3% of fraud while wrongly declining **16** of 42,488 legitimate customers. In the cost model that is **4,038** lost against **9,241** for doing nothing — a **56.3%** reduction in fraud losses.


The threshold is selected on the **validation** split and merely applied to test. Taking the argmin on test itself would report 3,629 instead of 4,038 — an 10.1% selection bias we do not claim.

Full before/after table: `reports/tuning_comparison.txt`. Curves: `reports/pr_curve.png`, `reports/cost_curve.png`. Limitations: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).
<!-- METRICS_TABLE_END -->

### Tuning did not help, and we are reporting that

The most useful number in this project is one that went the wrong way.

Optuna improved PR-AUC on the **validation** split, 0.8451 → 0.8532, so the
tuned configuration was selected — a decision made before looking at test. On
the held-out test split it then scored **0.8061 against the baseline's 0.8150**.
Worse, not better.

So we checked whether that gap is real, with a paired bootstrap (2,000
resamples, both models scored on the same resampled rows each draw):

| | value |
|---|---|
| difference (tuned − baseline) | **−0.0090** |
| 95% CI | **[−0.0387, +0.0221]** |
| tuned better in | 29.6% of resamples |
| significant at 95% | **no** |

The interval straddles zero: **the two models are statistically
indistinguishable on this test split.** With only 71 frauds in it, the 95% CI
on a single PR-AUC is about ±0.08 wide (baseline [0.7295, 0.8923], tuned
[0.7188, 0.8907]). Quoting a four-decimal improvement off 71 positives would be
claiming a precision the data cannot support.

Under the corrected threshold protocol the baseline is in fact the *cheaper*
model on test (3,535 against 4,038). We still ship the tuned one, because the
selection rule was fixed before any test number was seen and the tuned config
won on validation. Re-picking the winner after seeing the test set is precisely
the sin this whole section exists to avoid — and since the two are statistically
indistinguishable, switching would be chasing noise.

The honest conclusion: on a dataset with this few positives, 50 trials of
hyperparameter search buys almost nothing that survives contact with held-out
data. More labelled fraud would help far more than more search. We report this
rather than re-running the search until a seed made the table look better —
which, on noise this size, it eventually would. Full working:
[`reports/model_selection.md`](reports/model_selection.md).

The same verdict holds for LightGBM, CatBoost and ensembles — see
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

**Why PR-AUC and not accuracy.** Predicting "legitimate" for every transaction
scores **99.83% accuracy** and catches zero fraud. ROC-AUC is barely better as
a headline: its denominator contains ~284,000 true negatives, so thousands of
false positives hardly move it. Precision has no such cushion — it is exactly
what the merchant feels, because every false positive is a real customer
declined at checkout. The random baseline for PR-AUC here is 0.0017.

### The cost model — stated, not implied

The threshold is not 0.5. It is whatever minimises money lost, under
assumptions written down in [`src/config.py`](src/config.py) and defensible out
loud:

| | assumption | why |
|---|---|---|
| **False negative** | costs the **actual amount** of the missed transaction | the chargeback claws back the full value. We charge the real amount, not the average, because fraud amounts are heavily skewed (median 9.25, max 2,125) — averaging would let a model that misses the expensive frauds look identical to one that misses the cheap ones. |
| **False positive** | costs a flat **50** | one support contact plus the goodwill/abandonment cost of a declined customer. It is a *constant we chose*, not a fitted parameter. |
| **True positive / true negative** | cost 0 | savings show up as fraud cost avoided. |

Change `FP_COST` in `src/config.py` and the optimal threshold moves — which is
the sensitivity a merchant should be shown, not hidden from. At `FP_COST = 50`
the sweep across thresholds 0.01→0.99 produces `reports/cost_curve.png`, and
the minimum of that curve is the threshold the API actually serves with
(`models/serving_config.json`). Nothing is hard-coded downstream.

---

## Bounded, gated decisioning

The part that makes this a risk *manager* rather than a classifier behind HTTP.

1. **Three-way banding, not a binary.** Scores above the cost-optimal threshold
   are blocked; scores in a band below it go to `review` — a human queue —
   instead of being auto-decided. Uncertainty routes to a person.
2. **A hard cap on the block rate.** The service will never auto-block more
   than **5% of the last 200 decisions**. If a burst would breach that, block
   decisions are **force-downgraded to `review`** and the reason is logged. So
   a model that suddenly wants to block everything — drift, an attack, a bad
   deploy — degrades into a review queue, **not a merchant outage**. The cap is
   live from the very first request: on a cold window the rate is measured
   against an assumed floor of 20 decisions, so the first block still goes
   through (1 in 20 *is* 5%) but a second consecutive one is already held. An
   exemption for the warm-up period would have let a burst blow straight
   through the bound before the window filled — which is exactly the scenario
   the gate exists to survive.
3. **Everything is audited.** Each decision appends one line to
   `reports/audit_log.jsonl`: timestamp, SHA-256 of the input, score, decision,
   the decision the model *wanted*, whether the gate intervened and why, the
   rule id, the threshold used, and the model version. The raw feature vector
   is never written — only its hash.

See it fire:

```bash
python run.py api                  # terminal 1
python scripts/demo_gate.py --n 80 # terminal 2
```

80 known-fraud transactions arrive, the model wants to block all 80, and the
gate lets only the capped share through while routing the rest to review.

---

## Explainability

Every `/score` response carries the **top 3 contributing features** with signed
SHAP values and a direction. Global drivers are in `reports/shap_summary.png`.

Implementation note: this uses **XGBoost's built-in TreeSHAP**
(`Booster.predict(pred_contribs=True)`) rather than the `shap` package. It is
the same TreeSHAP algorithm — Lundberg's implementation was upstreamed into
XGBoost's C++ core — but it carries no numba/llvmlite dependency, so the
service starts on locked-down machines and explanation is fast enough to sit
inside the request path. Correctness is asserted, not assumed: `explain.py`
checks that contributions plus bias reconstruct the raw margin exactly.

`V1`–`V28` are **anonymised PCA components** published by the dataset authors.
We label them as such and deliberately do not invent business meanings for
them — a test enforces this, because inventing a story for `V17` on camera
would be a lie.

---

## Data

[`TekMonger/creditcard-fraud`](https://huggingface.co/datasets/TekMonger/creditcard-fraud)
on the Hugging Face Hub — a mirror of the ULB Machine Learning Group /
Worldline `creditcardfraud` benchmark. 284,807 real transactions over two days
in September 2013, 492 frauds (0.1727%).

`data_prep.py` verifies the row and fraud counts against the published figures
and warns loudly if they do not match, then:

- drops the **1,081 exact duplicate rows** *before* splitting — left in, the
  same transaction can land in both train and test and inflate the held-out
  score for free;
- stratifies 70/15/15 so every split holds the same fraud rate;
- **asserts** that no row appears in two splits, by row id *and* by SHA-1
  content hash, and prints a PASS/FAIL banner.

`tests/test_data_prep.py` re-checks this independently, and includes a test
that deliberately poisons a split to prove the guard actually fails when
leakage exists.

Why this matters: PR-AUC above ~0.98 on this dataset is a leakage smell, not a
result. Ours is in the published-baseline range, which is the point.

---

## Class imbalance: seventeen techniques, one honest table

579 legitimate transactions for every fraud, and only **331 frauds in the whole
training split**. That is the defining constraint of this problem, so it gets
measured rather than asserted. `python run.py imbalance` runs seventeen
treatments under identical hyperparameters, identical fixed boosting rounds and
**no early stopping** — stopping on validation would let each arm peek at the
split it is scored on, and the arms that overfit fastest would look best.

Four families, because they attack the problem in four different places
([`src/imbalance.py`](src/imbalance.py)):

| family | arms | what it changes |
|---|---|---|
| **resampling** | SMOTE, Borderline-SMOTE, ADASYN, SMOTE+ENN, random oversample, Gaussian jitter | the training set — more minority rows |
| **reweighting** | `scale_pos_weight`, inverse frequency, class-balanced *effective number* (Cui et al.), **amount-weighted** | the loss per row |
| **loss shaping** | focal loss, γ ∈ {1, 2} | the loss function |
| **target shaping** | asymmetric label smoothing | the labels |

Everything is implemented directly on numpy — `imbalanced-learn` is not a
dependency — so the cost-sensitive and amount-aware variants this project
actually needs are not bolted onto someone else's API. Two rules are enforced
in code rather than documented: resampling **refuses to run on anything but the
training split**, and neighbour search happens in z-scored space (on raw
columns every Euclidean neighbour is decided by `Amount` alone and SMOTE
degenerates into "interpolate between two transactions of similar size").

### The result: nothing beat the simple thing

| rank | arm | family | train frauds | val PR-AUC | ± seed |
|---:|---|---|---:|---:|---:|
| 1 | `scale_pos_weight` | reweight | 331 | **0.8478** | 0.0048 |
| 2 | `smote` | resample | 1,983 | 0.8455 | 0.0034 |
| 3 | `borderline+amount` | combo | 1,983 | 0.8437 | 0.0085 |
| 4 | `random_oversample` | resample | 1,983 | 0.8415 | 0.0053 |
| 5 | `smote_enn` | resample | 1,971 | 0.8413 | 0.0039 |
| 6 | `class_weight` | reweight | 331 | 0.8410 | 0.0042 |
| 7 | `jitter` | resample | 1,983 | 0.8389 | 0.0155 |
| 8 | `smote_x30` | resample | 9,914 | 0.8361 | 0.0027 |
| 9 | `focal_loss_g1` | loss | 331 | 0.8354 | 0.0038 |
| 10 | `focal_loss` | loss | 331 | 0.8351 | 0.0027 |
| 11 | `none` | baseline | 331 | 0.8351 | 0.0006 |
| 12 | `smote+smoothing` | combo | 1,983 | 0.8338 | 0.0037 |
| 13 | `label_smoothing` | targets | 331 | 0.8326 | 0.0047 |
| 14 | `borderline_smote` | resample | 1,983 | 0.8314 | 0.0010 |
| 15 | `effective_number` | reweight | 331 | 0.8251 | 0.0062 |
| 16 | `adasyn` | resample | 1,983 | 0.8193 | 0.0052 |
| 17 | `amount_weight` | reweight | 331 | 0.7858 | 0.0090 |

**`scale_pos_weight` — one constant multiplier, the recipe already shipped —
won the validation bench outright.** SMOTE came second, 0.0023 behind, which is
half the seed-to-seed spread of either arm. Full table with the cost and
precision/recall columns:
[`reports/imbalance_experiment.md`](reports/imbalance_experiment.md).

Three things worth taking from a negative result:

* **Synthetic minority data did not add information here.** SMOTE (0.8455)
  barely separated from plain duplication of the same rows (0.8415), and
  duplication adds nothing by construction. When interpolation and copying
  score the same, interpolation is not finding new structure — it is padding.
* **More synthesis made it worse.** 30× the frauds (`smote_x30`, 0.8361) scored
  below 6× (0.8455). Past a point most of what the model knows about fraud is
  our own interpolation.
* **The cost-sensitive arm was the worst of all** (`amount_weight`, 0.7858),
  and instructively so: weighting rows by the money at risk teaches the model to
  find *expensive* fraud, and PR-AUC counts every fraud equally. It bought the
  second-highest precision in the table (0.972) at a recall of 0.657.

### What actually moved the money: the threshold was never really chosen

The useful finding came from auditing the existing pipeline rather than from
adding to it.

`tune_optuna.refit_final` refits the winning model on train+val so no data is
wasted — correct — and `evaluate` then picks the operating threshold as the
argmin of the **validation** cost curve. For a model that has already trained on
validation, that curve is *identically zero* from 0.01 to 0.50: it separates its
own training data perfectly (tp=71, fp=0, fn=0). `np.argmin` returns the first
index of a tie, so the shipped threshold of **0.09 was a tie-break artefact**,
not a decision.

[`src/threshold.py`](src/threshold.py) and [`src/calibrate.py`](src/calibrate.py)
replace it with three fixes:

1. **Out-of-fold scores.** 5-fold CV over train+val, refitting the entire recipe
   inside each fold, gives ~400 out-of-sample frauds to place the threshold on
   instead of 71 in-sample ones. Resampling happens *inside* each fold —
   oversample first and split afterwards and a synthetic fraud interpolated from
   row 7 lands in training while row 7 lands in the held-out fold, which is the
   most common way SMOTE produces recall that evaporates in production.
2. **Smoothing.** A cost curve built from 71 events is a staircase whose every
   step is one fraud crossing the threshold. A 7-wide centred moving average
   states what we are willing to believe: a minimum that only exists inside a
   0.03-wide window is not one we can reproduce next month.
3. **A one-standard-error rule, paired.** Among thresholds indistinguishable
   from the best, take the most conservative — the tie is broken on the thing
   the cost model cannot price, customers wrongly declined.

The word *paired* is doing real work there. The first implementation used the
bootstrap spread of the cost **level**, which is ~24% of the minimum because it
is dominated by which expensive frauds happened to be drawn — a draw that moves
the whole curve up and down together. That admitted everything from 0.07 to
0.71 and chose a threshold costing 31% more than the minimum on the very data
used to choose it. Pairing the bootstrap — the same technique this repo already
uses to compare models — cancels the shared level noise and leaves the
uncertainty that actually matters: is operating *here* really no worse than
operating *there*.

![threshold selection](reports/threshold_selection.png)

**The admissible band is empty.** 0 of 241,167 out-of-fold transactions score
anywhere between 0.34 and 0.66, and 0 of 42,559 test transactions do either.
The model is bimodal — confident fraud or confident not — so every threshold in
that band produces byte-identical decisions. Moving to the top of it costs
*nothing*, not merely little. The old 0.09 sat below the gap, on the noisy side
where small score movements flip decisions.

### What it is worth, on the test split, measured once

| | shipped (threshold 0.09) | corrected (threshold 0.66) |
|---|---:|---:|
| PR-AUC | 0.8061 | 0.8000 |
| precision at the operating point | 0.7808 | **0.9322** |
| recall at the operating point | 0.8028 | 0.7746 |
| **good customers wrongly declined** | 16 | **4** |
| frauds missed | 14 | 16 |
| **cost** | 4,038 | **3,679** |
| do nothing | 9,241 | 9,241 |

**Four false positives instead of sixteen, and 9% less money lost**, from
trading two missed frauds for twelve retained customers. The ranking did not
improve — the paired bootstrap on PR-AUC is −0.0060, CI [−0.0271, +0.0114], not
significant, and no improvement is claimed. It is the same model. What changed
is where the decision boundary was put, and that is where the money was.

Scores are also **isotonic-calibrated**, selected on *cross-fitted* ECE
(0.000125 → 0.000044); an isotonic fit scored on its own training scores
reports ~0 by construction, so the report shows the tautology and the honest
number side by side. `scale_pos_weight` is itself a base-rate distortion —
telling the learner a fraud is worth 600 legitimate rows trains it to report
probabilities for a world with 600× more fraud than this one — and the cost
model multiplies those probabilities by money.

Nothing above is deployed automatically. The run writes
`models/serving_config_candidate.json` and leaves the live config untouched;
`python scripts/imbalance_experiment.py --promote` is a separate, deliberate
act, because a script that quietly repoints the service at a model because it
won a bench will one day repoint it at something worse, unattended.

---

## End-to-end smoke test

```bash
python run.py smoke        # ~25 s, no network, touches no shipped artefact
```

The unit tests check each part in isolation. This checks the parts are still
**wired together** — the failure mode that survives a green suite and kills a
live demo. Eighteen stages, in pipeline order: real splits load and subsample →
features are deterministic and their column order still matches
`feature_names.json` → every resampler adds minority rows only → resampling
val/test is refused → the focal-loss gradient matches its numerical derivative
→ a model trains and scores in [0,1] → threshold selection is stable and the
one-SE rule is never *less* conservative than the minimum → calibration reduces
ECE without reordering, and survives serialisation → the cost model is
arithmetically consistent at both extremes → SHAP values sum to the margin →
the API health-checks, scores and bands correctly → eight malformed payloads
all return 422 rather than crashing the response path → a 60-request burst is
downgraded by the gate and the rolling rate stays under the cap → every
decision reached the audit log, hashed, with unique ids → the serving config is
ordered `0 < review ≤ block < 1`.

It writes only to a temp directory, so it is safe to run immediately before a
demo. The exit code is the number of failed stages, and it also runs inside
`python run.py test` via `tests/test_smoke.py`.

Two real defects were found by writing it. The malformed-input stage sends raw
bytes rather than `json=` because `httpx` refuses to serialise `NaN` in the
*client* — which would have silently turned the most interesting case into a
test of `httpx` rather than of the API's non-finite handler. And `smote_enn`
was doing a 200,000 × 200,000 neighbour search that ran for over ten minutes;
it is now bounded to 25 s by the observation that a legitimate transaction far
from every fraud cannot be ENN-deleted, since all of its neighbours are
legitimate too.

---

## Tests

```bash
python run.py test
```

Covering: split disjointness by exact hash (and that the guard fails when it
should); artefact reload determinism; a PR-AUC sanity band that fails both if
the model is too weak **and** if it is implausibly perfect; cost-model
behaviour at the threshold extremes; API handling of malformed input, missing
fields and extreme amounts; the gate downgrading blocks under a 100-request
burst while leaving normal traffic untouched; the audit log being append-only
and hash-only; and SHAP additivity, stability across runs, and variation
between transactions.

Plus, for the imbalance work: every resampler adds minority rows only and never
invents a majority one; the same seed produces the same data and a different
seed does not; SMOTE's synthetic points provably stay inside the hull of the
real frauds while jitter's provably leave it; SMOTE+ENN never deletes a real
fraud; amount weights are winsorised so one outlier cannot dominate; the focal
gradient matches a finite difference to 1e-6 and its hessian is strictly
positive everywhere; smoothing reduces variance without changing the grid
length; the one-SE rule is never less conservative than the minimum and never
claims a cost below the argmin; a higher `FP_COST` provably pushes the
threshold up; isotonic calibration never inverts two scores; and calibrators
round-trip through JSON. 84 tests total, plus the 18-stage smoke test.

---

## Known limitations

Measured, not guessed — full working in
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md). The three that matter most:

**1. 492 frauds is the binding constraint.** The test split holds 71. The 95%
CI on test PR-AUC is **±0.087**, so anything that moves the metric by less than
~0.09 is undetectable here. That is why 50 Optuna trials showed no measurable
effect — and why adding LightGBM, CatBoost and ensembles doesn't either:

| candidate | Δ PR-AUC vs XGBoost | 95% CI | verdict |
|---|---:|---|---|
| lightgbm | +0.0138 | [−0.0139, +0.0419] | same |
| catboost | +0.0045 | [−0.0173, +0.0291] | same |
| ensemble: mean prob (3) | +0.0116 | [−0.0047, +0.0320] | same |

Every interval straddles zero (`python run.py ensemble`). The model family is
not the bottleneck.

**2. The random split is optimistic by ~0.16 PR-AUC** — eighteen times the
effect of the entire hyperparameter search. Training on the past and testing on
the future gives 0.7633 against the random split's 0.9228
(`python run.py timesplit`). We kept the random split for comparability with
published baselines for this benchmark, but a production system must be
validated time-ordered. This is the first thing we would change.

**3. There are no entity identifiers.** No card, customer, device or merchant
id, so velocity and graph features — same card five times in ten minutes, one
device across forty accounts — are simply not constructible. That is a far
lower ceiling than any model choice imposes.

Also, briefly: `V1`–`V28` are anonymised PCA components, so explanations cannot
name a business cause; the data is two days of 2013 European card traffic;
scores are not calibrated probabilities (calibration would be worth more than
ensembling); the 50-unit false-positive cost is an assumption, not a
measurement; there is no drift monitoring; the gate is per-process and
in-memory; the audit log is append-only by convention rather than
cryptographically tamper-evident; and review capacity is assumed infinite.

One bug worth naming: the first version of `evaluate.py` picked the
cost-optimal threshold on the **test** split and reported the cost there —
fitting the threshold to the same 71 frauds it was scored against. Selecting on
validation instead costs 4,038 rather than 3,629, an **11.3% selection bias**.
The corrected number is what this README reports.

## What's next

Drift monitoring on the score distribution with automatic threshold
recalibration; a real Razorpay webhook integration; hash-chained audit records;
shared gate state across replicas; and extending beyond card fraud to
return/refund abuse, which is the other half of the margin problem.
