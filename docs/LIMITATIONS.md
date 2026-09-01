# Limitations — what constrains this system, measured rather than guessed

Every number here comes from a script in this repo, not from intuition. Where a
limitation could be quantified, it was.

---

## The one that dominates everything: 492 frauds

The dataset has 284,807 transactions but only **492 frauds**, and after the
70/15/15 split the test set holds **71**. Every downstream conclusion is
noise-limited by that number.

Measured consequence — 95% bootstrap CI on test PR-AUC:

| model | PR-AUC | 95% CI | width |
|---|---:|---|---:|
| XGBoost (shipped) | 0.8061 | [0.7130, 0.8870] | **±0.087** |

Anything that moves PR-AUC by less than about 0.09 is **undetectable on this
data**. That single fact explains the two results below.

---

## 1. Hyperparameter tuning: no measurable effect

50 Optuna trials improved validation PR-AUC (0.8451 → 0.8532) but test PR-AUC
went 0.8150 → 0.8061. Paired bootstrap on the difference:

**−0.0090, 95% CI [−0.0387, +0.0221] → indistinguishable.**

See [`reports/model_selection.md`](../reports/model_selection.md).

## 2. LightGBM, CatBoost and ensembling: also no measurable effect

`python scripts/ensemble_experiment.py`. All thresholds selected on validation.

| model | test PR-AUC | 95% CI | test cost | fp | fn |
|---|---:|---|---:|---:|---:|
| xgboost (shipped) | 0.8061 | [0.7130, 0.8870] | 4,038 | 16 | 14 |
| **lightgbm** | **0.8199** | [0.7318, 0.8945] | **3,497** | 5 | 15 |
| catboost | 0.8106 | [0.7222, 0.8870] | 3,743 | 5 | 19 |
| ensemble: mean prob (3) | 0.8176 | [0.7300, 0.8950] | **3,489** | 5 | 15 |
| ensemble: rank-avg (3) | 0.8173 | [0.7253, 0.8934] | 11,395 | 178 | 9 |
| ensemble: rank-avg (xgb+lgb) | 0.8121 | [0.7178, 0.8899] | 15,946 | 269 | 10 |

Paired bootstrap against the shipped XGBoost, 2,000 resamples:

| candidate | Δ PR-AUC | 95% CI | verdict |
|---|---:|---|---|
| lightgbm | +0.0138 | [−0.0139, +0.0419] | same |
| catboost | +0.0045 | [−0.0173, +0.0291] | same |
| ensemble: mean prob (3) | +0.0116 | [−0.0047, +0.0320] | same |
| ensemble: rank-avg (3) | +0.0112 | [−0.0013, +0.0253] | same |
| ensemble: rank-avg (xgb+lgb) | +0.0061 | [−0.0079, +0.0205] | same |

**Every interval straddles zero.** LightGBM and the 3-model ensemble have
nicer point estimates, and the ensemble is the best on cost (3,489 vs 4,038) —
but none of it is statistically distinguishable from one XGBoost. Ensembling
here buys ~0.01 PR-AUC against a ±0.09 measurement error.

### Three real things that experiment did teach

1. **`scale_pos_weight` does not port across libraries.** XGBoost wants it
   (599). LightGBM is *hurt* by it: val PR-AUC 0.32 with, 0.85 without. Copying
   the imbalance recipe across is what makes a library look bad.
2. **LightGBM's default metric list breaks early stopping here.** With
   `binary_logloss` still in the list, stopping triggers on *any* metric failing
   to improve — and logloss degrades immediately under an imbalanced fit, so
   training halted at **iteration 2** (PR-AUC 0.0014). Set `metric="None"` and
   a proper eval function. LightGBM also needs a much lower learning rate and
   ~800 trees where XGBoost peaks at 160.
3. **Rank-averaging destroys the cost model.** The rank ensembles score fine on
   PR-AUC (0.817) but cost 3× more (11,395), because ranks are not
   probabilities, so a threshold chosen in rank-space does not transfer. If you
   ensemble, average probabilities — or recalibrate afterwards.

**Verdict on your question:** yes, you *can* add LightGBM and CatBoost, and the
code to do it is in the repo. It is not where the remaining value is. The
binding constraint is 492 positives, not the model family.

---

## 2b. Class-imbalance techniques: also no measurable effect

`python run.py imbalance` benchmarks seventeen treatments -- SMOTE,
Borderline-SMOTE, ADASYN, SMOTE+ENN, random oversampling, Gaussian jitter,
inverse-frequency and class-balanced weights, amount-weighted cost-sensitive
training, focal loss at two gammas, asymmetric label smoothing and two combos --
against the shipped `scale_pos_weight` under identical hyperparameters,
identical fixed rounds and no early stopping, three seeds each.

**The shipped one-line recipe won.** Second place (SMOTE, 0.8455 against
0.8478) is 0.0023 behind, which is half the seed-to-seed standard deviation of
either arm, and both sit far inside the +/-0.087 CI that 71 test frauds impose.

What that actually licenses us to say is narrow, so it is worth being precise:

* **Not** "SMOTE does not work". Only: on 331 training frauds of 28
  already-decorrelated PCA components, interpolating between them adds nothing
  a constant loss multiplier does not already provide. SMOTE scored within noise
  of plain duplication (0.8415), and duplication adds no information by
  construction -- so whatever SMOTE contributed here, it was not new structure.
* Synthesising *more* was worse, not better: 30x frauds scored 0.8361 against
  6x at 0.8455.
* The cost-sensitive `amount_weight` arm came last on PR-AUC (0.7858) because
  PR-AUC weights every fraud equally while that arm deliberately does not. It
  produced 0.972 precision at 0.657 recall -- a different operating point, not
  a worse model, and the leaderboard's ranking metric cannot express that.
  Judging cost-sensitive training by a cost-blind metric is a real limitation of
  this bench, and the reason its cost column is reported alongside.
* Untested here: resampling in a *learned* representation rather than raw PCA
  space, and any generative model of fraud. Both need more positives than this
  dataset has.

---

## 3. The random split is optimistic — the largest measured effect here

This is the most serious methodological limitation, and it is **bigger than
every modelling choice above combined.**

Our split is random. Production scores *future* transactions, so a random split
lets the model learn from transactions that happened after the ones it is
tested on. Training on the first 85% of the time window and testing on the last
15%, same model config, same split sizes:

| protocol | test PR-AUC |
|---|---:|
| RANDOM (our protocol) | 0.9228 |
| TIME-ordered (past → future) | **0.7633** |
| **optimism** | **+0.1595** |

The random split flatters the model by **~0.16 PR-AUC** — roughly *eighteen
times* the effect of the entire hyperparameter search, and larger than the
confidence interval. (Caveat: the time-ordered window holds only 52 frauds, so
this is directional, not decisive.)

We kept the random split because it is what the published baselines for this
benchmark use, and comparability was the point. But a production system must be
validated time-ordered, and this is the first thing to change.

---

## 4. The cost-optimal threshold was being selected on the test split (fixed)

**This was a genuine bug in the first version and has been corrected.**
`evaluate.py` originally swept thresholds on the test split, took the argmin,
and reported the cost at that argmin — fitting the threshold to the same 71
frauds it was scored against.

| | threshold | test cost |
|---|---:|---:|
| selected on test (old, optimistic) | 0.98 | 3,629 |
| selected on val, applied to test (correct) | 0.09 | **4,038** |
| selection bias | | **+408 (11.3%)** |

Corrected headline: **56.3% cost reduction, not 60.7%.**

Note also that the val-optimal threshold (0.09) and the test-optimal (0.98) are
nowhere near each other, while total cost varies only 4.1% across the whole
range above 0.5. The cost curve is so flat that **the argmin is not
identifiable** from this much data — which is itself the finding: have a cost
model, but do not believe the third decimal place of its minimum.

### 4b. …and the corrected version was still not really choosing a threshold

The fix above moved threshold selection from test to validation. That is the
right split, but for the *tuned* model it is not an out-of-sample one:
`tune_optuna.refit_final` refits the winning configuration on **train+val** so
no data is wasted. The tuned model therefore separates validation perfectly.

Measured on the validation split with the shipped tuned model:

| | value |
|---|---:|
| true positives | 71 of 71 |
| false positives | 0 |
| false negatives | 0 |
| cost at every threshold from 0.01 to 0.50 | **0** |

`np.argmin` returns the first index of a tie. **The shipped operating threshold
of 0.09 was that tie-break**, not a minimum. The baseline model, which is *not*
refit on val, selected 0.72 from the same code path — the eightfold gap between
the two was never a modelling difference at all.

**The fix** ([`src/threshold.py`](../src/threshold.py),
[`src/calibrate.py`](../src/calibrate.py)): score train+val out-of-fold with
5-fold CV, refitting the whole recipe inside each fold, then smooth the cost
curve and apply a *paired* one-standard-error rule.

| | threshold | test cost | FP | FN | precision |
|---|---:|---:|---:|---:|---:|
| tie-break artefact (shipped) | 0.09 | 4,038 | 16 | 14 | 0.7808 |
| out-of-fold + smoothed + paired one-SE | 0.66 | **3,679** | **4** | 16 | **0.9322** |

**Why the paired bootstrap, and not the ordinary one.** The first
implementation used the bootstrap standard error of the cost *level*. On this
data that is 3,003 against a minimum of 12,434 — 24% — because it is dominated
by which expensive frauds land in the resample, and that draw moves the entire
curve up or down together. Using it admitted every threshold from 0.07 to 0.71
and selected one costing 31% more than the minimum on the selection data
itself. Pairing the bootstrap, exactly as `paired_bootstrap_delta` already does
for model comparison, cancels the shared level noise and measures the quantity
that matters: whether operating *here* is really no worse than operating
*there*.

**The remaining honest caveat.** The admissible band [0.34, 0.66] contains
**zero** of the 241,167 out-of-fold transactions and zero of the 42,559 test
ones. The model is bimodal, so that band is empty score-space and every
threshold in it makes identical decisions. This is good — the operating point
is robust to the curve shifting underneath it — but it also means the one-SE
rule is not finely resolving a threshold here. It is finding the edge of a gap.
On a dataset with a denser score distribution the rule would have real work to
do and would need re-examining, not assuming.

---

## 5. Data limitations that no modelling can fix

- **No entity identifiers.** There is no card id, customer id, device id, or
  merchant id. Real fraud detection is dominated by *velocity and graph*
  features — same card five times in ten minutes, one device across forty
  accounts, a merchant whose chargeback rate just tripled. **None of them are
  constructible here.** This is a far bigger ceiling than model choice: it is
  the difference between "classify a transaction in isolation" and how fraud is
  actually caught in production.
- **`V1`–`V28` are anonymised PCA components.** No domain feature engineering
  is possible, and explanations cannot name a business cause. It also means the
  PCA basis was fitted on the full dataset before release — a mild, unavoidable
  leak baked into the benchmark itself.
- **Two days, September 2013, European cardholders.** Thirteen years old, one
  geography. EMV, 3-D Secure 2, tokenisation and contactless have all reshaped
  card fraud since. The *pipeline* transfers; these coefficients do not.
- **Amounts are EUR**, presented as generic currency units. The FP cost of 50 is
  on that scale, not rupees — a Razorpay deployment would restate both.
- **Label provenance is unknown.** Presumably confirmed chargebacks. Reporting
  lag means some "legitimate" rows are probably undetected fraud, which
  understates recall by an unmeasurable amount.
- **1,081 exact duplicate rows** existed and are dropped before splitting; left
  in, they leak across splits.

## 6. Model and system limitations

- **Scores are not calibrated probabilities.** `scale_pos_weight` deliberately
  distorts them, which is why the cost-optimal threshold lands at 0.09 or 0.98
  rather than somewhere interpretable. Isotonic or Platt calibration on the
  validation split would make the score meaningful as "probability of fraud" —
  a prerequisite for the cost model to be principled rather than empirical.
  **This is more valuable than ensembling.**
- **No drift monitoring.** The block-rate gate is a backstop for the symptom,
  not the cause.
- **The gate is per-process and in-memory** — no shared state across replicas,
  does not survive restart.
- **The audit log is append-only by convention**, not hash-chained or WORM.
- **Review capacity is assumed infinite.** The gate routes overflow to humans
  without modelling whether humans keep up.
- **The FP cost (50) is an assumption, not a measurement.**

---

## If you had one more week, in priority order

1. **Time-ordered validation.** The +0.16 optimism is the largest measured
   error in the project.
2. **Calibrate the scores.** Makes the cost model principled and the threshold
   interpretable.
3. **A dataset with entity ids**, enabling velocity/graph features. This is the
   only route to a genuinely better model.
4. Drift monitoring with threshold recalibration.
5. *Then*, if anything is left, ensembling — for the ~0.01 it is worth.
