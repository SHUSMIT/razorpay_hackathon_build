# Where we left off — resume here

Written 2026-09-02, late. Everything below is on disk and committed; nothing
needs redoing.

---

## State of the model

All three families are trained at a **60-day recency half-life** and their
artefacts are in `models/`. The half-life change was the big win of the
session: the tuner had been pinned to the floor of its own search grid.

| Model | at 180 d | at 60 d (current) |
|---|---|---|
| **XGBoost** | 70.49% | **79.09%** |
| HistGB | 65.86% | 75.00% |
| CatBoost | 59.68% | 71.67% |

Blend weights were searched over all 66 simplex points and came back
**pure XGBoost (1.0 / 0.0 / 0.0)** — the blend does not help even when the
search is working correctly. Calibration is good: expected calibration error
0.645% → 0.0117% on held-back validation.

Effective sample size is ~2,000 rows for each model. That is small, and it is
the honest cost of an aggressive half-life. It is recorded in
`reports/training_summary.json`.

---

## The one thing still in flight

**A re-tune of XGBoost's hyper-parameters at the 60-day half-life.** The
current parameters were tuned when the half-life was 180 days and the effective
sample was four times larger, so `min_child_weight=139` and `max_depth=7` are
probably too restrictive now.

It is **paused, not lost**. 4 of 30 trials are banked in `reports/optuna.db`
(best 79.67%, which has not yet beaten the current parameters' 80.05% on
val_fit). Resume with:

```bash
python -m src.tune --trials 30 --models xgboost --force
```

It picks up at trial 5. Give it the machine to itself — running it alongside a
CatBoost fit slowed it to one trial per twelve minutes.

The pre-retune parameters are backed up at
`models/best_params_xgboost.json.bak180` in case the re-tune is worse and we
want to revert.

---

## Next steps, in order

1. **Finish the re-tune** (~15 min on an idle machine). If it beats 80.05% on
   val_fit, retrain XGBoost and re-run `assess`; if not, keep what we have and
   say so.
2. **Decide the shipped model** on val_fit between: CatBoost, retuned XGBoost,
   and the blend. Right now XGBoost alone wins on every measurement.
3. **Re-run the tail of the pipeline** if anything changed:
   `python -m src.train && python -m src.assess && python -m src.context`,
   then `python scripts/update_readme.py`.
4. **Walk the dashboard** — this has NOT been done end to end against the
   finished models and is the last real risk before filming:
   ```bash
   python run.py api      # terminal 1
   python run.py app      # terminal 2 -> localhost:8501
   ```
   Check all four tabs: score a known fraud, deliberately draw one the model
   misses, run the burst on Live Score and watch the gate downgrade, resolve a
   case in Review Queue, confirm Model Performance renders both plots, and
   confirm the Audit Trail fills.
5. **Film**, following `docs/PITCH.md`.

---

## What was settled today (do not re-litigate)

Each of these was measured, and the evidence is in `docs/FINDINGS.md`:

- **Velocity features do not help here.** Eight causal per-card features were
  built and tested: 69.99% without, 69.09% with. Fraud cards in this dataset
  make 0.18 transactions in the prior hour against 0.18 for legitimate ones —
  the generator does not simulate card takeover. The code stays (it is valuable
  on real traffic) but the features are excluded from `FEATURES`.
- **Class weights on the recency-weighted population**: +0.03 pp. Noise.
- **Dropping near-zero-weight rows**: −2.42 pp. Those rows are not dead weight;
  they anchor the base rate.
- **Amount-weighted positives**: −10.90 pp on ranking. It *did* win on cost
  (61.0% vs 59.6% of loss avoided) but by catching fewer frauds, and the gap is
  within noise on 555 positives. Rejected.
- **Ensembling**: no gain. HistGB and XGBoost correlate at rho = 0.935.
- **PCA / deep learning**: not attempted, and should not be. Trees split on
  axis-aligned thresholds so rotation hurts, and PCA would destroy the
  named-feature explanations the whole service is built around.

---

## Known open issues

- **`tune.py` now scores trials on val_fit only** (changed today, so val_sel
  stays untouched for threshold selection). The three `best_params_*.json`
  files on disk were produced under the OLD behaviour, which scored on full
  validation. Only the XGBoost re-tune will use the corrected protocol.
- **The half-life is applied but its companions were not re-tuned around it**
  for CatBoost and HistGB. If either is ever shipped, re-tune it first.
- **The dashboard has not been verified against the finished models.** It boots
  and renders with the service offline, and `src/demo_payload.py` is covered by
  14 tests, but no one has clicked through the four tabs with a live API.

---

## Quick sanity check before starting

```bash
python -m pytest -q tests        # expect ~74 passed
git log --oneline | head -5
tail -20 reports/assess_log.txt  # the held-out numbers
```
