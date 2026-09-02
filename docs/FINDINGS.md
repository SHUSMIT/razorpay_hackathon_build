# What the data actually said

Six things were measured during this build that changed the model, and each
one would have quietly inflated the headline number if it had gone unchecked.
They are recorded here with the evidence, because "we checked and it was fine"
is not a claim anyone should take on trust.

Every number below is reproducible from the code in this repository.

---

## 1. A random split scores 685x higher — and it is a lie

The dataset covers 2010–2019. Split it at random and the model scores **61.0%
PR-AUC**. Split it by time, so evaluation only ever looks forward, and the same
model on the same features scores **0.089%** — barely above the 0.163% base
rate. That is a factor of roughly **685**, and the flattering number is the
wrong one.

The gap is not noise. It is the model recognising **people**.

There are only **2,000 cardholders and 6,146 cards** in this dataset, so the
combination of `yearly_income`, `total_debt`, `credit_score`,
`per_capita_income`, `current_age` and `num_credit_cards` is very close to a
unique fingerprint of a person. Under a random split, the same cardholder
appears on both sides, so the model learns *which people were defrauded* — and
scores brilliantly. Under a time split, the people defrauded in 2019 are not
the people defrauded in 2012, and the memorised answer is worthless.

Dropping the identity columns and the columns whose distribution marches with
the calendar (`card_age_years` and `years_since_pin_change` grow with the date,
so their test distribution is one the model never saw in training) took
validation PR-AUC from **0.089% to 4.657%** — a 52x improvement — on the
identical split.

| Configuration | val PR-AUC | Lift over base rate |
|---|---|---|
| All 36 features, deep trees | 0.089% | 1.5x |
| Behavioural features only | 0.146% | 2.4x |
| Behavioural + regularisation | 0.558% | 9.2x |
| Behavioural + regularisation + full train | **4.657%** | **76.9x** |

**What this means for the result:** the honest number is the small one. Any
published notebook on this dataset reporting 0.9+ AUC from a random split is
measuring identity recall, not fraud detection.

---

## 2. `merchant_state` was a lookup table wearing a feature's clothes

After the fix above, the model reached **86.87% PR-AUC**. That was too good,
so it got checked. Single-feature discrimination on validation:

| Feature | Alone, as a predictor | Lift |
|---|---|---|
| **merchant_state** | **89.39%** | **513x** |
| mcc_category | 1.83% | 11x |
| is_online | 1.34% | 8x |
| everything else | ≤ 0.33% | ≤ 2x |

One column beat the entire model. The reason is an artefact of how the
dataset's fraud labels were generated:

| Split | Period | Where the fraud is |
|---|---|---|
| train | 2010 → 2018-02 | 82.9% `ONLINE` (1.02% rate); Haiti 97.68%; Italy 19.91% |
| val | 2018-02 → 2018-11 | **97.7% of all fraud is `Italy`** (91.77% of Italy rows are fraud) |
| test | 2018-11 → 2019-09 | **94.9% of all fraud is `Italy`** (84.25% rate) |

A model that keeps this column is a one-rule lookup — *"Italy means fraud"* —
that would transfer to no real payment system, and would amount to geographic
profiling if it did. It is dropped.

Removing it costs a lot of headline and buys all of the credibility:
**46.63% → 15.18%** under identical settings. 15.18% on a 0.174% base rate is
still an **87x lift**, built on transaction behaviour rather than one string
comparison.

---

## 3. The fraud mechanism inverts halfway through the timeline

The most useful finding, and the one a random split hides completely.

| Split | Share of fraud that is… |
|---|---|
| train (2010 → 2018-02) | **83% card-not-present** (`Online Transaction`) |
| val (2018-02 → 2018-11) | **89% chip-present** (`Chip Transaction`) |
| test (2018-11 → 2019-09) | **86% chip-present** |

Card-not-present fraud runs at a 1.02% rate in the training era and 0.03% in
the validation era. The dominant attack changed completely.

This is why the early models looked so weak: they were not failing to learn,
they were faithfully learning a regime that no longer exists. Eight years of
training data was actively teaching the wrong pattern.

**The fix is exponential time decay on the training weights** (`src/weights.py`).
Every row keeps its place in the training set, but its influence halves every
*H* days. Discarding old rows outright would throw away millions of legitimate
transactions that still describe normal behaviour perfectly well; down-weighting
them keeps that value while letting the recent regime dominate.

*H* is not asserted. It is tuned by Optuna alongside every model
hyper-parameter, from the candidates {0 (off), 30, 60, 90, 120, 180, 365, 730}
days.

Measured on validation, identical features:

| Training weights | val PR-AUC |
|---|---|
| Class weights only | 15.18% |
| Class weights x recency decay | **46.00%** |

All three families independently selected **180 days** -- the most aggressive
decay the grid then offered. That unanimity was the clue: the search had hit
the edge of its own search space, so the real optimum was somewhere below and
had never been measured.

Extending the grid downward was worth roughly **nine points**:

| Half-life | val_fit | val_sel (held back) | Effective sample |
|---|---|---|---|
| 30 d | 78.59% | 75.98% | 1,140 |
| **60 d** | **79.99%** | **78.82%** | 1,983 |
| 90 d | 79.75% | 78.12% | 3,091 |
| 120 d | 77.26% | 76.36% | 4,665 |
| 180 d (the old floor) | 71.14% | 69.03% | 8,907 |

A genuine interior optimum this time, holding on both the slice used to choose
it and the slice held back. Retraining every family at 60 days lifted all
three: XGBoost 70.49% -> **79.09%**, HistGB 65.86% -> **75.00%**, CatBoost
59.68% -> **71.67%** on validation.

The cost is a small effective sample: at a 60-day half-life over an eight-year
span, the ~7.3M training rows carry the weight of about 2,000. That number is
recorded in `reports/training_summary.json` and should be read next to the
score.

The feature change also reordered the models. Before `merchant_state` was
removed, CatBoost led decisively (5.57% vs XGBoost's 0.72%) because its ordered
target statistics handle a 185-level categorical better than anything else here.
With that column gone, XGBoost leads (72.0% vs 61.9% on validation). CatBoost's
advantage was largely its skill at exploiting the shortcut.

---

## 4. A 5% subsample ranks hyper-parameters by noise

Searching on a stratified 5% sample of train leaves only ~508 frauds. A
configuration tuned on 508 positives scored **0.558%** where the same family on
full train scored **4.657%** — and worse than the absolute gap, the *ordering*
of configurations is unreliable at that size, because PR-AUC on a rare-event
problem is driven almost entirely by the positive count.

The search set therefore keeps **every fraud** and subsamples only the
negatives (10%). That is 740,467 rows with all 10,489 frauds: fast to search,
and faithful to how a configuration will rank at full scale. Every trial is
still scored on the **full** validation split for the same reason.

---

## 5. Velocity features do not help here, and the data says why

Every production fraud system leans on velocity: how many purchases this card
made in the last hour, how far this amount sits from its recent norm, whether
the merchant category is one it has never touched. This pipeline computes eight
of them causally (`src/prepare.py::add_velocity`, every window backward-looking
and excluding the row being scored).

They were then measured, with identical hyper-parameters, on validation:

| Feature set | val PR-AUC |
|---|---|
| 14 features, no velocity | **69.99%** |
| + `new_mcc_for_card` | 69.45% |
| + `new_mcc_for_card`, `amount_vs_card_recent` | 69.39% |
| all 8 velocity features | 69.09% |

Every addition made it slightly worse. That is not a bug, and the reason is
plain in the raw comparison of fraud against legitimate rows:

| Velocity feature | Fraud | Legitimate | Ratio |
|---|---|---|---|
| transactions in the prior hour | 0.18 | 0.18 | **0.99** |
| transactions in the prior 7 days | 7.28 | 8.43 | **0.86** |
| hours since the card was last used | 27.5 | 32.0 | 0.86 |
| amount vs the card's recent average | 2.60 | 1.69 | 1.54 |
| first use of this merchant category | 0.19 | 0.01 | 21.4 |

**A compromised card in the real world goes on a spree. Here it does not.**
Fraudulent cards are no busier than legitimate ones in the hour before the
event, and are *less* active over a week. This dataset's generator does not
simulate card takeover, so six of the eight features are noise the model has to
work around.

The one feature with genuine standalone signal, `new_mcc_for_card` (4.6x lift
alone), still does not improve the model — its information is already carried
by `mcc_category` and `is_online`.

**The features remain computed and stored in the parquet files but are excluded
from `FEATURES`.** On real payment traffic they would be among the most
valuable signals available, and the pipeline is built to move datasets. Here,
the honest thing is to measure them, report that they do not help, and not ship
complexity that buys nothing.

### What this says about pushing the score higher

The obvious levers were tried and measured:

- **More models / deeper stacking.** HistGB and XGBoost already agree at
  rho = 0.935 -- they are near-duplicates. The three-family blend measured
  -0.00% against the best single model, 95% CI [-0.51%, +0.47%]. More
  correlated members cannot help.
- **PCA.** Trees split on axis-aligned thresholds, so rotating the feature
  space makes splits worse, not better -- and it would destroy the named-feature
  explanations that are the point of this service.
- **Velocity engineering.** Measured above. No.

At 68.08% PR-AUC on a 0.176% base rate -- a 388x lift, catching 791 of 1,409
frauds while touching 236 legitimate customers out of 802,346 -- the model is
close to what these 14 features support. The remaining headroom is in the
DATA (richer signals a real processor has: device fingerprint, IP, session
behaviour, merchant history), not in more machinery on top of these columns.

---

## 6. Calibration costs 0.15pp of ranking, and is worth it

The served score is isotonic-calibrated so that a "0.56" means a 56% chance of
fraud -- without that the cost model, which multiplies the score by money, is
thresholding a number that does not mean what it says. It works: expected
calibration error falls from **0.645% to 0.0117%** on held-back validation.

It is not free. Isotonic regression is monotone, so it cannot reorder anything,
but its flat regions merge distinct scores into ties:

| | distinct score values |
|---|---|
| raw model output | 101,690 |
| after calibration | 8,328 |

Tied scores cannot be ranked against each other, so PR-AUC drops slightly --
measured at **78.895% -> 78.743%** on val_sel, about 0.15pp.

This matters for how results are reported. `src/assess.py` compares all
candidate models on their **uncalibrated** scores, so the table compares like
with like, and uses the **calibrated** score only for the shipped model's
operating point, because that is what the service actually thresholds.
Calibrating one model and not the others would have quietly reported the
shipped model as worse than its rivals for a reason that has nothing to do
with model quality.

---

## What is still uncertain

- **The remaining scores carry selection optimism.** The recency half-life and
  the model choice are both selected on validation, so validation numbers
  flatter. The number worth quoting is the held-out test result in
  `reports/assessment.json`, produced by a single pass in `src/assess.py`.
- **An aggressive half-life shrinks the effective sample.** A 180-day
  half-life over a 2010–2018 span means most rows contribute very little; the
  effective sample size is recorded in `reports/training_summary.json` and
  should be read alongside the score.
- **This dataset is at least partly synthetic.** Its patterns are cleaner than
  real payment traffic, and findings 2 and 3 are both artefacts of the
  generator rather than facts about the world. The *method* transfers; these
  particular numbers do not.
