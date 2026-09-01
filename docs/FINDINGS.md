# What the data actually said

Four things were measured during this build that changed the model, and each
one would have quietly inflated the headline number if it had gone unchecked.
They are recorded here with the evidence, because "we checked and it was fine"
is not a claim anyone should take on trust.

Every number below is reproducible from the code in this repository.

---

## 1. A random split scores 4x higher — and it is a lie

The dataset covers 2010–2019. Split it at random and the model scores **61%
PR-AUC**. Split it by time, so evaluation only ever looks forward, and the
same model scores **0.089%** — barely above the 0.163% base rate.

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
hyper-parameter, from the candidates {0 (off), 180, 365, 730, 1095} days.

Measured on validation, identical features:

| Training weights | val PR-AUC |
|---|---|
| Class weights only | 15.18% |
| Class weights x recency decay | **46.00%** |

The tuner independently selected the most aggressive decay available, which is
itself evidence about how fast this data goes stale.

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
