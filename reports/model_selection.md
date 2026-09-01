# Model selection — what we shipped and why

## Protocol (fixed before any test number was seen)

1. Select on the **validation** split.
2. Evaluate the selected model **once** on test.
3. Report that number whatever it says.

## Step 1 — selection (validation)

| model | validation PR-AUC | note |
|---|---:|---|
| baseline | 0.8451 | held-out during training (early stopping) |
| tuned (best Optuna trial) | 0.8532 | held-out during the search |

The tuned configuration won on validation (0.8532 vs 0.8451), so **the tuned model is what we ship**. That decision was made before looking at test.

> Note: the shipped tuned model was refit on train+val, so its own validation score (1.0000) is in-sample and meaningless as a comparison. The selection number above is the best trial's score during the search, when validation was genuinely held out.

## Steps 2–3 — the one look at test, reported honestly

| model | test PR-AUC | 95% CI |
|---|---:|---|
| baseline | 0.8150 | [0.7295, 0.8923] |
| tuned (shipped) | 0.8061 | [0.7188, 0.8907] |

**Tuning did not improve test PR-AUC.** The point estimate moved -0.0090, the wrong way.

### Is that difference real?

Paired bootstrap, 2,000 resamples, both models scored on the same resampled rows each draw:

- difference (tuned − baseline): **-0.0090**
- 95% CI: **[-0.0387, +0.0221]**
- tuned better in 29.6% of resamples
- significant at 95%: **no**

The interval straddles zero, so **the two models are statistically indistinguishable on this test split**. That is the correct conclusion, not a disappointing one: the split holds only 71 frauds, and the confidence interval on a single PR-AUC is roughly ±0.08 wide. Quoting a four-decimal improvement off 71 positives would be claiming precision the data cannot support.

**What tuning did buy**, visible at the operating point rather than in the summary metric: the tuned model is more conservative. It declines fewer legitimate customers (3 false positives at its cost-optimal threshold, against the baseline's 4) at the cost of missing two more frauds — a trade the cost model prices as roughly neutral.

**The honest takeaway:** on a dataset with this few positives, 50 trials of hyperparameter search buys almost nothing that survives contact with held-out data. More labelled fraud would help far more than more search. We report this rather than running the search repeatedly until a seed happened to make the table look better — which, on noise this size, it eventually would.
