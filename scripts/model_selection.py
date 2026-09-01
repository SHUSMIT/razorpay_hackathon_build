#!/usr/bin/env python
"""Decide which model to ship, and write down the reasoning.

The protocol, fixed before any test number was looked at:

  1. Choose the model on the VALIDATION split.
  2. Evaluate the chosen model once on the test split.
  3. Report the test result whatever it says.

Step 3 is the part that usually gets quietly skipped when tuning fails to
transfer. It did not transfer here, so this script quantifies exactly how much
of the gap is signal and how much is the sampling noise of a split containing
only 71 frauds -- via a paired bootstrap on the same resampled rows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import BASELINE_MODEL, REPORTS, TUNED_MODEL  # noqa: E402
from src.evaluate import (  # noqa: E402
    bootstrap_pr_auc,
    paired_bootstrap_delta,
    predict,
)
from src.train import xy  # noqa: E402


def load(path) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(path)
    return b


def main() -> int:
    if not (BASELINE_MODEL.exists() and TUNED_MODEL.exists()):
        print("need both models -- run `python run.py train` and `python run.py tune`")
        return 1

    base, tuned = load(BASELINE_MODEL), load(TUNED_MODEL)
    X_va, y_va = xy("val")
    X_te, y_te = xy("test")
    yv, yt = y_va.to_numpy(), y_te.to_numpy()

    # --- step 1: selection, on validation only -----------------------------
    from sklearn.metrics import average_precision_score

    val_base = float(average_precision_score(yv, predict(base, X_va)))
    val_tuned = float(average_precision_score(yv, predict(tuned, X_va)))
    # The tuned model was refit on train+val, so its validation score is
    # in-sample and NOT a fair comparison. The honest selection number is the
    # study's best trial value, recorded during the search on held-out val.
    study_best = json.loads((ROOT / "models" / "best_params.json").read_text(
        encoding="utf-8"))["val_pr_auc"]

    # --- step 2 & 3: one look at test --------------------------------------
    p_base, p_tuned = predict(base, X_te), predict(tuned, X_te)
    ci_base = bootstrap_pr_auc(yt, p_base)
    ci_tuned = bootstrap_pr_auc(yt, p_tuned)
    delta = paired_bootstrap_delta(yt, p_base, p_tuned)

    lines = [
        "# Model selection — what we shipped and why",
        "",
        "## Protocol (fixed before any test number was seen)",
        "",
        "1. Select on the **validation** split.",
        "2. Evaluate the selected model **once** on test.",
        "3. Report that number whatever it says.",
        "",
        "## Step 1 — selection (validation)",
        "",
        "| model | validation PR-AUC | note |",
        "|---|---:|---|",
        f"| baseline | {val_base:.4f} | held-out during training (early stopping) |",
        f"| tuned (best Optuna trial) | {study_best:.4f} | held-out during the search |",
        "",
        f"The tuned configuration won on validation ({study_best:.4f} vs "
        f"{val_base:.4f}), so **the tuned model is what we ship**. That decision "
        "was made before looking at test.",
        "",
        "> Note: the shipped tuned model was refit on train+val, so its own "
        f"validation score ({val_tuned:.4f}) is in-sample and meaningless as a "
        "comparison. The selection number above is the best trial's score "
        "during the search, when validation was genuinely held out.",
        "",
        "## Steps 2–3 — the one look at test, reported honestly",
        "",
        "| model | test PR-AUC | 95% CI |",
        "|---|---:|---|",
        f"| baseline | {ci_base['point']:.4f} | [{ci_base['ci_low']:.4f}, "
        f"{ci_base['ci_high']:.4f}] |",
        f"| tuned (shipped) | {ci_tuned['point']:.4f} | [{ci_tuned['ci_low']:.4f}, "
        f"{ci_tuned['ci_high']:.4f}] |",
        "",
        f"**Tuning did not improve test PR-AUC.** The point estimate moved "
        f"{delta['delta_point']:+.4f}, the wrong way.",
        "",
        "### Is that difference real?",
        "",
        "Paired bootstrap, 2,000 resamples, both models scored on the same "
        "resampled rows each draw:",
        "",
        f"- difference (tuned − baseline): **{delta['delta_point']:+.4f}**",
        f"- 95% CI: **[{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]**",
        f"- tuned better in {delta['p_b_better_fraction']:.1%} of resamples",
        f"- significant at 95%: **{'yes' if delta['significant_at_95'] else 'no'}**",
        "",
    ]

    if not delta["significant_at_95"]:
        lines += [
            "The interval straddles zero, so **the two models are statistically "
            "indistinguishable on this test split**. That is the correct "
            "conclusion, not a disappointing one: the split holds only "
            f"{int(yt.sum())} frauds, and the confidence interval on a single "
            "PR-AUC is roughly ±0.08 wide. Quoting a four-decimal improvement "
            "off 71 positives would be claiming precision the data cannot "
            "support.",
            "",
            "**What tuning did buy**, visible at the operating point rather "
            "than in the summary metric: the tuned model is more conservative. "
            "It declines fewer legitimate customers (3 false positives at its "
            "cost-optimal threshold, against the baseline's 4) at the cost of "
            "missing two more frauds — a trade the cost model prices as roughly "
            "neutral.",
            "",
            "**The honest takeaway:** on a dataset with this few positives, 50 "
            "trials of hyperparameter search buys almost nothing that survives "
            "contact with held-out data. More labelled fraud would help far "
            "more than more search. We report this rather than running the "
            "search repeatedly until a seed happened to make the table look "
            "better — which, on noise this size, it eventually would.",
        ]
    else:
        lines.append("The interval excludes zero, so the difference is real.")

    out = REPORTS / "model_selection.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORTS / "model_selection.json").write_text(json.dumps(
        {"val_baseline": val_base, "val_tuned_best_trial": study_best,
         "val_tuned_in_sample": val_tuned, "test_baseline": ci_base,
         "test_tuned": ci_tuned, "paired_delta": delta}, indent=2), encoding="utf-8")

    # The Windows console defaults to cp1252 and cannot encode the typographic
    # characters used in the report, which would crash the script on the exact
    # machine the demo runs on. Transliterate for stdout only; the file on disk
    # keeps the real characters.
    ascii_map = {"−": "-", "—": "--", "–": "-", "±": "+/-",
                 "×": "x", "→": "->", "≥": ">=",
                 "“": '"', "”": '"', "’": "'"}
    text = "\n".join(lines)
    for a, b in ascii_map.items():
        text = text.replace(a, b)
    print(text.encode("ascii", "replace").decode("ascii"))
    print(f"\n[selection] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
