#!/usr/bin/env python
"""Regenerate the README metrics table from reports/tuning_comparison.json.

Keeps the numbers in the README provably identical to the numbers the pipeline
actually produced -- no hand-typed metrics that quietly go stale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

START = "<!-- METRICS_TABLE_START -->"
END = "<!-- METRICS_TABLE_END -->"


def build_table(data: dict) -> str:
    b, t = data["baseline"], data["tuned"]
    bc, tc = b["cost_optimal"], t["cost_optimal"]
    rows = [
        ("PR-AUC", f"{b['pr_auc']:.4f}", f"**{t['pr_auc']:.4f}**",
         "headline metric; random baseline 0.0017"),
        ("ROC-AUC", f"{b['roc_auc']:.4f}", f"{t['roc_auc']:.4f}",
         "reported for comparability, not relied on"),
        ("Precision @ 0.50", f"{b['precision']:.4f}", f"{t['precision']:.4f}",
         "share of blocks that were really fraud"),
        ("Recall @ 0.50", f"{b['recall']:.4f}", f"{t['recall']:.4f}",
         "share of fraud caught"),
        ("F1 @ 0.50", f"{b['f1']:.4f}", f"{t['f1']:.4f}", ""),
        ("**Operating threshold**", f"{bc['threshold']:.2f}",
         f"**{tc['threshold']:.2f}**",
         "min of the cost curve **on validation**, not 0.5, not test"),
        ("Precision at that threshold", f"{bc['precision']:.4f}",
         f"**{tc['precision']:.4f}**", ""),
        ("Recall at that threshold", f"{bc['recall']:.4f}",
         f"**{tc['recall']:.4f}**", ""),
        ("False positives there", f"{int(bc['fp'])}", f"**{int(tc['fp'])}**",
         "good customers wrongly declined"),
        ("Missed frauds there", f"{int(bc['fn'])}", f"**{int(tc['fn'])}**",
         f"out of {t['n_fraud']} in the split"),
        ("Cost at that threshold", f"{bc['total_cost']:,.0f}",
         f"**{tc['total_cost']:,.0f}**",
         f"vs {t['do_nothing_cost']:,.0f} doing nothing"),
    ]
    saved = t["do_nothing_cost"] - tc["total_cost"]
    pct = 100 * saved / t["do_nothing_cost"]

    lines = [
        f"Held-out test split: **{t['n_rows']:,} transactions, {t['n_fraud']} frauds**. "
        f"Optuna: {data['n_trials']} TPE trials, best validation PR-AUC "
        f"{data['best_val_pr_auc']:.4f}.",
        "",
        "| metric | baseline | tuned | note |",
        "|---|---:|---:|---|",
    ]
    lines += [f"| {n} | {bv} | {tv} | {note} |" for n, bv, tv, note in rows]
    lines += [
        "",
        f"**Bottom line.** At the operating threshold the tuned model catches "
        f"{tc['recall']:.1%} of fraud while wrongly declining **{int(tc['fp'])}** of "
        f"{t['n_rows'] - t['n_fraud']:,} legitimate customers. In the cost model that "
        f"is **{tc['total_cost']:,.0f}** lost against **{t['do_nothing_cost']:,.0f}** "
        f"for doing nothing — a **{pct:.1f}%** reduction in fraud losses.",
        "",
        "",
        "The threshold is selected on the **validation** split and merely applied "
        "to test. Taking the argmin on test itself would report "
        f"{t['cost_oracle_test_argmin']['total_cost']:,.0f} instead of "
        f"{tc['total_cost']:,.0f} — an "
        f"{100 * t['selection_bias'] / tc['total_cost']:.1f}% selection bias we "
        "do not claim.",
        "",
        "Full before/after table: `reports/tuning_comparison.txt`. "
        "Curves: `reports/pr_curve.png`, `reports/cost_curve.png`. "
        "Limitations: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).",
    ]
    return "\n".join(lines)


def main() -> int:
    src = ROOT / "reports" / "tuning_comparison.json"
    if not src.exists():
        print(f"{src} not found -- run `python run.py tune` first")
        return 1
    table = build_table(json.loads(src.read_text(encoding="utf-8")))

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print("README markers missing")
        return 1
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{table}\n{END}{tail}", encoding="utf-8")
    print("README metrics table updated")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
