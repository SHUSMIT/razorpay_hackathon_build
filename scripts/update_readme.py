#!/usr/bin/env python
"""Regenerate the metrics block in README.md from the artefacts on disk.

The README states numbers, and numbers go stale the moment the pipeline is
re-run. Everything between the RESULTS markers is therefore generated from
reports/assessment.json and reports/training_summary.json rather than typed by
hand, so the README cannot drift away from what was actually measured.

Run: python scripts/update_readme.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import REPORTS  # noqa: E402
from src.fmt import money, pct  # noqa: E402

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _load(name: str) -> dict:
    path = REPORTS / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_block() -> str:
    a = _load("assessment.json")
    t = _load("training_summary.json")
    if not a:
        return ("_No assessment yet. Run `python run.py all`, then "
                "`python scripts/update_readme.py`._")

    shipped = a.get("shipped", "?")
    test = a.get("test", {})
    m = test.get(shipped, {})
    base = a.get("test_random_baseline", 0.0)
    cost = a.get("cost_optimal", {})
    nothing = a.get("do_nothing_cost", 0.0)
    saved = nothing - cost.get("total_cost", 0.0)
    ci = m.get("pr_auc_ci", {})
    lines = []

    lines.append(f"**Shipped model: `{shipped}`**, selected on validation and "
                 f"evaluated once on a held-out test split of "
                 f"{a.get('test_rows', 0):,} transactions containing "
                 f"{a.get('test_frauds', 0):,} frauds "
                 f"({pct(base, 3)} of traffic).\n")

    lines.append("| Metric | Held-out test |")
    lines.append("|---|---|")
    lines.append(f"| PR-AUC | **{pct(m.get('pr_auc'))}** "
                 f"(95% CI {pct(ci.get('ci_low'))} – {pct(ci.get('ci_high'))}) |")
    lines.append(f"| Lift over random | "
                 f"{m.get('pr_auc', 0) / max(base, 1e-12):,.0f}x |")
    lines.append(f"| ROC-AUC | {pct(m.get('roc_auc'))} |")
    lines.append(f"| Fraud caught (recall) | {pct(cost.get('recall'))} |")
    lines.append(f"| Blocks that were fraud (precision) | {pct(cost.get('precision'))} |")
    lines.append(f"| Legitimate traffic blocked | {pct(cost.get('block_rate'), 3)} |")
    lines.append("")

    lines.append("**Cost outcome.** A missed fraud costs the merchant the full "
                 "transaction amount; a wrongly blocked customer costs a flat "
                 f"{money(a.get('cost_model', {}).get('fp_cost_per_block'))} in "
                 "support and lost goodwill.\n")
    lines.append("| | Cost on the test split |")
    lines.append("|---|---|")
    lines.append(f"| Do nothing | {money(nothing)} |")
    lines.append(f"| This service | {money(cost.get('total_cost'))} |")
    lines.append(f"| **Avoided** | **{money(saved)}** "
                 f"({pct(saved / nothing if nothing else 0, 1)}) |")
    lines.append("")

    band = a.get("review_band", [])
    if len(band) == 2:
        lines.append(f"Operating point: block at **{pct(band[1])}**, human review "
                     f"from **{pct(band[0])}**. Both chosen on a slice of "
                     f"validation that the models, the blend weights and the "
                     f"calibrator never saw.\n")

    bias = a.get("selection_bias")
    if bias:
        lines.append(f"For honesty: picking the threshold on the test split "
                     f"itself would have looked {money(abs(bias))} better. That "
                     f"is selection bias, and it is not used.\n")

    if test:
        lines.append("**All candidates on the held-out split**\n")
        lines.append("| Model | PR-AUC | ROC-AUC |")
        lines.append("|---|---|---|")
        for n, mm in sorted(test.items(), key=lambda kv: -kv[1].get("pr_auc", 0)):
            mark = " (shipped)" if n == shipped else ""
            lines.append(f"| {n}{mark} | {pct(mm.get('pr_auc'))} | "
                         f"{pct(mm.get('roc_auc'))} |")
        lines.append("")

    chk = a.get("check_set") or {}
    if chk:
        lines.append("**Independent 1% check set** — the month *after* the test "
                     "period, never touched by any stage of the pipeline:")
        lines.append("")
        lines.append("| | Check set |")
        lines.append("|---|---|")
        lines.append(f"| Transactions | {chk.get('rows', 0):,} "
                     f"({chk.get('frauds', 0):,} fraud) |")
        lines.append(f"| PR-AUC | {pct(chk.get('pr_auc'))} |")
        c = chk.get("cost_at_selected_threshold", {})
        lines.append(f"| Recall at the shipped threshold | {pct(c.get('recall'))} |")
        lines.append(f"| Precision at the shipped threshold | {pct(c.get('precision'))} |")
        lines.append("")

    cmp_ = a.get("ensemble_vs_best_single") or {}
    if cmp_:
        verdict = ("a statistically significant improvement"
                   if cmp_.get("significant_at_95")
                   else "**not** a statistically significant improvement")
        lines.append(f"The blend is {verdict} over the best single model "
                     f"(`{cmp_.get('best_single')}`): paired bootstrap delta "
                     f"{pct(cmp_.get('delta_point'))}, 95% CI "
                     f"{pct(cmp_.get('ci_low'))} – {pct(cmp_.get('ci_high'))}.\n")

    cal = a.get("calibration", {})
    if cal.get("method") and cal["method"] != "none":
        lines.append(f"Scores are {cal['method']}-calibrated: expected "
                     f"calibration error {pct(cal.get('ece_before'), 4)} → "
                     f"{pct(cal.get('ece_after'), 4)} on held-back validation.\n")

    for row in t.get("models", []):
        if row.get("subsampled"):
            lines.append(f"> Note: `{row['model']}` was fitted on "
                         f"{row['rows_used']:,} rows rather than the full split "
                         f"because memory ran short. All "
                         f"{row.get('frauds_used', 0):,} frauds were kept.\n")
        if row.get("recency_half_life_days"):
            lines.append(f"> `{row['model']}` training weights halve every "
                         f"{row['recency_half_life_days']} days "
                         f"(effective sample size "
                         f"{row.get('effective_sample_size', 0):,.0f} rows).\n")
    return "\n".join(lines)


def main() -> int:
    readme = ROOT / "README.md"
    if not readme.exists():
        print("README.md not found", file=sys.stderr)
        return 1
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print(f"markers {START} / {END} not found in README.md", file=sys.stderr)
        return 1
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{build_block()}\n{END}{tail}",
                      encoding="utf-8")
    print("[readme] metrics block regenerated from reports/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
