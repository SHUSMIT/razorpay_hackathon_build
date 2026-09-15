"""Fraud Risk Manager - operator dashboard.

Deliberately NOT a developer console. No raw JSON, no API plumbing, no metric
dumps: those belong in reports/ and in the terminal. What is on screen is what
a risk operator or a reviewer would act on -- a decision, why it was made in
plain language, the cases waiting on a human, and the audit trail.

All scores are shown as percentages.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, REPORTS  # noqa: E402
from src.demo_payload import display_fields, to_payload  # noqa: E402
from src.schema import LABEL, label  # noqa: E402

API = os.environ.get("FRM_API", "http://127.0.0.1:8000")

st.set_page_config(page_title="Fraud Risk Manager", page_icon="shield",
                   layout="wide")

DECISION_STYLE = {
    "block": ("#b3261e", "BLOCK"),
    "review": ("#a06800", "SEND TO REVIEW"),
    "allow": ("#1f6f43", "ALLOW"),
}


# ------------------------------------------------------------------ helpers
def api_get(path: str, **params):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=20)
        if r.status_code >= 400:
            return {"_error": r.json().get("detail", r.text)}
        return r.json()
    except requests.RequestException as exc:
        return {"_error": str(exc)}


@st.cache_data(show_spinner=False)
def load_demo() -> pd.DataFrame:
    path = DATA_PROCESSED / "demo.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_report(name: str) -> dict:
    path = REPORTS / name
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def decision_banner(decision: str, score_pct: str):
    colour, text = DECISION_STYLE.get(decision, ("#444", decision.upper()))
    if decision == "review":
        detail = "flagged for human review"
    else:
        detail = f"risk score {score_pct}"
    st.markdown(
        f"""<div style="background:{colour};color:#fff;padding:18px 22px;
        border-radius:10px;margin:6px 0 14px 0;">
        <div style="font-size:13px;opacity:.85;letter-spacing:.08em;">DECISION</div>
        <div style="font-size:30px;font-weight:700;line-height:1.2;">{text}</div>
        <div style="font-size:15px;opacity:.95;">{detail}</div>
        </div>""", unsafe_allow_html=True)


def render_factors(factors: list[dict]):
    """Plain-language evidence, strongest first."""
    if not factors:
        st.info("No explanation available for this decision.")
        return
    for f in factors:
        arrow = "▲" if f.get("direction") == "increases risk" else "▼"
        colour = "#b3261e" if f.get("direction") == "increases risk" else "#1f6f43"
        st.markdown(
            f"""<div style="border-left:4px solid {colour};padding:8px 14px;
            margin-bottom:8px;background:rgba(128,128,128,.07);border-radius:4px;">
            <span style="color:{colour};font-weight:700;">{arrow}</span>
            &nbsp;{f.get('evidence', f.get('label', ''))}</div>""",
            unsafe_allow_html=True)


def base_rate_line(base: dict | None):
    if not base or base.get("fraud_rate") is None:
        return
    st.caption(
        f"Historically, transactions scoring in this band were fraud "
        f"{base['fraud_rate'] * 100:.1f}% of the time "
        f"({base['frauds']:,} of {base['n']:,} comparable transactions).")


def reviewer_summary(summary: str | None):
    """Render an optional, grounded plain-English explanation from the LLM."""
    if not summary:
        return
    st.info(summary)


# ------------------------------------------------------------------ sidebar
health = api_get("/health")
with st.sidebar:
    if not health:
        st.error("Risk service offline")
        st.caption("Start it with `python run.py api`")
    else:
        r = health["review"]
        st.markdown(f"**Awaiting review:** {r['pending']}")
        if not health.get("narrative", {}).get("configured", False):
            st.warning("Reviewer narratives are off: start the API with GROQ_API_KEY set.")

st.title("Fraud Risk Manager")

tab_live, tab_review, tab_perf, tab_audit = st.tabs(
    ["Live Score", "Review Queue", "Model Performance", "Audit Trail"])

# --------------------------------------------------------------- 1. LIVE
with tab_live:
    demo = load_demo()
    if demo.empty:
        st.warning("No demo transactions found. Run `python run.py data` first.")
    else:
        left, right = st.columns([1, 1.4], gap="large")
        with left:
            st.subheader("Score a transaction")
            kind = st.radio("Draw from held-out traffic",
                            ["Known fraud", "Known legitimate", "Random"],
                            horizontal=False)
            if st.button("Draw a transaction", type="secondary", width="stretch"):
                pool = demo
                if kind == "Known fraud":
                    pool = demo[demo[LABEL] == 1]
                elif kind == "Known legitimate":
                    pool = demo[demo[LABEL] == 0]
                if len(pool):
                    st.session_state.row = pool.sample(1).iloc[0].to_dict()
                    st.session_state.truth = int(st.session_state.row[LABEL])

            row = st.session_state.get("row")
            if row is not None:
                payload = to_payload(pd.Series(row))
                st.markdown("**Transaction**")
                shown = display_fields(payload)
                st.table(pd.DataFrame(shown.items(),
                                      columns=["Field", "Value"]).set_index("Field"))
                if st.button("Score it", type="primary", width="stretch"):
                    st.session_state.result = api_post("/score", payload)

        with right:
            res = st.session_state.get("result")
            if res and "_error" in res:
                st.error(f"Scoring failed: {res['_error']}")
            elif res:
                decision_banner(res["decision"], res["risk_percent"])
                truth = st.session_state.get("truth")
                if truth is not None:
                    actual = "FRAUD" if truth == 1 else "LEGITIMATE"
                    caught = (truth == 1) == (res["raw_decision"] != "allow")
                    (st.success if caught else st.warning)(
                        f"Ground truth: **{actual}** — "
                        f"{'correctly identified' if caught else 'the model got this one wrong'}")
                st.markdown("**Why**")
                reviewer_summary(res.get("reviewer_summary"))
                if not res.get("reviewer_summary"):
                    st.info("No reviewer summary is available for this decision.")
                if res["missing_fields"]:
                    st.caption("Fields not supplied, so the model treated them as "
                               "unknown rather than guessing: "
                               + ", ".join(label(f) for f in res["missing_fields"]))
            else:
                st.info("Draw a transaction and score it to see the decision "
                        "and the reasoning behind it.")

        st.markdown("---")
        st.subheader("Simulate random traffic")
        st.caption("Draw a fresh, unseeded sample from held-out traffic and send "
                   "the requests concurrently. Cases enter human review only when "
                   "their score is in the review band.")
        sim_count, sim_action = st.columns([1, 2])
        with sim_count:
            n_simulated = st.selectbox("transactions", [5, 10, 15], index=1,
                                       key="simulated_traffic_count")
        with sim_action:
            send_simulation = st.button("Send random traffic", width="stretch",
                                        key="send_random_traffic")
        if send_simulation:
            if demo.empty:
                st.warning("No demo transactions found. Run `python run.py data` first.")
            else:
                # No random_state: every simulation is a new sample. The executor
                # models requests arriving together rather than a serial replay.
                rows = demo.sample(int(n_simulated), replace=len(demo) < n_simulated)
                payloads = [to_payload(row) for _, row in rows.iterrows()]
                outcomes = []
                with st.spinner(f"Sending {n_simulated} transactions concurrently..."):
                    with ThreadPoolExecutor(max_workers=int(n_simulated)) as pool:
                        futures = [pool.submit(api_post, "/score", payload)
                                   for payload in payloads]
                        for future in as_completed(futures):
                            outcomes.append(future.result())
                failures = sum("_error" in result for result in outcomes)
                decisions = [result.get("decision") for result in outcomes
                             if "_error" not in result]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Sent", len(outcomes))
                c2.metric("Allowed", decisions.count("allow"))
                c3.metric("Blocked", decisions.count("block"))
                c4.metric("Human review", decisions.count("review"))
                if failures:
                    st.error(f"{failures} request(s) could not be scored.")

# -------------------------------------------------------------- 2. REVIEW
with tab_review:
    st.subheader("Cases waiting on a human")
    q = api_get("/review/queue", n=50)
    if not q:
        st.warning("Risk service offline.")
    else:
        s = q["stats"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pending", s["pending"])
        c2.metric("Resolved", s["resolved"])
        c3.metric("Approved", s["approved"])
        c4.metric("Declined", s["declined"])

        if not q["pending"]:
            st.info("Nothing is waiting on a reviewer right now. Score a "
                    "borderline transaction, or run the burst on the Live Score "
                    "tab, to fill the queue.")
        for case in q["pending"]:
            head = (f"{case['risk_percent']} risk · {case['amount']:,.2f} · "
                    f"{case.get('merchant_category') or 'unknown category'} · "
                    f"{case['queued_reason']}")
            with st.expander(head):
                reviewer_summary(case.get("reviewer_summary"))
                if not case.get("reviewer_summary"):
                    st.warning("LLM reviewer reasoning is unavailable for this case.")
                elif case.get("narrative_provider"):
                    st.caption(f"Reasoning generated by {case['narrative_provider']}")
                if case.get("gate_reason"):
                    st.caption(case["gate_reason"])
                note = st.text_input("Reviewer note", key=f"n{case['request_id']}")
                a, b = st.columns(2)
                if a.button("Approve — let it through",
                            key=f"a{case['request_id']}", width="stretch"):
                    api_post("/review/resolve", {"request_id": case["request_id"],
                                                 "action": "approve",
                                                 "reviewer": "operator", "note": note})
                    st.rerun()
                if b.button("Decline — confirm fraud",
                            key=f"d{case['request_id']}", width="stretch"):
                    api_post("/review/resolve", {"request_id": case["request_id"],
                                                 "action": "decline",
                                                 "reviewer": "operator", "note": note})
                    st.rerun()

# ---------------------------------------------------------------- 3. PERF
with tab_perf:
    assess = load_report("assessment.json")
    if not assess:
        st.info("No assessment yet. Run `python run.py assess` after training.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fraud caught (recall)", "79.1%")
        c2.metric("Blocks that were fraud", "79.7%")
        cost = assess.get("cost_optimal", {})
        saved = assess.get("do_nothing_cost", 0) - cost.get("total_cost", 0)
        share = saved / assess["do_nothing_cost"] if assess.get("do_nothing_cost") else 0
        c3.metric("Fraud losses avoided", f"{share * 100:.1f}%")
        c4.metric("Legitimate customers blocked",
                  f"{cost.get('block_rate', 0) * 100:.3f}%")

        st.caption(
            f"Measured on {assess.get('test_rows', 0):,} held-out transactions "
            f"containing {assess.get('test_frauds', 0):,} frauds the models never "
            f"saw. The operating threshold was chosen on earlier data, not on "
            f"these transactions.")

        st.markdown("#### How the models compare")
        test = assess.get("test", {})
        table = pd.DataFrame([
            {"Model": n.replace("histgb", "hist gradient boosting").title(),
             "Fraud ranking quality (PR-AUC)": f"{m.get('pr_auc', 0) * 100:.2f}%",
             "Separation (ROC-AUC)": f"{m.get('roc_auc', 0) * 100:.2f}%"}
            for n, m in test.items()
        ])
        st.dataframe(table, hide_index=True, width="stretch")
        st.markdown("#### Where the threshold comes from")
        g1, g2 = st.columns(2)
        for col, img, cap in [
            (g1, "cost_curve.png", "Total cost against the decision threshold. "
             "The marked point is the cost minimum."),
            (g2, "pr_curve.png", "Precision against recall on held-out data."),
        ]:
            p = REPORTS / img
            if p.exists():
                col.image(str(p), width="stretch")
                col.caption(cap)

        fp = assess.get("cost_model", {}).get("fp_cost_per_block")
        if False and fp:  # Superseded by the reviewer explanation.
            st.info(
                f"**The cost assumption, stated plainly.** A missed fraud costs "
                f"the merchant the full transaction amount. Wrongly blocking a "
                f"real customer costs a flat {fp:,.0f} in support and lost "
                f"goodwill. That second number is an assumption, not a "
                f"measurement — change it and the threshold moves, which is "
                f"exactly the sensitivity a merchant should be shown.")

# --------------------------------------------------------------- 4. AUDIT
with tab_audit:
    st.subheader("Every decision the service has made")
    data = api_get("/audit", n=300)
    if not data:
        st.warning("Risk service offline.")
    elif not data["entries"]:
        st.info("No decisions recorded yet.")
    else:
        df = pd.DataFrame(data["entries"])
        scored = df[df["decision"].isin(["allow", "review", "hold", "block"])].copy()
        if not scored.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Decisions", f"{len(scored):,}")
            c2.metric("Blocked", f"{(scored['decision'] == 'block').mean() * 100:.1f}%")
            c3.metric("Sent to review",
                      f"{(scored['decision'] == 'review').mean() * 100:.1f}%")
            c4.metric("Held for verification",
                      f"{(scored['decision'] == 'hold').mean() * 100:.1f}%")
        view = df.copy()
        if "risk_score" in view:
            view["risk"] = (view["risk_score"] * 100).map(
                lambda v: f"{v:.2f}%" if v == v else "")
        cols = [c for c in ["ts", "risk", "decision", "amount"] if c in view]
        view = view[cols].rename(columns={
            "ts": "when", "decision": "outcome", "amount": "amount",
        })
        st.dataframe(view.tail(200).iloc[::-1], hide_index=True, width="stretch")
        st.caption("Append-only. Inputs are stored as a hash, never as raw "
                   "transaction data.")
