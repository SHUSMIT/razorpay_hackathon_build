"""Central configuration.

Every tunable assumption lives here so it can be defended out loud rather than
appearing from nowhere inside a function. Two of them -- the false-positive cost
and the review capacity -- move the operating point directly, so they are
documented at length rather than left as bare numbers.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

for _d in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
MODEL_VERSION = "frm-1.0.0"

# ------------------------------------------------------------ machine budget
# This trains on a laptop, not a cluster, and the machine has to stay usable
# while it does. Both numbers are deliberately conservative and both can be
# raised from the environment on a bigger box:
#
#   FRM_THREADS   worker threads per model (default: leave half the logical
#                 cores free so the desktop stays responsive)
#   FRM_RAM_GB    memory ceiling handed to CatBoost, which is the only member
#                 that will happily consume everything available
N_THREADS = int(os.environ.get("FRM_THREADS") or
                max(2, (os.cpu_count() or 4) // 2))

# Leave at least this much RAM for the rest of the machine. Below it, the
# training stage subsamples rather than being killed by the OOM reaper --
# losing some rows beats losing the whole run.
RESERVE_RAM_GB = float(os.environ.get("FRM_RESERVE_RAM_GB") or 2.0)
CATBOOST_RAM_LIMIT = os.environ.get("FRM_RAM_GB") or "2gb"
# CatBoost builds combination statistics across categorical columns. With 185
# merchant states and 108 merchant categories the default complexity of 4 is a
# memory bomb on 7M rows, and buys very little here.
CATBOOST_MAX_CTR_COMPLEXITY = 1


def available_ram_gb() -> float | None:
    """Free physical RAM, or None when it cannot be determined."""
    try:
        if os.name == "nt":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _Status()
            st.dwLength = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return st.ullAvailPhys / (1024 ** 3)
            return None
        return (os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / (1024 ** 3)
    except Exception:  # noqa: BLE001 - a probe must never break training
        return None

# ------------------------------------------------------------- cost modelling
# ASSUMPTION, stated explicitly and defended in the README and the pitch:
#
#   A FALSE NEGATIVE (fraud we let through) costs the merchant the full value
#   of that transaction. We charge the ACTUAL amount of each missed
#   transaction, not an average, because the fraud amount distribution is
#   skewed and an average would flatter us.
#
#   A FALSE POSITIVE (legitimate payment we block) costs a fixed friction
#   penalty: one support contact plus the goodwill and abandonment cost of a
#   customer declined at checkout. We set this at 50 currency units. It is a
#   stated constant, not a fitted parameter -- change it here and the
#   cost-optimal threshold moves, which is exactly the sensitivity a merchant
#   should be shown.
FP_COST = 50.0
FN_COST_MODE = "amount"   # "amount" (per-transaction) or "average"
CHARGEBACK_FEE = 0.0      # optional flat fee on top of each missed fraud

# --------------------------------------------------------------- decisioning
# The review band is sized by REVIEW CAPACITY, not by a factor of the block
# threshold. A merchant's review team can look at a fixed share of traffic per
# day; that operational limit is what should decide the band's width. We target
# 0.2% of transactions and read the score that produces it off held-out
# validation predictions.
#
# This is not cosmetic. Under the old "half the block threshold" rule the band
# caught 7 of 42,559 transactions -- 0.016% -- so the human queue was
# decorative and essentially every review came from a gate downgrade.
REVIEW_CAPACITY_RATE = 0.002

# Bounded / gated decisioning: the service will never auto-block more than this
# share of traffic in the rolling window. Beyond it, blocks are force-downgraded
# to "review" and the downgrade is logged with a reason.
MAX_BLOCK_RATE = 0.05     # 5% of the rolling window
ROLLING_WINDOW = 200      # decisions considered when measuring the block rate
# Floor for the rate denominator on a cold window. NOT an exemption: the gate is
# active from the first request. With a floor of 20 and a 5% cap the first block
# is admitted (1/20 == 5%) but a second consecutive one is held, so genuine
# fraud is still blocked at startup while a burst cannot slip through before the
# window fills.
GATE_MIN_SAMPLE = 20

# ----------------------------------------------------------------- artefacts
SERVING_CONFIG = MODELS / "serving_config.json"
FEATURE_CONTEXT = MODELS / "feature_context.json"
CATEGORIES = MODELS / "categories.json"
AUDIT_LOG = REPORTS / "audit_log.jsonl"
REVIEW_QUEUE = REPORTS / "review_queue.jsonl"
# Optuna studies live in SQLite so an interrupted search RESUMES instead of
# starting from trial zero. On a memory-tight machine that is not a nicety.
OPTUNA_DB = REPORTS / "optuna.db"
VAL_PRED_DIR = MODELS / "val_preds"
VAL_PRED_DIR.mkdir(parents=True, exist_ok=True)
