"""Central configuration. Every tunable assumption lives here so it can be
defended out loud rather than appearing from nowhere inside a function."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SAMPLE = ROOT / "data" / "sample"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

for _d in (DATA_RAW, DATA_PROCESSED, DATA_SAMPLE, MODELS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data source
# The ULB / Worldline "creditcardfraud" benchmark, mirrored on the HF Hub.
# 284,807 transactions, 492 frauds (0.1727%). Features V1..V28 are PCA
# components published by the original authors; Time and Amount are raw.
HF_DATASET_REPO = "TekMonger/creditcard-fraud"
HF_DATASET_FILE = "creditcard.csv"
HF_DATASET_MIRROR = "megloughney/creditcardfraud"  # identical file, fallback
EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492

RANDOM_SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ------------------------------------------------------------- cost modelling
# ASSUMPTION (stated explicitly, defended in the README and the pitch):
#
#   False negative (fraud we let through) costs the merchant the full value of
#   that transaction -- the chargeback claws back the amount. We charge the
#   ACTUAL amount of each missed transaction, not a flat average, because the
#   fraud amount distribution is heavily skewed and an average would flatter us.
#
#   False positive (legitimate payment we block) costs a fixed friction
#   penalty: one support contact plus the goodwill/abandonment cost of a
#   customer whose card was declined at checkout. We set this at 50 currency
#   units. It is a constant, not a fitted parameter -- change it here and the
#   cost-optimal threshold moves, which is exactly the sensitivity a merchant
#   should be shown.
#
# Dataset amounts are EUR (European cardholders, Sept 2013). We present the
# numbers as generic "currency units" and keep FP_COST on the same scale.
FP_COST = 50.0          # cost of wrongly blocking one good transaction
FN_COST_MODE = "amount"  # "amount" (per-transaction) or "average"
CHARGEBACK_FEE = 0.0     # optional flat fee added on top of each missed fraud

# ------------------------------------------------------------- decisioning
# Three-way banding around the cost-optimal threshold. Everything between
# REVIEW_BAND_LOW and the block threshold goes to a human queue instead of an
# automatic decision.
REVIEW_BAND_WIDTH = 0.5   # fallback only: block_threshold * this factor
#
# Preferred: size the review band by REVIEW CAPACITY, not by a magic factor.
# A merchant's review team can look at a fixed share of traffic per day; that
# operational limit is what should decide how wide the band is. We target
# 0.2% of transactions (~85 cases per 42.5k), and the score that produces it
# is read off out-of-fold predictions in models/feature_context.json.
#
# This matters: with the band derived from block_threshold * 0.5, only 7 of
# 42,559 held-out transactions (0.016%) ever reached a human -- the review
# queue was decorative. Capacity-based sizing gives it real volume.
REVIEW_CAPACITY_RATE = 0.002

# Bounded/gated decisioning: the service will never auto-block more than this
# share of traffic in the rolling window. Beyond it, blocks are force-downgraded
# to "review" and the downgrade is logged with a reason.
MAX_BLOCK_RATE = 0.05     # 5% of the rolling window
ROLLING_WINDOW = 200      # decisions considered when measuring block rate
# Floor for the rate denominator on a cold window. NOT an exemption: the gate
# is active from the very first request. With a floor of 20 and a 5% cap, the
# first block is admitted (1/20 == 5%) but a second consecutive one is held --
# so genuine fraud is still blocked at startup while a burst cannot slip
# through before the window fills.
GATE_MIN_SAMPLE = 20

MODEL_VERSION = "frm-0.1.0"

# ----------------------------------------------------------------- artefacts
BASELINE_MODEL = MODELS / "baseline.json"
TUNED_MODEL = MODELS / "tuned.json"
BEST_PARAMS = MODELS / "best_params.json"
FEATURE_NAMES = MODELS / "feature_names.json"
SERVING_CONFIG = MODELS / "serving_config.json"
AUDIT_LOG = REPORTS / "audit_log.jsonl"
REVIEW_QUEUE = REPORTS / "review_queue.jsonl"
FEATURE_CONTEXT = MODELS / "feature_context.json"
OPTUNA_DB = REPORTS / "optuna_study.db"
