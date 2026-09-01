"""EDA source-of-truth. Executed into notebooks/01_eda.ipynb by build_notebook.py.

Kept as a plain script so it is diffable and testable; the notebook is the
rendered artefact. Covers: class imbalance, Amount/Time by class, V-feature
correlation with the label, and why accuracy is a meaningless metric here.
"""
# %% [markdown]
# # EDA — merchant transaction fraud (ULB `creditcardfraud`)
#
# 284,807 real card transactions from a two-day window in September 2013
# (European cardholders), released by the ULB Machine Learning Group with
# Worldline. `V1`–`V28` are PCA components published by the dataset authors —
# the raw features were confidential. `Time` and `Amount` are untransformed.
#
# This is a published benchmark, which is the point: our numbers can be checked
# against other people's numbers.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_prep import download

plt.rcParams["figure.figsize"] = (10, 4)
df = download()
print(df.shape)
df.head()

# %% [markdown]
# ## 1. Class imbalance — the fact that governs every later decision

# %%
counts = df.Class.value_counts()
rate = df.Class.mean()
print(f"legitimate : {counts[0]:>8,}")
print(f"fraud      : {counts[1]:>8,}")
print(f"fraud rate : {rate:.4%}  (1 in {1 / rate:,.0f} transactions)")

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.5))
a.bar(["legit", "fraud"], counts.values, color=["#2563eb", "#dc2626"])
a.set_yscale("log")
a.set_title("Class counts (log scale)")
b.bar(["legit", "fraud"], counts.values / len(df) * 100, color=["#2563eb", "#dc2626"])
b.set_ylabel("% of transactions")
b.set_title("Class share (linear) — fraud is invisible at this scale")
plt.tight_layout()

# %% [markdown]
# ## 2. Why accuracy is a meaningless metric here
#
# A model that predicts "legitimate" for every single transaction — one line of
# code, zero intelligence, catches zero fraud — scores:

# %%
trivial_acc = 1 - rate
print(f"accuracy of 'always predict legitimate' : {trivial_acc:.4%}")
print(f"fraud caught by that model              : 0 of {counts[1]}")
print(f"money it saves the merchant             : 0")

# %% [markdown]
# 99.83% accuracy, useless model. So we report **PR-AUC**, not accuracy and not
# ROC-AUC.
#
# ROC-AUC is also misleading here. Its x-axis is the false positive rate,
# `FP / (FP + TN)`, and `TN` is ~284,000. Thousands of false positives barely
# move that denominator, so ROC-AUC looks flattering even when the model is
# unusable in production. Precision, `TP / (TP + FP)`, has no such cushion — it
# is exactly the quantity a merchant feels, because every false positive is a
# real customer whose card was declined at checkout.
#
# The random-guessing baseline for PR-AUC is the positive rate itself
# (0.0017), so a PR-AUC of ~0.82 is roughly **475×** better than chance.

# %%
from sklearn.metrics import average_precision_score, roc_auc_score

rng = np.random.default_rng(0)
# A deliberately weak "model": pure noise, plus a faint real signal.
noise = rng.random(len(df))
weak = 0.85 * noise + 0.15 * df.Class
print(f"random scores       -> ROC-AUC {roc_auc_score(df.Class, noise):.3f}  "
      f"PR-AUC {average_precision_score(df.Class, noise):.5f}")
print(f"barely-useful model -> ROC-AUC {roc_auc_score(df.Class, weak):.3f}  "
      f"PR-AUC {average_precision_score(df.Class, weak):.5f}")
print("\nNote how ROC-AUC flatters the weak model while PR-AUC stays honest.")

# %% [markdown]
# ## 3. Amount — the reason a cost model is not optional

# %%
legit, fraud = df[df.Class == 0].Amount, df[df.Class == 1].Amount
summary = pd.DataFrame({"legitimate": legit.describe(), "fraud": fraud.describe()})
print(summary.round(2))
print(f"\ntotal value of all fraud in the dataset: {fraud.sum():,.0f}")
print(f"median fraud {fraud.median():.2f} vs median legit {legit.median():.2f}")
print(f"largest fraud {fraud.max():,.2f}")

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.5))
a.hist(np.log1p(legit), bins=60, alpha=0.6, density=True, label="legit", color="#2563eb")
a.hist(np.log1p(fraud), bins=60, alpha=0.6, density=True, label="fraud", color="#dc2626")
a.set_xlabel("log1p(Amount)")
a.set_title("Amount distribution by class")
a.legend()
b.boxplot([legit, fraud], tick_labels=["legit", "fraud"], showfliers=False)
b.set_ylabel("Amount")
b.set_title("Amount by class (outliers hidden)")
plt.tight_layout()

# %% [markdown]
# Fraud amounts are heavily skewed: most are small, a few are large. This is
# why the cost model charges a false negative the **actual** amount of the
# missed transaction rather than a flat average — averaging would let a model
# that misses the expensive frauds look identical to one that misses the cheap
# ones.

# %% [markdown]
# ## 4. Time — is there a time-of-day signal worth encoding?

# %%
df["hour"] = (df.Time / 3600) % 24
by_hour = df.groupby(df.hour.astype(int)).agg(
    n=("Class", "size"), frauds=("Class", "sum"))
by_hour["fraud_rate"] = by_hour.frauds / by_hour.n

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.5))
a.bar(by_hour.index, by_hour.n, color="#94a3b8")
a.set_title("Transaction volume by hour of day")
a.set_xlabel("hour")
b.bar(by_hour.index, by_hour.fraud_rate * 100, color="#dc2626")
b.axhline(rate * 100, ls="--", color="#0f172a", label="overall rate")
b.set_title("Fraud rate by hour (%)")
b.set_xlabel("hour")
b.legend()
plt.tight_layout()
print(by_hour.round(5))

# %% [markdown]
# Volume collapses overnight but the fraud *rate* rises sharply in those hours.
# Absolute elapsed `Time` would not generalise past this two-day window, so the
# feature we actually build is cyclical hour-of-day (`hour_sin`, `hour_cos`),
# which keeps 23:00 and 01:00 adjacent.

# %% [markdown]
# ## 5. Which V-features carry the label signal?

# %%
v_cols = [f"V{i}" for i in range(1, 29)]
corr = df[v_cols + ["Amount"]].corrwith(df.Class).sort_values(key=abs, ascending=False)
print(corr.head(12).round(4))

fig, ax = plt.subplots(figsize=(11, 3.5))
top = corr.head(15)[::-1]
ax.barh(top.index, top.values,
        color=["#dc2626" if v > 0 else "#2563eb" for v in top.values])
ax.set_xlabel("Pearson correlation with the fraud label")
ax.set_title("Strongest linear label correlations")
plt.tight_layout()

# %% [markdown]
# V17, V14, V12 and V10 lead. These are *linear* correlations on PCA
# components — useful as a sanity check that signal exists, not as a feature
# selection method, since a boosted tree exploits interactions no correlation
# will show. We keep all 28 and let TreeSHAP tell us afterwards what the model
# actually used.

# %%
fig, axes = plt.subplots(1, 4, figsize=(13, 3))
for ax, c in zip(axes, ["V17", "V14", "V12", "V10"]):
    ax.hist(df[df.Class == 0][c], bins=60, density=True, alpha=0.6,
            label="legit", color="#2563eb")
    ax.hist(df[df.Class == 1][c], bins=60, density=True, alpha=0.6,
            label="fraud", color="#dc2626")
    ax.set_title(c)
axes[0].legend()
plt.tight_layout()

# %% [markdown]
# The class-conditional distributions genuinely separate — this is a learnable
# problem, not noise.

# %% [markdown]
# ## 6. Duplicate rows — a real leakage trap

# %%
dups = df.duplicated().sum()
print(f"exact duplicate rows: {dups}")
print("These are dropped in src/data_prep.py BEFORE splitting. Left in, the")
print("same transaction can land in both train and test, which inflates the")
print("held-out score for free. Our split drops them and then asserts")
print("disjointness by row hash.")

# %% [markdown]
# ## Takeaways carried into the model
#
# 1. 0.17% positives → PR-AUC is the headline metric; accuracy is banned.
# 2. Fraud amounts are skewed → cost model charges the real amount per miss.
# 3. Time-of-day matters, absolute time does not → cyclical encoding.
# 4. Signal is real and non-linear → gradient-boosted trees, all 28 components.
# 5. Duplicates exist → drop before splitting, then assert disjointness loudly.
