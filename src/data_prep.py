"""Download the ULB creditcardfraud benchmark from the Hugging Face Hub and
produce a leakage-checked stratified 70/15/15 split.

The leakage check is loud and explicit on purpose: a silently-overlapping split
is the single most common source of fake-looking results on this dataset.
"""
from __future__ import annotations

import hashlib
import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PROCESSED,
    DATA_RAW,
    DATA_SAMPLE,
    EXPECTED_FRAUDS,
    EXPECTED_ROWS,
    HF_DATASET_FILE,
    HF_DATASET_MIRROR,
    HF_DATASET_REPO,
    RANDOM_SEED,
    REPORTS,
)

RAW_CSV = DATA_RAW / "creditcard.csv"


def download() -> pd.DataFrame:
    """Fetch the dataset from the HF Hub (cached after the first call)."""
    from huggingface_hub import hf_hub_download

    if RAW_CSV.exists():
        print(f"[data] using cached raw file {RAW_CSV}")
        return pd.read_csv(RAW_CSV)

    last_err = None
    for repo in (HF_DATASET_REPO, HF_DATASET_MIRROR):
        try:
            print(f"[data] downloading {repo}/{HF_DATASET_FILE} from Hugging Face ...")
            path = hf_hub_download(repo, HF_DATASET_FILE, repo_type="dataset")
            df = pd.read_csv(path)
            RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(RAW_CSV, index=False)
            print(f"[data] saved raw copy to {RAW_CSV}")
            return df
        except Exception as exc:  # noqa: BLE001 - fall through to the mirror
            print(f"[data] {repo} failed: {type(exc).__name__}: {exc}")
            last_err = exc
    raise RuntimeError("could not download the dataset from any mirror") from last_err


def verify_source(df: pd.DataFrame) -> None:
    """Confirm we actually got the published benchmark, not a lookalike."""
    print(f"[verify] shape={df.shape}")
    frauds = int(df["Class"].sum())
    print(f"[verify] frauds={frauds}  rate={frauds / len(df):.6f}")
    if df.shape[0] != EXPECTED_ROWS or frauds != EXPECTED_FRAUDS:
        print(
            f"[verify] WARNING: expected {EXPECTED_ROWS} rows / {EXPECTED_FRAUDS} "
            f"frauds (published ULB figures) but got {df.shape[0]} / {frauds}. "
            "Benchmark comparisons against published baselines are NOT valid."
        )
    else:
        print("[verify] OK - matches the published ULB figures exactly.")


def row_hashes(df: pd.DataFrame) -> pd.Series:
    """Content hash of every row, used for the leakage assertion."""
    payload = df.round(10).astype(str).agg("|".join, axis=1)
    return payload.map(lambda s: hashlib.sha1(s.encode()).hexdigest())


def prepare(seed: int = RANDOM_SEED) -> dict:
    df = download()
    verify_source(df)

    # Exact duplicate rows genuinely exist in this dataset. Left in place they
    # can land on both sides of the split, which is real leakage -- so drop
    # them before splitting rather than papering over it later.
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[data] dropped {before - len(df)} exact duplicate rows -> {len(df)} unique")

    df["row_id"] = df.index
    df["row_hash"] = row_hashes(df.drop(columns=["row_id"]))

    y = df["Class"]
    train, holdout = train_test_split(
        df, test_size=0.30, stratify=y, random_state=seed, shuffle=True
    )
    val, test = train_test_split(
        holdout, test_size=0.50, stratify=holdout["Class"], random_state=seed, shuffle=True
    )
    splits = {"train": train, "val": val, "test": test}

    print("\n[balance] class balance per split")
    print(f"{'split':<8}{'rows':>9}{'frauds':>9}{'rate':>12}")
    summary = {}
    for name, part in splits.items():
        n, f = len(part), int(part["Class"].sum())
        print(f"{name:<8}{n:>9,}{f:>9,}{f / n:>12.6f}")
        summary[name] = {"rows": n, "frauds": f, "fraud_rate": f / n}

    check_leakage(splits)

    for name, part in splits.items():
        out = DATA_PROCESSED / f"{name}.parquet"
        part.reset_index(drop=True).to_parquet(out, index=False)
        print(f"[data] wrote {out}  ({len(part):,} rows)")

    # Small committed sample so the repo demos without a download.
    sample = pd.concat(
        [test[test.Class == 1].head(40), test[test.Class == 0].head(160)]
    ).sample(frac=1, random_state=seed)
    sample.to_csv(DATA_SAMPLE / "sample_transactions.csv", index=False)
    print(f"[data] wrote demo sample ({len(sample)} rows) to data/sample/")

    (REPORTS / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def check_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Assert no row appears in more than one split, by id AND by content hash."""
    print("\n" + "=" * 62)
    print("LEAKAGE CHECK - no row may appear in more than one split")
    print("=" * 62)
    names = list(splits)
    clean = True
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            id_overlap = set(splits[a]["row_id"]) & set(splits[b]["row_id"])
            hash_overlap = set(splits[a]["row_hash"]) & set(splits[b]["row_hash"])
            status = "PASS" if not id_overlap and not hash_overlap else "FAIL"
            clean &= status == "PASS"
            print(
                f"  {a:<6} vs {b:<6}  shared row_ids={len(id_overlap):<6} "
                f"shared row_hashes={len(hash_overlap):<6} [{status}]"
            )
    total = sum(len(p) for p in splits.values())
    uniq_ids = len(set().union(*[set(p["row_id"]) for p in splits.values()]))
    print(f"  total rows across splits = {total:,}; distinct row_ids = {uniq_ids:,}")
    if total != uniq_ids:
        clean = False
    print("=" * 62)
    print("LEAKAGE CHECK: " + ("PASSED - splits are disjoint" if clean else "FAILED"))
    print("=" * 62 + "\n")
    assert clean, "LEAKAGE DETECTED - refusing to continue"


if __name__ == "__main__":
    prepare()
    sys.exit(0)
