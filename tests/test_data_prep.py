"""The split must be disjoint. This is the test that keeps the headline
numbers honest, so it checks by exact row hash, not just by count."""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import DATA_PROCESSED
from src.data_prep import check_leakage, row_hashes

SPLITS = ("train", "val", "test")


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    frames = {}
    for name in SPLITS:
        p = DATA_PROCESSED / f"{name}.parquet"
        if not p.exists():
            pytest.skip(f"{p} missing -- run `make data` first")
        frames[name] = pd.read_parquet(p)
    return frames


def test_no_row_id_overlap(splits):
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            assert not set(splits[a].row_id) & set(splits[b].row_id), f"{a}/{b} share rows"


def test_no_content_hash_overlap(splits):
    """The strict version: identical feature vectors must not span splits."""
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            shared = set(splits[a].row_hash) & set(splits[b].row_hash)
            assert not shared, f"{a}/{b} share {len(shared)} identical rows"


def test_stored_hashes_match_recomputed(splits):
    """Guards against a stale parquet whose row_hash no longer describes it."""
    df = splits["test"].head(500)
    feature_cols = [c for c in df.columns if c not in ("row_id", "row_hash")]
    recomputed = row_hashes(df[feature_cols].reset_index(drop=True))
    assert (recomputed.to_numpy() == df.row_hash.to_numpy()).all()


def test_check_leakage_raises_on_real_leakage(splits):
    """The guard must actually fail when leakage is introduced on purpose."""
    train = splits["train"]
    poisoned = {"train": train, "val": splits["val"],
                "test": pd.concat([splits["test"], train.head(5)])}
    with pytest.raises(AssertionError):
        check_leakage(poisoned)


def test_stratification_holds(splits):
    rates = {k: v.Class.mean() for k, v in splits.items()}
    assert all(r > 0 for r in rates.values()), "a split has no fraud at all"
    assert max(rates.values()) - min(rates.values()) < 5e-4, f"strata drifted: {rates}"


def test_split_proportions(splits):
    total = sum(len(v) for v in splits.values())
    assert abs(len(splits["train"]) / total - 0.70) < 0.01
    assert abs(len(splits["val"]) / total - 0.15) < 0.01
    assert abs(len(splits["test"]) / total - 0.15) < 0.01
