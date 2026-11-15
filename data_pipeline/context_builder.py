"""Multi-scale causal context tensor builder over delta-second time buckets."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from data_pipeline.schema import CTX_BASE_COLS, CTX_LOAD_KEY_COLS


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def build_context_grid(
    bucket_tables: Dict[str, pd.DataFrame],
    min_bucket: int,
    max_bucket: int,
    bucket_sec: int = 5,
    roll_sec: int = 60,
    ewm_halflife_sec: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build dense [num_buckets, 36] context matrix and [num_buckets, 3] observation mask.

    Features (36 total):
      - 12 base bucket metrics (signed log1p)
      - 12 causal rolling means over `roll_sec` (default 60s = 12 buckets)
      - 12 causal EWMA over `ewm_halflife_sec` (default 300s = 60 buckets)
    """
    full_index = pd.Index(np.arange(min_bucket, max_bucket + 1, dtype=np.int64), name="bucket")
    num_buckets = len(full_index)

    obs_mask = np.zeros((num_buckets, 3), dtype=np.bool_)
    base_df = pd.DataFrame(0.0, index=full_index, columns=CTX_BASE_COLS, dtype=np.float64)

    for src_idx, src_name in enumerate(["process", "memory", "disk"]):
        tbl = bucket_tables.get(src_name, pd.DataFrame())
        if tbl is not None and not tbl.empty:
            in_range = tbl.index.intersection(full_index)
            obs_mask[full_index.get_indexer(in_range), src_idx] = True
            for col in tbl.columns:
                if col in base_df.columns:
                    base_df.loc[in_range, col] = tbl.loc[in_range, col].to_numpy(dtype=np.float64)

    # Apply signed_log1p to base bucket values before causal smoothing
    log_base = pd.DataFrame(
        signed_log1p(base_df.to_numpy(dtype=np.float64)),
        index=full_index,
        columns=CTX_BASE_COLS,
    )

    roll_steps = max(1, int(round(roll_sec / bucket_sec)))
    halflife_steps = max(1.0, float(ewm_halflife_sec) / float(bucket_sec))

    roll_df = log_base.rolling(window=roll_steps, min_periods=1).mean()
    roll_df.columns = [f"{c}_roll" for c in CTX_BASE_COLS]

    ewm_df = log_base.ewm(halflife=halflife_steps, min_periods=1).mean()
    ewm_df.columns = [f"{c}_ewm" for c in CTX_BASE_COLS]

    all_df = pd.concat([log_base, roll_df, ewm_df], axis=1)
    feature_names = list(all_df.columns)  # 36 columns
    ctx_raw = all_df.to_numpy(dtype=np.float32)

    # Composite host load score (unscaled causal rolling load summary)
    load_cols_idx = [feature_names.index(c) for c in CTX_LOAD_KEY_COLS if c in feature_names]
    raw_load_score = ctx_raw[:, load_cols_idx].mean(axis=1).astype(np.float32)

    return ctx_raw, obs_mask, raw_load_score, feature_names
