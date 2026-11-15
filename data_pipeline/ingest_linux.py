"""Robust Linux host log parser and bucket aggregator (drops label/type columns)."""
from __future__ import annotations

import glob
import io
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_pipeline.schema import LINUX_SOURCES

_NUM_TOKEN_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([kKmMgGtT%]?)\s*$")
_UNIT_MULT = {
    "": 1.0,
    "%": 1.0,
    "k": 1e3,
    "K": 1e3,
    "m": 1e6,
    "M": 1e6,
    "g": 1e9,
    "G": 1e9,
    "t": 1e12,
    "T": 1e12,
}


def parse_dirty_numeric_series(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Parse dirty atop strings ('1184K', '97%', '0456 6', 'apache2').

    Returns:
        values: float64 Series (NaN where unparseable)
        is_bad: bool Series (True where non-empty, non-'-' value failed to parse)
    """
    if pd.api.types.is_numeric_dtype(s):
        vals = s.astype(np.float64)
        return vals.fillna(0.0), vals.isna()

    raw = s.fillna("").astype(str).str.strip()
    is_missing = (raw == "") | (raw == "-") | (raw.str.lower() == "nan")
    # Handle space-separated numbers like "0456 6" -> take first token
    first_tok = raw.str.split().str[0].fillna("")

    extracted = first_tok.str.extract(_NUM_TOKEN_RE)
    num_part = pd.to_numeric(extracted[0], errors="coerce")
    unit_part = extracted[1].fillna("").map(_UNIT_MULT).fillna(1.0)
    vals = num_part * unit_part
    is_bad = (~is_missing) & vals.isna()
    return vals.fillna(0.0).astype(np.float64), is_bad


def _read_linux_csv_strided(path: Path, stride: int = 1) -> pd.DataFrame:
    stride = max(1, int(stride))
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()
        if not header:
            return pd.DataFrame()
        if stride == 1:
            body = f.read()
        else:
            lines = [line for idx, line in enumerate(f) if idx % stride == 0]
            body = "".join(lines)
    buf = io.StringIO(header + body)
    return pd.read_csv(buf, dtype=str, on_bad_lines="skip", low_memory=False)


def load_and_clean_linux_kind(
    linux_dir: Path,
    kind: str,
    stride: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    spec = LINUX_SOURCES[kind]
    matched: List[str] = []
    for pat in spec["patterns"]:
        matched.extend(glob.glob(str(linux_dir / pat)))
    files = sorted({os.path.normpath(p) for p in matched})
    if not files:
        return pd.DataFrame(), {"raw_rows": 0, "kept_rows": 0, "dropped_bad_rows": 0}

    frames: List[pd.DataFrame] = []
    for fp in files:
        df = _read_linux_csv_strided(Path(fp), stride=stride)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(), {"raw_rows": 0, "kept_rows": 0, "dropped_bad_rows": 0}

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df.columns = [c.strip().lstrip("\ufeff") for c in raw_df.columns]
    raw_rows = len(raw_df)

    # Explicitly drop label and type columns to prevent label leakage
    for drop_col in spec["drop_cols"]:
        if drop_col in raw_df.columns:
            raw_df = raw_df.drop(columns=[drop_col])

    raw_df = raw_df.drop_duplicates().reset_index(drop=True)

    ts_vals, ts_bad = parse_dirty_numeric_series(raw_df["ts"])
    bad_mask = ts_bad | (ts_vals <= 0)
    clean_cols: Dict[str, pd.Series] = {"ts": ts_vals.astype(np.int64)}

    for col in spec["num_cols"]:
        if col in raw_df.columns:
            vals, is_bad = parse_dirty_numeric_series(raw_df[col])
            clean_cols[col] = vals
            bad_mask = bad_mask | is_bad
        else:
            clean_cols[col] = pd.Series(0.0, index=raw_df.index)

    for col in spec["cat_cols"]:
        if col in raw_df.columns:
            clean_cols[col] = raw_df[col].fillna("-").astype(str).str.strip()
        else:
            clean_cols[col] = pd.Series("-", index=raw_df.index)

    clean_df = pd.DataFrame(clean_cols)
    dropped = int(bad_mask.sum())
    clean_df = clean_df.loc[~bad_mask].reset_index(drop=True)
    stats = {
        "raw_rows": int(raw_rows),
        "kept_rows": int(len(clean_df)),
        "dropped_bad_rows": dropped,
    }
    return clean_df, stats


def aggregate_linux_buckets(
    linux_dir: Path,
    bucket_sec: int = 5,
    stride: int = 1,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, int]]]:
    """Aggregate process, memory, and disk logs into delta-second time buckets."""
    bucket_tables: Dict[str, pd.DataFrame] = {}
    all_stats: Dict[str, Dict[str, int]] = {}

    # 1. Process buckets
    proc_df, proc_stats = load_and_clean_linux_kind(linux_dir, "process", stride=stride)
    all_stats["process"] = proc_stats
    if not proc_df.empty:
        proc_df["bucket"] = proc_df["ts"] // bucket_sec
        proc_df["state_r"] = (proc_df["State"] == "R").astype(np.float64)
        grp = proc_df.groupby("bucket", sort=True)
        bucket_tables["process"] = pd.DataFrame(
            {
                "proc_rows": grp["ts"].size().astype(np.float64),
                "proc_cpu_sum": grp["CPU"].sum(),
                "proc_cpu_max": grp["CPU"].max(),
                "proc_trun_sum": grp["TRUN"].sum(),
                "proc_tslpi_sum": grp["TSLPI"].sum(),
                "proc_state_r_ratio": grp["state_r"].mean(),
            }
        )
    else:
        bucket_tables["process"] = pd.DataFrame()

    # 2. Memory buckets
    mem_df, mem_stats = load_and_clean_linux_kind(linux_dir, "memory", stride=stride)
    all_stats["memory"] = mem_stats
    if not mem_df.empty:
        mem_df["bucket"] = mem_df["ts"] // bucket_sec
        grp = mem_df.groupby("bucket", sort=True)
        bucket_tables["memory"] = pd.DataFrame(
            {
                "mem_minflt_sum": grp["MINFLT"].sum(),
                "mem_majflt_sum": grp["MAJFLT"].sum(),
                "mem_rsize_sum": grp["RSIZE"].sum(),
                "mem_usage_max": grp["MEM"].max(),
            }
        )
    else:
        bucket_tables["memory"] = pd.DataFrame()

    # 3. Disk buckets
    disk_df, disk_stats = load_and_clean_linux_kind(linux_dir, "disk", stride=stride)
    all_stats["disk"] = disk_stats
    if not disk_df.empty:
        disk_df["bucket"] = disk_df["ts"] // bucket_sec
        grp = disk_df.groupby("bucket", sort=True)
        bucket_tables["disk"] = pd.DataFrame(
            {
                "disk_rddsk_sum": grp["RDDSK"].sum(),
                "disk_wrdsk_sum": grp["WRDSK"].sum(),
            }
        )
    else:
        bucket_tables["disk"] = pd.DataFrame()

    return bucket_tables, all_stats
