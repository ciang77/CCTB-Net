"""Network flow CSV ingestion, identifier stripping, and global stable ordering."""
from __future__ import annotations

import glob
import hashlib
import io
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data_pipeline.schema import (
    CLASS_TO_ID,
    NET_CAT_RAW_COLS,
    NET_NUM_RAW_COLS,
    NET_REQUIRED_CSV_COLS,
)


def _natural_key(path_str: str) -> Tuple[int, str]:
    stem = Path(path_str).stem
    parts = stem.split("_")
    num = int(parts[-1]) if parts[-1].isdigit() else 0
    return num, path_str


def _read_network_csv_sampled(path: Path, sample_frac: float, seed: int) -> pd.DataFrame:
    """Read a network CSV file; when sample_frac < 1.0, read evenly-spaced contiguous slices."""
    file_size = path.stat().st_size
    if sample_frac >= 0.999 or file_size < 2 * 1024 * 1024:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
        if sample_frac < 0.999 and len(df) > 0:
            block = 32
            stride = max(1, int(round(1.0 / sample_frac)))
            offset = seed % stride
            mask = ((np.arange(len(df)) // block) % stride) == offset
            if not mask.any():
                mask[0] = True
            df = df.loc[mask].reset_index(drop=True)
        return df

    # Fast contiguous multi-window byte reader for large files when sample_frac < 1.0
    num_windows = 20
    target_bytes = max(64 * 1024, int(file_size * sample_frac))
    win_bytes = max(8 * 1024, target_bytes // num_windows)

    with path.open("rb") as f:
        header_line = f.readline()
        data_start = f.tell()
        usable = max(1, file_size - data_start - win_bytes)
        chunks: List[bytes] = [header_line]
        for w in range(num_windows):
            pos = data_start + int((w / max(1, num_windows - 1)) * usable)
            f.seek(pos)
            if pos > data_start:
                f.readline()  # align to next full line
            chunk = f.read(win_bytes)
            last_nl = chunk.rfind(b"\n")
            if last_nl > 0:
                chunk = chunk[: last_nl + 1]
            chunks.append(chunk)

    text = b"".join(chunks).decode("utf-8-sig", errors="replace")
    return pd.read_csv(io.StringIO(text), dtype=str, on_bad_lines="skip", low_memory=False)


def _compute_entity_keys(df: pd.DataFrame, mode: str) -> np.ndarray:
    src = df["src_ip"].fillna("-").astype(str).str.strip().to_numpy()
    dst = df["dst_ip"].fillna("-").astype(str).str.strip().to_numpy()
    if mode == "global":
        return np.zeros(len(df), dtype=np.int64)
    if mode == "dst_ip":
        keys = pd.Series(dst)
    else:
        lo = np.minimum(src, dst)
        hi = np.maximum(src, dst)
        keys = pd.Series(lo) + "|" + pd.Series(hi)
    return pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64).view(np.int64)


class RawCategoryPool:
    """Encodes raw categorical strings into compact int32 codes (0 reserved for '-' / missing)."""

    def __init__(self) -> None:
        self.col_to_map: Dict[str, Dict[str, int]] = {c: {"-": 0, "": 0} for c in NET_CAT_RAW_COLS}

    def encode_column(self, col: str, s: pd.Series) -> np.ndarray:
        mapping = self.col_to_map[col]
        vals = s.fillna("-").astype(str).str.strip()
        vals = vals.where(vals != "", "-")
        uniq = vals.unique()
        for u in uniq:
            if u not in mapping:
                mapping[u] = len(mapping)
        return vals.map(mapping).to_numpy(dtype=np.int32)


def ingest_network_files(
    network_dir: Path,
    sample_frac: float = 1.0,
    sample_seed: int = 42,
    entity_key_mode: str = "host_pair",
) -> Tuple[Dict[str, np.ndarray], RawCategoryPool]:
    """Read network CSVs, strip identifier columns, and return globally sorted arrays."""
    files = sorted(glob.glob(str(network_dir / "Network_dataset_*.csv")), key=_natural_key)
    if not files:
        files = sorted(glob.glob(str(network_dir / "*.csv")), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No network CSV files found in {network_dir}")

    cat_pool = RawCategoryPool()
    ts_list, dur_list, ts_end_list = [], [], []
    file_idx_list, row_idx_list, entity_list = [], [], []
    dst_port_list, y_cls_list, y_bin_list = [], [], []
    num_lists: List[np.ndarray] = []
    cat_lists: List[np.ndarray] = []

    for f_idx, fp in enumerate(files):
        df = _read_network_csv_sampled(Path(fp), sample_frac=sample_frac, seed=sample_seed + f_idx)
        if df.empty:
            continue
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        missing_cols = [c for c in NET_REQUIRED_CSV_COLS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns {missing_cols} in {fp}")

        ts = pd.to_numeric(df["ts"], errors="coerce").to_numpy(dtype=np.float64)
        dur = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        dur = np.maximum(dur, 0.0)
        type_str = df["type"].fillna("").astype(str).str.strip().str.lower()
        y_cls = type_str.map(CLASS_TO_ID).to_numpy(dtype=np.float64)

        valid = np.isfinite(ts) & (ts > 0) & np.isfinite(y_cls)
        if not valid.all():
            df = df.loc[valid].reset_index(drop=True)
            ts = ts[valid]
            dur = dur[valid]
            y_cls = y_cls[valid]
        if len(df) == 0:
            continue

        n = len(df)
        ts_end = ts + dur
        entity_key = _compute_entity_keys(df, mode=entity_key_mode)
        dst_port = pd.to_numeric(df["dst_port"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32)
        y_cls_i = y_cls.astype(np.int64)
        y_bin_i = (y_cls_i != 0).astype(np.float32)

        num_mat = np.zeros((n, len(NET_NUM_RAW_COLS)), dtype=np.float32)
        num_mat[:, 0] = dur.astype(np.float32)
        for j, col in enumerate(NET_NUM_RAW_COLS[1:], start=1):
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            num_mat[:, j] = np.maximum(vals, 0.0).astype(np.float32)

        # Fast path when all raw codes fit in int16 (standard for TON_IoT categoricals)
        cat_mat = np.zeros((n, len(NET_CAT_RAW_COLS)), dtype=np.int32)
        for j, col in enumerate(NET_CAT_RAW_COLS):
            cat_mat[:, j] = cat_pool.encode_column(col, df[col])

        ts_list.append(ts)
        dur_list.append(dur)
        ts_end_list.append(ts_end)
        file_idx_list.append(np.full(n, f_idx, dtype=np.int16))
        row_idx_list.append(np.arange(n, dtype=np.int32))
        entity_list.append(entity_key)
        dst_port_list.append(dst_port)
        y_cls_list.append(y_cls_i)
        y_bin_list.append(y_bin_i)
        num_lists.append(num_mat)
        max_c = int(cat_mat.max()) if n > 0 else 0
        cat_lists.append(cat_mat.astype(np.int16) if max_c < 32767 else cat_mat)

    ts_all = np.concatenate(ts_list)
    del ts_list, dur_list
    ts_end_all = np.concatenate(ts_end_list)
    del ts_end_list
    f_all = np.concatenate(file_idx_list)
    del file_idx_list
    r_all = np.concatenate(row_idx_list)
    del row_idx_list

    # Global deterministic sort by (ts_end, file_idx, row_idx)
    order = np.lexsort((r_all, f_all, ts_end_all))
    del f_all, r_all

    def _take_and_free(lst: List[np.ndarray]) -> np.ndarray:
        cat_arr = np.concatenate(lst, axis=0)
        lst.clear()
        out = cat_arr[order]
        del cat_arr
        return out

    arrays = {
        "ts": ts_all[order],
        "ts_end": ts_end_all[order],
        "entity_key": _take_and_free(entity_list),
        "dst_port": _take_and_free(dst_port_list),
        "y_cls": _take_and_free(y_cls_list),
        "y_bin": _take_and_free(y_bin_list),
        "num_raw": _take_and_free(num_lists),
        "cat_raw": _take_and_free(cat_lists),
    }
    del ts_all, ts_end_all, order
    return arrays, cat_pool
