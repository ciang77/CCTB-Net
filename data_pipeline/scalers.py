"""Train-only feature scalers, categorical vocabularies, and destination port bucketing."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class TrainOnlyRobustScaler:
    """Applies signed_log1p -> Train quantile clipping -> Train median/IQR scaling."""

    def __init__(self, clip_quantile: float = 0.999) -> None:
        self.clip_quantile = float(clip_quantile)
        self.low_clip: np.ndarray | None = None
        self.high_clip: np.ndarray | None = None
        self.median: np.ndarray | None = None
        self.iqr: np.ndarray | None = None
        self.fitted_count: int = 0

    def fit(self, x_train_raw: np.ndarray, max_fit_samples: int = 1_500_000) -> "TrainOnlyRobustScaler":
        if len(x_train_raw) == 0:
            raise ValueError("Cannot fit TrainOnlyRobustScaler on empty training array")
        self.fitted_count = int(len(x_train_raw))
        if len(x_train_raw) > max_fit_samples:
            sel = np.linspace(0, len(x_train_raw) - 1, num=max_fit_samples, dtype=np.int64)
            sample = x_train_raw[sel]
        else:
            sample = x_train_raw
        lx = signed_log1p(sample.astype(np.float64))
        q_lo = max(0.0, 1.0 - self.clip_quantile)
        q_hi = min(1.0, self.clip_quantile)
        self.low_clip = np.quantile(lx, q_lo, axis=0)
        self.high_clip = np.quantile(lx, q_hi, axis=0)
        clipped = np.clip(lx, self.low_clip, self.high_clip)
        self.median = np.median(clipped, axis=0)
        q25 = np.quantile(clipped, 0.25, axis=0)
        q75 = np.quantile(clipped, 0.75, axis=0)
        iqr = q75 - q25
        self.iqr = np.where(iqr > 1e-6, iqr, 1.0)
        return self

    def transform_into(self, x_raw: np.ndarray, out_arr: np.ndarray, chunk_size: int = 2_000_000) -> None:
        assert self.median is not None and self.iqr is not None
        lo = self.low_clip.astype(np.float32)
        hi = self.high_clip.astype(np.float32)
        med = self.median.astype(np.float32)
        iqr = self.iqr.astype(np.float32)
        for s in range(0, len(x_raw), chunk_size):
            e = min(len(x_raw), s + chunk_size)
            lx = signed_log1p(x_raw[s:e].astype(np.float32))
            clipped = np.clip(lx, lo, hi)
            out_arr[s:e] = ((clipped - med) / iqr).astype(np.float32)

    def transform(self, x_raw: np.ndarray) -> np.ndarray:
        assert self.median is not None and self.iqr is not None
        lx = signed_log1p(x_raw.astype(np.float32))
        clipped = np.clip(lx, self.low_clip.astype(np.float32), self.high_clip.astype(np.float32))
        scaled = (clipped - self.median.astype(np.float32)) / self.iqr.astype(np.float32)
        return scaled.astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_quantile": self.clip_quantile,
            "fitted_count": self.fitted_count,
            "low_clip": self.low_clip.tolist(),
            "high_clip": self.high_clip.tolist(),
            "median": self.median.tolist(),
            "iqr": self.iqr.tolist(),
        }


class TrainOnlyStandardScaler:
    """StandardScaler fitted strictly on context buckets referenced by Train flows."""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.fitted_count: int = 0

    def fit(self, x_train: np.ndarray) -> "TrainOnlyStandardScaler":
        if len(x_train) == 0:
            raise ValueError("Cannot fit TrainOnlyStandardScaler on empty array")
        self.fitted_count = int(len(x_train))
        x64 = x_train.astype(np.float64)
        self.mean = x64.mean(axis=0)
        std = x64.std(axis=0)
        self.std = np.where(std > 1e-6, std, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        return ((x.astype(np.float32) - self.mean.astype(np.float32)) / self.std.astype(np.float32)).astype(
            np.float32
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fitted_count": self.fitted_count,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


class TrainOnlyCategoricalEncoder:
    """Maps raw categorical integer codes using ONLY training split frequencies.

    Index 0 = missing ('-')
    Index 1 = OOV (unseen in Train or count < min_freq in Train)
    Index 2..V-1 = in-vocabulary categories from Train
    """

    def __init__(self, min_freq: int = 20) -> None:
        self.min_freq = max(1, int(min_freq))
        self.luts: List[np.ndarray] = []
        self.vocab_sizes: List[int] = []

    def fit_transform_into(
        self, cat_raw: np.ndarray, train_mask: np.ndarray, out_arr: np.ndarray
    ) -> None:
        n, num_cols = cat_raw.shape
        self.luts = []
        self.vocab_sizes = []

        for j in range(num_cols):
            col_all = cat_raw[:, j]
            max_code = int(col_all.max()) if n > 0 else 0
            lut = np.ones(max_code + 1, dtype=np.int16)  # default to 1 (OOV)
            lut[0] = 0  # 0 is always missing ('-')

            train_codes = col_all[train_mask]
            counts = np.bincount(train_codes, minlength=max_code + 1)
            next_id = 2
            for code in range(1, max_code + 1):
                if counts[code] >= self.min_freq:
                    lut[code] = next_id
                    next_id += 1

            out_arr[:, j] = lut[col_all]
            self.luts.append(lut)
            self.vocab_sizes.append(next_id)

    def fit_transform(self, cat_raw: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
        out = np.zeros(cat_raw.shape, dtype=np.int16)
        self.fit_transform_into(cat_raw, train_mask, out)
        return out


class TrainOnlyPortBucketer:
    """Buckets destination ports using ONLY Train top-K frequent ports + range buckets.

    0 = missing / invalid (<0)
    1 = well-known (0..1023, not in top-K)
    2 = registered (1024..49151, not in top-K)
    3 = ephemeral (>=49152, not in top-K)
    4..(4+K-1) = Top-K frequent ports in Train
    """

    def __init__(self, top_k: int = 32) -> None:
        self.top_k = int(top_k)
        self.port_to_idx: Dict[int, int] = {}
        self.vocab_size: int = 4

    def fit_transform(self, dst_port: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
        train_ports = dst_port[train_mask]
        valid_train = train_ports[train_ports >= 0]
        if len(valid_train) > 0:
            uniq, counts = np.unique(valid_train, return_counts=True)
            order = np.argsort(-counts)[: self.top_k]
            top_ports = uniq[order]
        else:
            top_ports = np.array([], dtype=np.int32)

        self.port_to_idx = {int(p): 4 + idx for idx, p in enumerate(top_ports)}
        self.vocab_size = 4 + len(self.port_to_idx)

        out = np.zeros(len(dst_port), dtype=np.int16)
        valid = dst_port >= 0
        p_valid = dst_port[valid]
        res = np.ones(len(p_valid), dtype=np.int16)
        res[(p_valid >= 1024) & (p_valid < 49152)] = 2
        res[p_valid >= 49152] = 3
        for p_val, idx in self.port_to_idx.items():
            res[p_valid == p_val] = idx
        out[valid] = res
        return out
