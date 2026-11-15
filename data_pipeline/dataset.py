"""End-to-end preprocessing orchestrator and vectorized memmap Dataset for CCTB-Net."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import torch

from data_pipeline.context_builder import build_context_grid
from data_pipeline.ingest_linux import aggregate_linux_buckets
from data_pipeline.ingest_network import ingest_network_files
from data_pipeline.scalers import (
    TrainOnlyCategoricalEncoder,
    TrainOnlyPortBucketer,
    TrainOnlyRobustScaler,
    TrainOnlyStandardScaler,
)
from data_pipeline.schema import CLASS_NAMES, NUM_CLASSES, SPLIT_NAMES, SPLIT_TO_ID
from data_pipeline.splitter import split_temporal_with_purge
from data_pipeline.windowing import build_entity_streams, compute_full_numeric_features
from engine.config import get_profile_dirs


def build_processed_dataset(cfg: Dict[str, Any], force: bool = False) -> Path:
    """Run the full Phase 1 preprocessing pipeline and save memory-mappable arrays."""
    dirs = get_profile_dirs(cfg)
    proc_dir = dirs["processed_dir"]
    meta_path = proc_dir / "metadata.json"
    if meta_path.exists() and not force:
        return proc_dir

    proc_dir.mkdir(parents=True, exist_ok=True)
    dcfg = cfg["data"]
    bucket_sec = int(dcfg["ctx_bucket_sec"])
    ctx_steps = int(dcfg["ctx_steps"])
    window_len = int(dcfg["window_len"])

    # 1. Ingest & globally sort network flows by (ts_end, file_idx, row_idx)
    net_arrays, _ = ingest_network_files(
        dirs["network_raw"],
        sample_frac=float(dcfg.get("sample_frac", 1.0)),
        sample_seed=int(dcfg.get("sample_seed", 42)),
        entity_key_mode=str(dcfg.get("entity_key", "host_pair")),
    )
    ts_end = net_arrays["ts_end"]
    y_cls = net_arrays["y_cls"]
    y_bin = net_arrays["y_bin"]
    n_flows = len(ts_end)

    # Flush non-feature arrays early to keep peak RAM low on 16GB hosts
    np.save(proc_dir / "ts.npy", net_arrays.pop("ts"))
    np.save(proc_dir / "ts_end.npy", ts_end)
    np.save(proc_dir / "y_cls.npy", y_cls.astype(np.int64))
    np.save(proc_dir / "y_bin.npy", y_bin.astype(np.float32))
    net_arrays.pop("y_cls", None)
    net_arrays.pop("y_bin", None)
    net_arrays.pop("ts_end", None)
    gc.collect()

    # 2. Build entity streams and 14 numeric features
    stream_order, stream_pos, stream_start, dt_prev = build_entity_streams(
        net_arrays.pop("entity_key"), ts_end
    )
    np.save(proc_dir / "stream_order.npy", stream_order)
    np.save(proc_dir / "stream_pos.npy", stream_pos)
    np.save(proc_dir / "stream_start.npy", stream_start)
    del stream_order, stream_pos, stream_start

    num_full_raw = compute_full_numeric_features(net_arrays.pop("num_raw"), dt_prev)
    del dt_prev
    gc.collect()

    # 3. Leakage-free temporal split (P1 episode-blocked or P2 global chronological)
    eff_embargo = max(float(dcfg.get("embargo_sec", 60.0)), float((ctx_steps + 1) * bucket_sec))
    raw_split, target_split, episode_id = split_temporal_with_purge(
        ts_end=ts_end,
        y_cls=y_cls,
        protocol=str(dcfg.get("protocol", "P1")),
        split_ratios=dcfg.get("split_ratios", [0.60, 0.10, 0.10, 0.20]),
        episode_bucket_sec=int(dcfg.get("episode_bucket_sec", 60)),
        min_episode_flows=int(dcfg.get("min_episode_flows", 5000)),
        embargo_sec=eff_embargo,
    )
    np.save(proc_dir / "raw_split.npy", raw_split)
    np.save(proc_dir / "target_split.npy", target_split)
    np.save(proc_dir / "episode_id.npy", episode_id)
    del episode_id

    train_mask = raw_split == SPLIT_TO_ID["train"]
    del raw_split
    if not train_mask.any():
        raise RuntimeError("No training samples produced after temporal splitting.")

    # 4. Fit numeric & categorical encoders strictly on Train and stream directly to memmap .npy files
    num_scaler = TrainOnlyRobustScaler(clip_quantile=float(dcfg.get("clip_quantile", 0.999)))
    num_scaler.fit(num_full_raw[train_mask])
    x_num_mm = np.lib.format.open_memmap(
        proc_dir / "x_num.npy", mode="w+", dtype=np.float32, shape=(n_flows, num_full_raw.shape[1])
    )
    num_scaler.transform_into(num_full_raw, x_num_mm)
    x_num_mm.flush()
    del x_num_mm, num_full_raw
    gc.collect()

    cat_raw = net_arrays.pop("cat_raw")
    dst_port = net_arrays.pop("dst_port")
    x_cat_mm = np.lib.format.open_memmap(
        proc_dir / "x_cat.npy", mode="w+", dtype=np.int16, shape=(n_flows, cat_raw.shape[1] + 1)
    )
    cat_encoder = TrainOnlyCategoricalEncoder(min_freq=int(dcfg.get("cat_min_freq", 20)))
    cat_encoder.fit_transform_into(cat_raw, train_mask, x_cat_mm[:, :-1])
    del cat_raw

    port_bucketer = TrainOnlyPortBucketer(top_k=int(dcfg.get("top_k_ports", 32)))
    x_cat_mm[:, -1] = port_bucketer.fit_transform(dst_port, train_mask)
    x_cat_mm.flush()
    del x_cat_mm, dst_port
    cat_vocab_sizes = cat_encoder.vocab_sizes + [port_bucketer.vocab_size]
    gc.collect()

    # 5. Ingest Linux logs and build multi-scale context grid
    bucket_tables, linux_stats = aggregate_linux_buckets(
        dirs["linux_raw"],
        bucket_sec=bucket_sec,
        stride=int(dcfg.get("linux_stride", 1)),
    )
    flow_bucket = (ts_end // bucket_sec).astype(np.int64)
    min_bucket = int(flow_bucket.min()) - ctx_steps - 2
    max_bucket = int(flow_bucket.max()) + 1

    ctx_raw, ctx_obs_mask, raw_load_score, ctx_feature_names = build_context_grid(
        bucket_tables=bucket_tables,
        min_bucket=min_bucket,
        max_bucket=max_bucket,
        bucket_sec=bucket_sec,
        roll_sec=int(dcfg.get("ctx_roll_sec", 60)),
        ewm_halflife_sec=int(dcfg.get("ctx_ewm_halflife_sec", 300)),
    )

    # Strict past alignment: for flow at bucket k, context window is [k - T, ..., k - 1]
    # In grid coordinates (0 = min_bucket), ctx_end_idx = (k - 1) - min_bucket
    ctx_end_idx = (flow_bucket - 1 - min_bucket).astype(np.int32)

    # Identify context buckets referenced by Train flows only for fitting StandardScaler
    train_ctx_ends = np.unique(ctx_end_idx[train_mask])
    Offsets = np.arange(-(ctx_steps - 1), 1, dtype=np.int32)
    train_bucket_indices = np.unique(
        np.clip((train_ctx_ends[:, None] + Offsets[None, :]).reshape(-1), 0, len(ctx_raw) - 1)
    )
    ctx_scaler = TrainOnlyStandardScaler().fit(ctx_raw[train_bucket_indices])
    ctx_scaled = ctx_scaler.transform(ctx_raw)

    flow_load_score = raw_load_score[np.clip(ctx_end_idx, 0, len(raw_load_score) - 1)]
    train_norm_mask = train_mask & (y_cls == 0)
    if train_norm_mask.any():
        high_load_thresh = float(
            np.quantile(flow_load_score[train_norm_mask], float(dcfg.get("high_load_quantile", 0.90)))
        )
    else:
        high_load_thresh = float(
            np.quantile(flow_load_score[train_mask], float(dcfg.get("high_load_quantile", 0.90)))
        )

    # Save remaining context & alignment arrays
    np.save(proc_dir / "ctx_end_idx.npy", ctx_end_idx)
    np.save(proc_dir / "flow_load_score.npy", flow_load_score.astype(np.float32))
    np.save(proc_dir / "ctx_scaled.npy", ctx_scaled)
    np.save(proc_dir / "ctx_obs_mask.npy", ctx_obs_mask)

    # Split x class table
    split_counts: Dict[str, Dict[str, int]] = {}
    for s_name in SPLIT_NAMES:
        s_id = SPLIT_TO_ID[s_name]
        m = target_split == s_id
        c_vec = np.bincount(y_cls[m], minlength=NUM_CLASSES) if m.any() else np.zeros(NUM_CLASSES, int)
        split_counts[s_name] = {CLASS_NAMES[i]: int(c_vec[i]) for i in range(NUM_CLASSES)}
        split_counts[s_name]["_total"] = int(m.sum())

    train_class_counts = np.bincount(y_cls[target_split == SPLIT_TO_ID["train"]], minlength=NUM_CLASSES)
    metadata = {
        "n_flows": int(n_flows),
        "window_len": window_len,
        "ctx_steps": ctx_steps,
        "ctx_bucket_sec": bucket_sec,
        "min_bucket": min_bucket,
        "max_bucket": max_bucket,
        "num_features": 14,
        "cat_features": 23,
        "ctx_features": int(ctx_scaled.shape[1]),
        "cat_vocab_sizes": [int(v) for v in cat_vocab_sizes],
        "ctx_feature_names": ctx_feature_names,
        "high_load_threshold": high_load_thresh,
        "train_class_counts": [int(c) for c in train_class_counts],
        "split_counts": split_counts,
        "linux_stats": linux_stats,
        "num_scaler": num_scaler.to_dict(),
        "ctx_scaler": ctx_scaler.to_dict(),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return proc_dir


class CCTBDataset:
    """Vectorized window gatherer over memory-mapped processed arrays."""

    def __init__(self, proc_dir: Path, cfg: Dict[str, Any]) -> None:
        self.proc_dir = Path(proc_dir)
        self.cfg = cfg
        with (self.proc_dir / "metadata.json").open("r", encoding="utf-8") as f:
            self.meta: Dict[str, Any] = json.load(f)

        self.window_len = int(self.meta["window_len"])
        self.ctx_steps = int(self.meta["ctx_steps"])

        num_path = self.proc_dir / "x_num.npy"
        cat_path = self.proc_dir / "x_cat.npy"
        total_feat_bytes = num_path.stat().st_size + cat_path.stat().st_size
        # Load feature matrices sequentially into RAM when <= 3.0 GB (avoids slow random 4KB page faults)
        mmap_mode = None if total_feat_bytes <= 3_000_000_000 else "r"
        self.x_num = np.load(num_path, mmap_mode=mmap_mode)
        self.x_cat = np.load(cat_path, mmap_mode=mmap_mode)
        self.ts_end = np.load(self.proc_dir / "ts_end.npy")
        self.y_cls = np.load(self.proc_dir / "y_cls.npy")
        self.y_bin = np.load(self.proc_dir / "y_bin.npy")
        self.raw_split = np.load(self.proc_dir / "raw_split.npy")
        self.target_split = np.load(self.proc_dir / "target_split.npy")
        self.episode_id = np.load(self.proc_dir / "episode_id.npy")
        self.stream_order = np.load(self.proc_dir / "stream_order.npy")
        self.stream_pos = np.load(self.proc_dir / "stream_pos.npy")
        self.stream_start = np.load(self.proc_dir / "stream_start.npy")
        self.ctx_end_idx = np.load(self.proc_dir / "ctx_end_idx.npy")
        self.flow_load_score = np.load(self.proc_dir / "flow_load_score.npy")
        self.ctx_scaled = np.load(self.proc_dir / "ctx_scaled.npy")
        self.ctx_obs_mask = np.load(self.proc_dir / "ctx_obs_mask.npy")

        self._win_offsets = np.arange(-(self.window_len - 1), 1, dtype=np.int32)
        self._ctx_offsets = np.arange(-(self.ctx_steps - 1), 1, dtype=np.int32)

        train_counts = np.asarray(self.meta["train_class_counts"], dtype=np.int64)
        minority_thresh = int(cfg["data"].get("minority_max_count", 5000))
        self.minority_classes = set(np.flatnonzero((train_counts > 0) & (train_counts <= minority_thresh)).tolist())
        self.jitter_std = float(cfg["data"].get("minority_jitter_std", 0.05))

    def get_split_indices(self, split_name: str, cap_per_class: int | None = None, max_total: int | None = None) -> np.ndarray:
        s_id = SPLIT_TO_ID[split_name]
        idx = np.flatnonzero(self.target_split == s_id).astype(np.int64)
        if len(idx) == 0:
            return idx

        if cap_per_class is not None and cap_per_class > 0:
            kept = []
            y_sub = self.y_cls[idx]
            for c in range(NUM_CLASSES):
                c_idx = idx[y_sub == c]
                if len(c_idx) > cap_per_class:
                    # Time-uniform subsampling preserves temporal evolution
                    sel = np.linspace(0, len(c_idx) - 1, num=cap_per_class, dtype=np.int64)
                    c_idx = c_idx[sel]
                if len(c_idx) > 0:
                    kept.append(c_idx)
            idx = np.sort(np.concatenate(kept)) if kept else np.zeros(0, dtype=np.int64)

        if max_total is not None and max_total > 0 and len(idx) > max_total:
            sel = np.linspace(0, len(idx) - 1, num=max_total, dtype=np.int64)
            idx = idx[sel]

        return idx

    def gather_batch(
        self,
        target_indices: np.ndarray,
        training: bool = False,
        rng: np.random.Generator | None = None,
    ) -> Dict[str, np.ndarray]:
        """Gather a batch of flow windows and strictly-past context windows."""
        b = len(target_indices)
        pos = self.stream_pos[target_indices][:, None] + self._win_offsets[None, :]  # [B, L]
        s_start = self.stream_start[target_indices][:, None]

        valid_stream = pos >= s_start
        clipped_pos = np.clip(pos, 0, len(self.stream_order) - 1)
        flow_ids = self.stream_order[clipped_pos]  # [B, L]

        # Enforce same split and same episode inside window to prevent cross-split contamination
        same_split = self.raw_split[flow_ids] == self.raw_split[target_indices][:, None]
        same_ep = self.episode_id[flow_ids] == self.episode_id[target_indices][:, None]
        m_net = valid_stream & same_split & same_ep
        m_net[:, -1] = True  # target flow itself is always valid

        safe_flow_ids = np.where(m_net, flow_ids, target_indices[:, None])
        x_net_num = np.asarray(self.x_num[safe_flow_ids], dtype=np.float32).copy()
        x_net_cat = np.asarray(self.x_cat[safe_flow_ids], dtype=np.int64).copy()

        # Zero out left-padded slots
        pad_mask = ~m_net
        if pad_mask.any():
            x_net_num[pad_mask] = 0.0
            x_net_cat[pad_mask] = 0

        # Context window [B, T, 36] and observation mask [B, T, 3]
        c_pos = self.ctx_end_idx[target_indices][:, None] + self._ctx_offsets[None, :]
        c_pos_clipped = np.clip(c_pos, 0, len(self.ctx_scaled) - 1)
        x_ctx = self.ctx_scaled[c_pos_clipped].astype(np.float32, copy=True)
        m_ctx = self.ctx_obs_mask[c_pos_clipped].astype(np.bool_, copy=True)

        y_cls = self.y_cls[target_indices].astype(np.int64)
        y_bin = self.y_bin[target_indices].astype(np.float32)

        # Optional Gaussian jitter augmentation for ultra-minority classes (e.g., mitm) during training
        if training and self.minority_classes and self.jitter_std > 0:
            is_min = np.isin(y_cls, list(self.minority_classes))
            if is_min.any():
                gen = rng if rng is not None else np.random.default_rng()
                noise = gen.normal(0.0, self.jitter_std, size=x_net_num[is_min].shape).astype(np.float32)
                noise *= m_net[is_min][..., None].astype(np.float32)
                x_net_num[is_min] += noise

        return {
            "x_net_num": x_net_num,
            "x_net_cat": x_net_cat,
            "m_net": m_net,
            "x_ctx": x_ctx,
            "m_ctx": m_ctx,
            "y_cls": y_cls,
            "y_bin": y_bin,
            "ctx_load": self.flow_load_score[target_indices].astype(np.float32),
            "ts_end": self.ts_end[target_indices].astype(np.float64),
            "flow_id": target_indices.astype(np.int64),
        }

    def iter_batches(
        self,
        split_name: str,
        batch_size: int,
        training: bool = False,
        seed: int = 42,
        max_eval_samples: int | None = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        dcfg = self.cfg["data"]
        if training:
            indices = self.get_split_indices(
                "train",
                cap_per_class=int(dcfg.get("train_cap_per_class", 250000)),
            )
            if len(indices) == 0:
                return
            rng = np.random.default_rng(seed)
            y_sub = self.y_cls[indices]
            cls_counts = np.bincount(y_sub, minlength=NUM_CLASSES).astype(np.float64)
            # Square-root inverse frequency weighting
            cls_weights = np.where(cls_counts > 0, 1.0 / np.sqrt(np.maximum(1.0, cls_counts)), 0.0)
            sample_w = cls_weights[y_sub]
            sample_w /= sample_w.sum()

            epoch_n = min(int(dcfg.get("epoch_samples", len(indices))), max(batch_size, len(indices) * 2))
            chosen = rng.choice(indices, size=epoch_n, replace=True, p=sample_w)
        else:
            cap = max_eval_samples if max_eval_samples is not None else int(dcfg.get("eval_max_samples", 300000))
            chosen = self.get_split_indices(
                split_name,
                max_total=cap,
            )
            rng = None

        for start in range(0, len(chosen), batch_size):
            batch_idx = chosen[start : start + batch_size]
            np_batch = self.gather_batch(batch_idx, training=training, rng=rng)
            yield {
                "x_net_num": torch.from_numpy(np_batch["x_net_num"]),
                "x_net_cat": torch.from_numpy(np_batch["x_net_cat"]),
                "m_net": torch.from_numpy(np_batch["m_net"]),
                "x_ctx": torch.from_numpy(np_batch["x_ctx"]),
                "m_ctx": torch.from_numpy(np_batch["m_ctx"]),
                "y_cls": torch.from_numpy(np_batch["y_cls"]),
                "y_bin": torch.from_numpy(np_batch["y_bin"]),
                "ctx_load": torch.from_numpy(np_batch["ctx_load"]),
                "ts_end": torch.from_numpy(np_batch["ts_end"]),
                "flow_id": torch.from_numpy(np_batch["flow_id"]),
            }
