"""Strict anti-leakage unit tests for CCTB-Net Phase 1-3 pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.dataset import CCTBDataset, build_processed_dataset
from data_pipeline.ingest_linux import load_and_clean_linux_kind, parse_dirty_numeric_series
from data_pipeline.scalers import (
    TrainOnlyCategoricalEncoder,
    TrainOnlyPortBucketer,
    TrainOnlyRobustScaler,
)
from data_pipeline.schema import NET_CAT_RAW_COLS, NET_NUM_RAW_COLS, SPLIT_TO_ID
from decision.aci import run_aci_stream


def _write_synthetic_ton_iot(tmp_path: Path) -> dict:
    net_dir = tmp_path / "raw_net"
    lin_dir = tmp_path / "raw_linux"
    net_dir.mkdir(parents=True, exist_ok=True)
    lin_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
    base_ts = 1554200000
    rows = []
    # Two contiguous episodes aligned on 60s boundaries: 360 normal flows (720s), then 360 scanning flows (720s)
    for i in range(720):
        is_atk = i >= 360
        ts = base_ts + i * 2
        dur = 5.0 if (i == 10) else float(rng.uniform(0.01, 0.5))
        # Inject an extreme value and unseen category only at the tail of each episode (Test split)
        in_test_tail = (320 <= i < 360) or (680 <= i < 720)
        row = {
            "ts": str(ts),
            "src_ip": f"192.168.1.{10 + (i % 4)}",
            "dst_ip": "192.168.1.100",
            "src_port": str(10000 + i),
            "dst_port": "55555" if in_test_tail else ("80" if i % 2 == 0 else "443"),
            "duration": str(dur),
            "src_bytes": "999999999" if in_test_tail else str(int(rng.integers(50, 500))),
            "dst_bytes": str(int(rng.integers(50, 500))),
            "missed_bytes": "0",
            "src_pkts": "4",
            "src_ip_bytes": "200",
            "dst_pkts": "4",
            "dst_ip_bytes": "200",
            "http_request_body_len": "0",
            "http_response_body_len": "0",
            "proto": "test_only_proto" if in_test_tail else "tcp",
            "service": "http",
            "conn_state": "SF",
            "label": "1" if is_atk else "0",
            "type": "scanning" if is_atk else "normal",
        }
        for c in NET_CAT_RAW_COLS:
            row.setdefault(c, "-")
        for c in ["dns_query", "ssl_subject", "ssl_issuer", "http_uri", "http_referrer", "http_user_agent", "weird_addl"]:
            row[c] = "SECRET_IDENTIFIER_SHOULD_BE_DROPPED"
        rows.append(row)

    # Shuffle a couple of rows inside the CSV to verify global sorting by ts_end = ts + duration
    rows[10], rows[11] = rows[11], rows[10]
    pd.DataFrame(rows).to_csv(net_dir / "Network_dataset_1.csv", index=False)

    # Linux logs with dirty suffixes ('1184K', '97%'), space values ('0456 6'), misaligned row ('apache2'), and label/type
    lin_ts = [base_ts + k * 5 for k in range(-20, 360)]
    proc_df = pd.DataFrame(
        {
            "ts": [str(t) for t in lin_ts],
            "PID": ["100"] * len(lin_ts),
            "TRUN": ["1"] * len(lin_ts),
            "TSLPI": ["5"] * len(lin_ts),
            "TSLPU": ["0"] * len(lin_ts),
            "POLI": ["norm"] * len(lin_ts),
            "NICE": ["0"] * len(lin_ts),
            "PRI": ["120"] * len(lin_ts),
            "RTPR": ["0"] * len(lin_ts),
            "CPUNR": ["1"] * len(lin_ts),
            "Status": ["N"] * len(lin_ts),
            "EXC": ["0"] * len(lin_ts),
            "State": ["R"] * len(lin_ts),
            "CPU": ["50%"] * len(lin_ts),
            "CMD": ["apache2"] * len(lin_ts),
            "label": ["1"] * len(lin_ts),
            "type": ["ddos"] * len(lin_ts),
        }
    )
    proc_df.to_csv(lin_dir / "Linux_process_1.csv", index=False)

    mem_df = pd.DataFrame(
        {
            "ts": [str(t) for t in lin_ts],
            "PID": ["100"] * len(lin_ts),
            "MINFLT": ["100K"] * len(lin_ts),
            "MAJFLT": ["2"] * len(lin_ts),
            "VSTEXT": ["11156K"] * len(lin_ts),
            "VSIZE": ["20M"] * len(lin_ts),
            "RSIZE": ["4M"] * len(lin_ts),
            "VGROW": ["0456 6"] * len(lin_ts),
            "RGROW": ["0"] * len(lin_ts),
            "MEM": ["12%"] * len(lin_ts),
            "CMD": ["apache2"] * len(lin_ts),
            "label": ["1"] * len(lin_ts),
            "type": ["ddos"] * len(lin_ts),
        }
    )
    mem_df.to_csv(lin_dir / "linux_memory1.csv", index=False)

    disk_rows = []
    for idx, t in enumerate(lin_ts):
        disk_rows.append(
            {
                "ts": str(t),
                "PID": "100",
                "RDDSK": "5420K",
                "WRDSK": "1184K",
                "WCANCL": "0",
                # Inject one corrupted misaligned string value 'apache2' in numeric column DSK
                "DSK": "apache2" if idx == 5 else "93%",
                "CMD": "firefox",
                "label": "1",
                "type": "ddos",
            }
        )
    pd.DataFrame(disk_rows).to_csv(lin_dir / "linux_disk_1.csv", index=False)

    cfg = {
        "project": {"name": "pytest_leakage", "seed": 7, "profile": "pytest"},
        "paths": {
            "network_dir": str(net_dir),
            "linux_dir": str(lin_dir),
            "data_root": str(tmp_path / "data"),
            "output_root": str(tmp_path / "outputs"),
        },
        "data": {
            "sample_frac": 1.0,
            "sample_seed": 7,
            "linux_stride": 1,
            "entity_key": "host_pair",
            "window_len": 8,
            "ctx_bucket_sec": 5,
            "ctx_steps": 6,
            "ctx_roll_sec": 30,
            "ctx_ewm_halflife_sec": 60,
            "protocol": "P1",
            "split_ratios": [0.60, 0.10, 0.10, 0.20],
            "episode_bucket_sec": 60,
            "min_episode_flows": 100,
            "embargo_sec": 35,
            "cat_min_freq": 2,
            "top_k_ports": 4,
            "clip_quantile": 0.99,
            "high_load_quantile": 0.90,
            "train_cap_per_class": 1000,
            "epoch_samples": 256,
            "eval_max_samples": 1000,
            "minority_jitter_std": 0.0,
            "minority_max_count": 10,
        },
    }
    return cfg


def test_dirty_linux_parser_and_label_drop(tmp_path: Path) -> None:
    cfg = _write_synthetic_ton_iot(tmp_path)
    vals, is_bad = parse_dirty_numeric_series(pd.Series(["1184K", "97%", "0456 6", "-", "apache2"]))
    assert np.isclose(vals.iloc[0], 1184000.0)
    assert np.isclose(vals.iloc[1], 97.0)
    assert np.isclose(vals.iloc[2], 456.0)
    assert np.isclose(vals.iloc[3], 0.0)
    assert not is_bad.iloc[:4].any()
    assert bool(is_bad.iloc[4]) is True

    disk_df, stats = load_and_clean_linux_kind(Path(cfg["paths"]["linux_dir"]), "disk")
    assert "label" not in disk_df.columns
    assert "type" not in disk_df.columns
    assert stats["dropped_bad_rows"] == 1


def test_end_to_end_anti_leakage_invariants(tmp_path: Path) -> None:
    cfg = _write_synthetic_ton_iot(tmp_path)
    proc_dir = build_processed_dataset(cfg, force=True)
    ds = CCTBDataset(proc_dir, cfg)

    # 1. Global monotonic sorting by ts_end = ts + duration
    assert np.all(np.diff(ds.ts_end) >= 0.0), "Flows must be globally sorted by ts_end = ts + duration"

    # 2. Within every episode, split order is strictly Train(0) -> Val(1) -> Cal(2) -> Test(3)
    for ep in np.unique(ds.episode_id):
        ep_mask = ds.episode_id == ep
        ep_splits = ds.raw_split[ep_mask]
        assert np.all(np.diff(ep_splits) >= 0), f"Episode {ep} splits are not chronologically ordered"

    # 3. Embargo zone check & strict past context alignment
    delta = cfg["data"]["ctx_bucket_sec"]
    for split_id in [SPLIT_TO_ID["val"], SPLIT_TO_ID["cal"], SPLIT_TO_ID["test"]]:
        target_idx = np.flatnonzero(ds.target_split == split_id)
        assert len(target_idx) > 0
        # Context end bucket k-1 must strictly precede flow's bucket k = floor(ts_end / delta)
        flow_bucket = (ds.ts_end[target_idx] // delta).astype(np.int64)
        actual_end_bucket = ds.meta["min_bucket"] + ds.ctx_end_idx[target_idx]
        assert np.all(actual_end_bucket == flow_bucket - 1), "Context window leaked current/future bucket"

        # Gather windows and verify no flow in a window belongs to a different split or episode
        batch = ds.gather_batch(target_idx, training=False)
        pos = ds.stream_pos[target_idx][:, None] + ds._win_offsets[None, :]
        flow_ids = ds.stream_order[np.clip(pos, 0, len(ds.stream_order) - 1)]
        valid_slots = batch["m_net"]
        assert np.all(ds.raw_split[flow_ids][valid_slots] == split_id)

    # 4. Train-only vocabulary & port bucketing (test-only category 'test_only_proto' -> OOV=1)
    test_idx = ds.get_split_indices("test")
    proto_col = ds.x_cat[test_idx, 0]
    assert (proto_col == 1).any(), "Unseen test category must map to OOV index 1"

    # 5. Train-only numeric scaler check (high_clip of src_bytes must not reflect test-only 999,999,999 spike)
    src_bytes_high_clip = ds.meta["num_scaler"]["high_clip"][1]
    assert src_bytes_high_clip < np.log1p(100000.0), "Numeric scaler leaked extreme test values"


def test_aci_long_run_coverage() -> None:
    rng = np.random.default_rng(99)
    n_cal, n_test, k = 500, 1500, 5
    logits_cal = rng.normal(0, 1.5, size=(n_cal, k)).astype(np.float32)
    probs_cal = np.exp(logits_cal) / np.exp(logits_cal).sum(axis=1, keepdims=True)
    y_cal = np.array([rng.choice(k, p=probs_cal[i]) for i in range(n_cal)], dtype=np.int64)

    logits_test = rng.normal(0, 1.5, size=(n_test, k)).astype(np.float32)
    probs_test = np.exp(logits_test) / np.exp(logits_test).sum(axis=1, keepdims=True)
    y_test = np.array([rng.choice(k, p=probs_test[i]) for i in range(n_test)], dtype=np.int64)

    res = run_aci_stream(probs_cal, y_cal, probs_test, y_test, alpha=0.05, eta=0.01, window_size=400)
    cov = float(res["pred_sets"][np.arange(n_test), y_test].mean())
    assert 0.90 <= cov <= 0.99
