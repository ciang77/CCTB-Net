"""Benchmark CCTB-Net computational efficiency (latency breakdown, throughput, peak VRAM) for Section 3.5.2.

Supports:
1. Standard single-platform benchmark (`python -m evaluation.plot_efficiency`)
2. Lightweight real-window bundle export for Google Colab (`python -m evaluation.plot_efficiency --export-bundle`)
3. Dual-platform Edge (GTX 1650) vs. Server (Colab T4/L4) comparison (`python -m evaluation.plot_efficiency --compare`)
"""
from __future__ import annotations

import argparse
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_pipeline.dataset import CCTBDataset
from decision.aci import run_aci_stream, run_static_cp
from decision.secondary_judge import EvidenceBatch, StructuredRuleJudge, run_decision_pipeline
from engine.config import REPO_ROOT, get_profile_dirs, load_config
from models.cctb_net import CCTBJointLoss, CCTBNet

# Global Times New Roman font configuration (fallback gracefully on Linux/Colab if Times New Roman is absent)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

CACHE_FILENAME = "efficiency_windows.npz"


def _slugify_device(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    for prefix in ("nvidia_geforce_", "nvidia_", "tesla_"):
        if slug.startswith(prefix) and len(slug) > len(prefix):
            pass
    if "1650" in slug:
        return "gtx1650"
    if "t4" in slug:
        return "tesla_t4"
    if "l4" in slug:
        return "nvidia_l4"
    if "a100" in slug:
        return "nvidia_a100"
    return slug or "cpu"


class CachedEfficiencyDataset:
    """Drop-in lightweight dataset wrapper backed by pre-extracted real windows (`efficiency_windows.npz`)."""

    def __init__(self, proc_dir: Path) -> None:
        self.proc_dir = Path(proc_dir)
        with (self.proc_dir / "metadata.json").open("r", encoding="utf-8") as f:
            self.meta: Dict[str, Any] = json.load(f)
        cache_path = self.proc_dir / CACHE_FILENAME
        if not cache_path.exists():
            raise FileNotFoundError(f"Lightweight window cache not found: {cache_path}")
        self._npz = np.load(cache_path)

    def iter_batches(
        self,
        split_name: str,
        batch_size: int,
        training: bool = False,
        seed: int = 42,
        max_eval_samples: int | None = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        prefix = "train" if training else "test"
        total_avail = int(len(self._npz[f"{prefix}_y_cls"]))
        cap = total_avail if max_eval_samples is None else min(int(max_eval_samples), total_avail)

        keys = ["x_net_num", "x_net_cat", "m_net", "x_ctx", "m_ctx", "y_cls", "y_bin", "ctx_load"]
        arrays = {k: self._npz[f"{prefix}_{k}"][:cap] for k in keys}

        for start in range(0, cap, batch_size):
            end = min(start + batch_size, cap)
            yield {
                "x_net_num": torch.from_numpy(arrays["x_net_num"][start:end].astype(np.float32, copy=False)),
                "x_net_cat": torch.from_numpy(arrays["x_net_cat"][start:end].astype(np.int64, copy=False)),
                "m_net": torch.from_numpy(arrays["m_net"][start:end].astype(np.bool_, copy=False)),
                "x_ctx": torch.from_numpy(arrays["x_ctx"][start:end].astype(np.float32, copy=False)),
                "m_ctx": torch.from_numpy(arrays["m_ctx"][start:end].astype(np.bool_, copy=False)),
                "y_cls": torch.from_numpy(arrays["y_cls"][start:end].astype(np.int64, copy=False)),
                "y_bin": torch.from_numpy(arrays["y_bin"][start:end].astype(np.float32, copy=False)),
                "ctx_load": torch.from_numpy(arrays["ctx_load"][start:end].astype(np.float32, copy=False)),
            }


class SyntheticEfficiencyDataset:
    """Zero-upload fallback dataset using metadata.json shapes/vocabs when git-cloned on Colab without 3.4GB data."""

    def __init__(self, proc_dir: Path) -> None:
        self.proc_dir = Path(proc_dir)
        with (self.proc_dir / "metadata.json").open("r", encoding="utf-8") as f:
            self.meta: Dict[str, Any] = json.load(f)

    def iter_batches(
        self,
        split_name: str,
        batch_size: int,
        training: bool = False,
        seed: int = 42,
        max_eval_samples: int | None = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        total = 6144 if training else (50000 if max_eval_samples is None else int(max_eval_samples))
        w_len = int(self.meta.get("window_len", 16))
        c_steps = int(self.meta.get("ctx_steps", 12))
        n_num = int(self.meta.get("num_features", 14))
        n_ctx = int(self.meta.get("ctx_features", 36))
        vocabs = [max(2, int(v)) for v in self.meta.get("cat_vocab_sizes", [16] * 23)]
        n_cat = len(vocabs)

        rng = np.random.default_rng(seed)
        # Generate one canonical pool of 2048 windows and slice batches from it before timing starts
        pool_n = min(total, 2048)
        pool_num = rng.standard_normal((pool_n, w_len, n_num), dtype=np.float32)
        pool_cat = np.stack(
            [rng.integers(1, v, size=(pool_n, w_len), dtype=np.int64) for v in vocabs],
            axis=-1,
        )
        pool_m_net = np.ones((pool_n, w_len), dtype=np.bool_)
        pool_ctx = rng.standard_normal((pool_n, c_steps, n_ctx), dtype=np.float32)
        pool_m_ctx = np.ones((pool_n, c_steps, 3), dtype=np.bool_)
        pool_y_cls = rng.integers(0, 10, size=(pool_n,), dtype=np.int64)
        pool_y_bin = (pool_y_cls > 0).astype(np.float32)
        pool_load = rng.standard_normal((pool_n,), dtype=np.float32)

        for start in range(0, total, batch_size):
            b = min(batch_size, total - start)
            idx = np.arange(start, start + b) % pool_n
            yield {
                "x_net_num": torch.from_numpy(pool_num[idx].copy()),
                "x_net_cat": torch.from_numpy(pool_cat[idx].copy()),
                "m_net": torch.from_numpy(pool_m_net[idx].copy()),
                "x_ctx": torch.from_numpy(pool_ctx[idx].copy()),
                "m_ctx": torch.from_numpy(pool_m_ctx[idx].copy()),
                "y_cls": torch.from_numpy(pool_y_cls[idx].copy()),
                "y_bin": torch.from_numpy(pool_y_bin[idx].copy()),
                "ctx_load": torch.from_numpy(pool_load[idx].copy()),
            }


def load_efficiency_dataset(proc_dir: Path, cfg: Dict[str, Any]) -> Any:
    """Load full CCTBDataset if x_num.npy exists; else CachedEfficiencyDataset; else SyntheticEfficiencyDataset."""
    proc_dir = Path(proc_dir)
    if (proc_dir / "x_num.npy").exists() and (proc_dir / "x_cat.npy").exists():
        return CCTBDataset(proc_dir, cfg)
    if (proc_dir / CACHE_FILENAME).exists():
        return CachedEfficiencyDataset(proc_dir)
    print("[Efficiency] Using metadata-driven tensor generator (GitHub zero-upload mode).", flush=True)
    return SyntheticEfficiencyDataset(proc_dir)


def export_colab_bundle(
    zip_path: Path | None = None,
    max_test_windows: int = 50000,
    max_train_windows: int = 6144,
) -> Path:
    """Extract exact 50,000 test windows + 6,144 train windows and package a self-contained Colab zip bundle."""
    cfg = load_config("configs/full_eval.yaml")
    dirs = get_profile_dirs(cfg)
    proc_dir = dirs["processed_dir"]
    run_dir = dirs["run_dir"]

    print(f"[Bundle] Loading CCTBDataset from {proc_dir} ...", flush=True)
    dataset = CCTBDataset(proc_dir, cfg)

    test_idx = dataset.get_split_indices("test", max_total=max_test_windows)
    print(f"[Bundle] Gathering {len(test_idx)} real test windows ...", flush=True)
    test_batch = dataset.gather_batch(test_idx, training=False)

    train_all_idx = dataset.get_split_indices(
        "train",
        cap_per_class=int(cfg["data"].get("train_cap_per_class", 250000)),
    )
    rng = np.random.default_rng(42)
    y_sub = dataset.y_cls[train_all_idx]
    cls_counts = np.bincount(y_sub, minlength=10).astype(np.float64)
    cls_weights = np.where(cls_counts > 0, 1.0 / np.sqrt(np.maximum(1.0, cls_counts)), 0.0)
    sample_w = cls_weights[y_sub]
    sample_w /= sample_w.sum()
    train_chosen = rng.choice(train_all_idx, size=max_train_windows, replace=True, p=sample_w)
    print(f"[Bundle] Gathering {len(train_chosen)} real train windows ...", flush=True)
    train_batch = dataset.gather_batch(train_chosen, training=True, rng=rng)

    cache_path = proc_dir / CACHE_FILENAME
    print(f"[Bundle] Saving compressed window cache to {cache_path} ...", flush=True)
    np.savez_compressed(
        cache_path,
        test_x_net_num=test_batch["x_net_num"].astype(np.float32),
        test_x_net_cat=test_batch["x_net_cat"].astype(np.int16),
        test_m_net=test_batch["m_net"].astype(np.bool_),
        test_x_ctx=test_batch["x_ctx"].astype(np.float32),
        test_m_ctx=test_batch["m_ctx"].astype(np.bool_),
        test_y_cls=test_batch["y_cls"].astype(np.int64),
        test_y_bin=test_batch["y_bin"].astype(np.float32),
        test_ctx_load=test_batch["ctx_load"].astype(np.float32),
        train_x_net_num=train_batch["x_net_num"].astype(np.float32),
        train_x_net_cat=train_batch["x_net_cat"].astype(np.int16),
        train_m_net=train_batch["m_net"].astype(np.bool_),
        train_x_ctx=train_batch["x_ctx"].astype(np.float32),
        train_m_ctx=train_batch["m_ctx"].astype(np.bool_),
        train_y_cls=train_batch["y_cls"].astype(np.int64),
        train_y_bin=train_batch["y_bin"].astype(np.float32),
        train_ctx_load=train_batch["ctx_load"].astype(np.float32),
    )

    if zip_path is None:
        zip_path = REPO_ROOT / "colab_efficiency_bundle.zip"

    include_files: List[Path] = [
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / "configs" / "default.yaml",
        REPO_ROOT / "configs" / "full_eval.yaml",
        proc_dir / "metadata.json",
        cache_path,
        run_dir / "best.pt",
        run_dir / "preds" / "val_preds.npz",
        run_dir / "preds" / "cal_preds.npz",
        run_dir / "preds" / "test_preds.npz",
        run_dir / "preds" / "calibration_meta.json",
        REPO_ROOT / "outputs" / "summary_tables" / "efficiency_benchmark_gtx1650.json",
    ]
    for pkg in ("data_pipeline", "decision", "engine", "evaluation", "models"):
        include_files.extend((REPO_ROOT / pkg).glob("*.py"))

    print(f"[Bundle] Packaging self-contained Colab archive -> {zip_path} ...", flush=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in include_files:
            if fpath.exists():
                rel = fpath.relative_to(REPO_ROOT)
                zf.write(fpath, arcname=str(rel).replace("\\", "/"))

    sz_mb = zip_path.stat().st_size / (1024.0 ** 2)
    print(f"[Bundle] Done! Archive size: {sz_mb:.2f} MB ({zip_path})", flush=True)
    return zip_path


def _benchmark_gpu_configs(
    cfg: Dict[str, Any],
    dataset: Any,
    model: CCTBNet,
    criterion: CCTBJointLoss,
    device: torch.device,
) -> Dict[str, Any]:
    """Measure real GPU throughput and peak VRAM across batch sizes and AMP vs FP32."""
    batch_sizes = [128, 256, 512, 1024, 2048]
    infer_results: List[Dict[str, Any]] = []

    for bs in batch_sizes:
        batches = list(dataset.iter_batches("test", batch_size=bs, training=False, max_eval_samples=min(50000, bs * 40)))
        for use_amp in [True, False]:
            if use_amp and device.type != "cuda":
                continue
            # Warmup
            with torch.no_grad():
                for b in batches[:3]:
                    db = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        _ = model(db)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)

            n_flows = 0
            t0 = time.perf_counter()
            with torch.no_grad():
                for b in batches:
                    db = {
                        k: (
                            v.pin_memory().to(device, non_blocking=True)
                            if (device.type == "cuda" and isinstance(v, torch.Tensor))
                            else (v.to(device) if isinstance(v, torch.Tensor) else v)
                        )
                        for k, v in b.items()
                    }
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        out = model(db)
                    _ = out["logits"].float().cpu()
                    n_flows += len(b["y_cls"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            peak_mb = (
                float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
                if device.type == "cuda"
                else 0.0
            )
            infer_results.append(
                {
                    "batch_size": bs,
                    "amp": use_amp,
                    "flows": int(n_flows),
                    "sec": round(elapsed, 4),
                    "throughput_kflows_s": round((n_flows / max(1e-6, elapsed)) / 1000.0, 2),
                    "est_1m_sec": round((elapsed / max(1, n_flows)) * 1_000_000.0, 2),
                    "peak_vram_mb": round(peak_mb, 2),
                }
            )

    # Measure training peak VRAM (forward + backward + optimizer step) for bs=512 (AMP vs FP32) and bs=1024 (AMP)
    train_vram: List[Dict[str, Any]] = []
    train_specs = [
        ("Infer\n(bs=1024, AMP)", "infer", 1024, True),
        ("Train\n(bs=512, AMP)", "train", 512, True),
        ("Train\n(bs=512, FP32)", "train", 512, False),
        ("Train\n(bs=1024, AMP)", "train", 1024, True),
    ]
    for label, mode, bs, use_amp in train_specs:
        if mode == "infer":
            match = [r for r in infer_results if r["batch_size"] == bs and r["amp"] == use_amp]
            peak_mb = match[0]["peak_vram_mb"] if match else 0.0
            train_vram.append({"label": label, "mode": mode, "batch_size": bs, "amp": use_amp, "peak_vram_mb": peak_mb})
            continue

        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
        batches = list(dataset.iter_batches("train", batch_size=bs, training=True, seed=42))[:6]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for b in batches:
            db = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                out = model(db)
                loss = criterion(out, db["y_cls"], db["y_bin"])["loss"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 2)
        else:
            peak_mb = 0.0
        train_vram.append({"label": label, "mode": mode, "batch_size": bs, "amp": use_amp, "peak_vram_mb": round(peak_mb, 2)})
        model.eval()

    return {"infer_sweep": infer_results, "vram_profiles": train_vram}


def _load_or_synthesize_decision_preds(pred_dir: Path, n_flows: int = 1_000_000, seed: int = 42) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load real val/cal/test_preds.npz if present; otherwise synthesize 1M-flow arrays with matching ~14.2% uncertain rate."""
    val_p = pred_dir / "val_preds.npz"
    cal_p = pred_dir / "cal_preds.npz"
    test_p = pred_dir / "test_preds.npz"
    if val_p.exists() and cal_p.exists() and test_p.exists():
        return np.load(val_p), np.load(cal_p), np.load(test_p)

    print("[Efficiency] Synthesizing 1M-flow decision arrays for ACI + Jev-B benchmark (GitHub zero-upload mode)...", flush=True)
    rng = np.random.default_rng(seed)

    def _make_split(n: int, s: int) -> Dict[str, np.ndarray]:
        r = np.random.default_rng(s)
        y_cls = r.integers(0, 10, size=(n,), dtype=np.int64)
        logits = r.standard_normal((n, 10), dtype=np.float32)
        # Boost true class logit so ~85.8% form singletons and ~14.2% form uncertain sets under alpha=0.05
        boost = np.where(r.random(n) < 0.858, 4.2, 1.35).astype(np.float32)
        logits[np.arange(n), y_cls] += boost
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        probs = (exp_l / exp_l.sum(axis=1, keepdims=True)).astype(np.float32)
        s_anom = np.where(y_cls == 0, r.exponential(0.15, size=n), r.exponential(2.5, size=n) + 0.8).astype(np.float32)
        p_anom = (1.0 / (1.0 + np.exp(-(np.log1p(s_anom) - 0.7)))).astype(np.float32)
        ctx_load = r.standard_normal((n,), dtype=np.float32)
        return {
            "probs": probs,
            "y_cls": y_cls,
            "s_anom": s_anom,
            "p_anom": p_anom,
            "ctx_load": ctx_load,
        }

    # Use a 64k base tile repeated to 1M so initialization takes <0.3s while ACI/Jev execute on full 1,000,000 rows
    def _make_1m(s: int) -> Dict[str, np.ndarray]:
        base = _make_split(65536, s)
        reps = (n_flows + 65535) // 65536
        return {k: np.tile(v, (reps, 1))[:n_flows].copy() if v.ndim == 2 else np.tile(v, reps)[:n_flows].copy() for k, v in base.items()}

    return _make_1m(seed), _make_1m(seed + 1), _make_1m(seed + 2)


def _benchmark_decision_backend(cfg: Dict[str, Any], run_dir: Path) -> Dict[str, float]:
    """Measure exact 1M-flow runtime for Numba ACI and Jev-B secondary adjudication."""
    pred_dir = run_dir / "preds"
    seed = int(cfg["project"].get("seed", 42))
    val_npz, cal_npz, test_npz = _load_or_synthesize_decision_preds(pred_dir, n_flows=1_000_000, seed=seed)

    alpha = float(cfg["decision"].get("alpha", 0.05))
    eta = float(cfg["decision"].get("eta", 0.005))
    aci_win = int(cfg["decision"].get("aci_window", 20000))
    aci_bins = int(cfg["decision"].get("aci_num_bins", 4096))
    score_type = str(cfg["decision"].get("score_type", "aps"))

    val_cp = run_static_cp(cal_npz["probs"], cal_npz["y_cls"], val_npz["probs"], alpha=alpha, score_type=score_type, seed=seed)
    val_ev = EvidenceBatch(val_npz["probs"], val_cp["pred_sets"], val_npz["s_anom"], val_npz["p_anom"], val_npz["ctx_load"])
    judge = StructuredRuleJudge(cfg).fit(val_ev, val_npz["y_cls"])

    # Warmup Numba JIT
    _ = run_aci_stream(
        cal_npz["probs"][:1000], cal_npz["y_cls"][:1000],
        test_npz["probs"][:2000], test_npz["y_cls"][:2000],
        alpha=alpha, eta=eta, window_size=500, num_bins=aci_bins, label_delay=0, score_type=score_type, seed=seed,
    )

    t0 = time.perf_counter()
    res = run_aci_stream(
        cal_npz["probs"], cal_npz["y_cls"], test_npz["probs"], test_npz["y_cls"],
        alpha=alpha, eta=eta, window_size=aci_win, num_bins=aci_bins, label_delay=0, score_type=score_type, seed=seed,
    )
    t_aci = time.perf_counter() - t0

    t1 = time.perf_counter()
    test_ev = EvidenceBatch(test_npz["probs"], res["pred_sets"], test_npz["s_anom"], test_npz["p_anom"], test_npz["ctx_load"])
    _ = run_decision_pipeline(test_ev, judge)
    t_jev = time.perf_counter() - t1

    return {
        "n_test_flows": int(len(test_npz["y_cls"])),
        "aci_sec": round(t_aci, 3),
        "jev_sec": round(t_jev, 3),
        "decision_total_sec": round(t_aci + t_jev, 3),
    }


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=14.0)


def _save_and_show(fig: plt.Figure, svg_path: Path) -> Path:
    plt.tight_layout()
    fig.savefig(svg_path, format="svg", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return svg_path


def generate_dual_platform_comparison(
    edge_json_path: Path | None = None,
    server_json_path: Path | None = None,
) -> Tuple[Path, Path, Path]:
    """Generate Edge (GTX 1650) vs. Server (Colab T4/L4) comparison tables (.md, .csv) and dual-panel SVG figure."""
    tables_dir = REPO_ROOT / "outputs" / "summary_tables"
    fig_dir = REPO_ROOT / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if edge_json_path is None:
        edge_json_path = tables_dir / "efficiency_benchmark_gtx1650.json"

    if server_json_path is None:
        candidates = sorted(
            [
                p
                for p in tables_dir.glob("efficiency_benchmark_*.json")
                if p.name != "efficiency_benchmark_gtx1650.json"
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            server_json_path = candidates[0]
        else:
            default_json = tables_dir / "efficiency_benchmark.json"
            if default_json.exists():
                with default_json.open("r", encoding="utf-8") as f:
                    tmp = json.load(f)
                if "1650" not in tmp.get("gpu_device", ""):
                    server_json_path = default_json

    if not edge_json_path.exists():
        raise FileNotFoundError(f"Edge benchmark JSON not found: {edge_json_path}")
    if server_json_path is None or not server_json_path.exists():
        raise FileNotFoundError(
            "Server (T4/L4) benchmark JSON not found in outputs/summary_tables/. "
            "Run plot_efficiency on Colab first or pass --server-json <path>."
        )

    with edge_json_path.open("r", encoding="utf-8") as f:
        edge = json.load(f)
    with server_json_path.open("r", encoding="utf-8") as f:
        server = json.load(f)

    edge_name = edge.get("gpu_device", "NVIDIA GeForce GTX 1650")
    server_name = server.get("gpu_device", "Server GPU")

    def _index_sweep(data: Dict[str, Any]) -> Dict[Tuple[int, bool], Dict[str, Any]]:
        return {(int(r["batch_size"]), bool(r["amp"])): r for r in data.get("infer_sweep", [])}

    e_map = _index_sweep(edge)
    s_map = _index_sweep(server)
    batch_sizes = sorted({int(r["batch_size"]) for r in edge.get("infer_sweep", [])})

    # Write CSV comparison
    csv_path = tables_dir / "efficiency_dual_platform.csv"
    csv_lines = [
        "batch_size,edge_gpu,edge_amp_kflows_s,edge_fp32_kflows_s,edge_amp_speedup,edge_vram_mb,"
        "server_gpu,server_amp_kflows_s,server_fp32_kflows_s,server_amp_speedup,server_vram_mb,server_vs_edge_amp_gain"
    ]
    md_rows = []
    for bs in batch_sizes:
        e_amp = e_map.get((bs, True), {})
        e_fp32 = e_map.get((bs, False), {})
        s_amp = s_map.get((bs, True), {})
        s_fp32 = s_map.get((bs, False), {})

        ea_tp = float(e_amp.get("throughput_kflows_s", 0.0))
        ef_tp = float(e_fp32.get("throughput_kflows_s", 0.0))
        sa_tp = float(s_amp.get("throughput_kflows_s", 0.0))
        sf_tp = float(s_fp32.get("throughput_kflows_s", 0.0))

        e_ratio = ea_tp / max(1e-6, ef_tp) if ef_tp > 0 else 0.0
        s_ratio = sa_tp / max(1e-6, sf_tp) if sf_tp > 0 else 0.0
        cross_gain = sa_tp / max(1e-6, ea_tp) if ea_tp > 0 else 0.0

        e_vram = float(e_amp.get("peak_vram_mb", 0.0))
        s_vram = float(s_amp.get("peak_vram_mb", 0.0))

        csv_lines.append(
            f"{bs},{edge_name},{ea_tp:.2f},{ef_tp:.2f},{e_ratio:.2f}x,{e_vram:.2f},"
            f"{server_name},{sa_tp:.2f},{sf_tp:.2f},{s_ratio:.2f}x,{s_vram:.2f},{cross_gain:.2f}x"
        )
        md_rows.append(
            f"| {bs} | {ea_tp:.2f} | {ef_tp:.2f} | {e_ratio:.2f}x | {sa_tp:.2f} | {sf_tp:.2f} | **{s_ratio:.2f}x** | **{cross_gain:.2f}x** | {s_vram:.1f} MB |"
        )

    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    e_lat = edge["latency_1m_flows"]
    s_lat = server["latency_1m_flows"]
    md_path = tables_dir / "efficiency_dual_platform.md"
    md_content = [
        "# CCTB-Net 双平台计算效率与吞吐量对比表（边缘端 vs. 云/服务端）",
        "",
        f"- **边缘受限平台（Edge）**：{edge_name}（4 GB VRAM，TU117 核心，无 Tensor Cores）",
        f"- **云/服务端平台（Server）**：{server_name}（配备硬件 Tensor Cores）",
        f"- **模型规模**：参数量 `{edge['num_parameters']:,}`（约 {edge['num_parameters']/1e3:.1f}K），FP32 权重体积 `{edge['model_size_mb_fp32']:.3f} MB`",
        "",
        "## 1. 百万流（1M Flows）端到端时延拆解对比（Batch Size = 1024, AMP）",
        "",
        f"| 阶段 / 指标 | 边缘端 ({edge_name}) | 服务端 ({server_name}) | 加速比 (Edge / Server) |",
        "|---|---:|---:|---:|",
        f"| 骨干网络推理 (Backbone GPU) | {e_lat['backbone_gpu_sec']:.2f} s | {s_lat['backbone_gpu_sec']:.2f} s | {e_lat['backbone_gpu_sec']/max(1e-6, s_lat['backbone_gpu_sec']):.2f}x |",
        f"| 在线保形推断 (ACI, Numba JIT) | {e_lat['aci_numba_sec']:.3f} s | {s_lat['aci_numba_sec']:.3f} s | {e_lat['aci_numba_sec']/max(1e-6, s_lat['aci_numba_sec']):.2f}x |",
        f"| 二次证据裁决 (Jev-B) | {e_lat['jev_b_sec']:.3f} s | {s_lat['jev_b_sec']:.3f} s | {e_lat['jev_b_sec']/max(1e-6, s_lat['jev_b_sec']):.2f}x |",
        f"| **端到端总时延 (End-to-End Total)** | **{e_lat['end_to_end_sec']:.2f} s** | **{s_lat['end_to_end_sec']:.2f} s** | **{e_lat['end_to_end_sec']/max(1e-6, s_lat['end_to_end_sec']):.2f}x** |",
        f"| **单流平均时延 (Per-Flow Latency)** | **{e_lat['per_flow_ms']:.4f} ms** | **{s_lat['per_flow_ms']:.4f} ms** | - |",
        f"| **端到端总吞吐量 (End-to-End Throughput)** | **{e_lat['throughput_kflows_s']:.2f} kflows/s** | **{s_lat['throughput_kflows_s']:.2f} kflows/s** | **{s_lat['throughput_kflows_s']/max(1e-6, e_lat['throughput_kflows_s']):.2f}x** |",
        "",
        "## 2. 不同批大小（Batch Size）下 AMP (FP16) 与 FP32 骨干吞吐量对比 (kflows/s)",
        "",
        f"| Batch Size | Edge AMP | Edge FP32 | Edge AMP/FP32 | Server AMP | Server FP32 | Server AMP/FP32 | Server vs. Edge (AMP) | Peak VRAM |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *md_rows,
        "",
    ]
    md_path.write_text("\n".join(md_content), encoding="utf-8")

    # Plot dual-panel comparison SVG
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=600)
    fig.patch.set_facecolor("white")

    c_server_amp = "#1F4E78"
    c_server_fp32 = "#5B9BD5"
    c_edge_amp = "#C55A11"
    c_edge_fp32 = "#F4B183"

    bs_x = np.arange(len(batch_sizes))
    ax1.plot(
        bs_x,
        [s_map[(b, True)]["throughput_kflows_s"] for b in batch_sizes],
        marker="o",
        linewidth=2.4,
        markersize=7.5,
        color=c_server_amp,
        label=f"{server_name} (AMP FP16)",
    )
    ax1.plot(
        bs_x,
        [s_map[(b, False)]["throughput_kflows_s"] for b in batch_sizes],
        marker="s",
        linestyle="--",
        linewidth=2.0,
        markersize=6.5,
        color=c_server_fp32,
        label=f"{server_name} (FP32)",
    )
    ax1.plot(
        bs_x,
        [e_map[(b, True)]["throughput_kflows_s"] for b in batch_sizes],
        marker="^",
        linewidth=2.2,
        markersize=7.0,
        color=c_edge_amp,
        label=f"{edge_name} (AMP FP16)",
    )
    ax1.plot(
        bs_x,
        [e_map[(b, False)]["throughput_kflows_s"] for b in batch_sizes],
        marker="d",
        linestyle="--",
        linewidth=2.0,
        markersize=6.5,
        color=c_edge_fp32,
        label=f"{edge_name} (FP32)",
    )

    all_tp = [
        r["throughput_kflows_s"]
        for r in (edge.get("infer_sweep", []) + server.get("infer_sweep", []))
    ]
    ax1.set_ylim(0.0, max(all_tp) * 1.25)
    ax1.set_xticks(bs_x)
    ax1.set_xticklabels([str(b) for b in batch_sizes], fontsize=13.0, fontweight="bold")
    ax1.set_xlabel("Batch Size", fontsize=15.0, labelpad=8)
    ax1.set_ylabel("Backbone Throughput (k flows/s)", fontsize=15.0, labelpad=8)
    ax1.set_title("(a) Dual-Platform Throughput Scaling (AMP vs. FP32)", fontsize=15.0, fontweight="bold", pad=10)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(frameon=False, loc="upper left", fontsize=11.5)
    _style_axis(ax1)

    # Panel (b): 1M-Flow End-to-End Latency Comparison
    stages = ["Backbone\n(GPU AMP)", "ACI\n(Numba)", "Jev-B\nAdjudicator", "End-to-End\nTotal"]
    e_vals = [e_lat["backbone_gpu_sec"], e_lat["aci_numba_sec"], e_lat["jev_b_sec"], e_lat["end_to_end_sec"]]
    s_vals = [s_lat["backbone_gpu_sec"], s_lat["aci_numba_sec"], s_lat["jev_b_sec"], s_lat["end_to_end_sec"]]

    x_pos = np.arange(len(stages))
    width = 0.35
    bars_e = ax2.bar(x_pos - width / 2, e_vals, width=width, color=c_edge_amp, label=f"Edge ({edge_name})")
    bars_s = ax2.bar(x_pos + width / 2, s_vals, width=width, color=c_server_amp, label=f"Server ({server_name})")

    max_lat = max(max(e_vals), max(s_vals))
    for b, val in zip(bars_e, e_vals):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            val + max_lat * 0.02,
            f"{val:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10.5,
            color="#222222",
        )
    for b, val in zip(bars_s, s_vals):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            val + max_lat * 0.02,
            f"{val:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color=c_server_amp,
        )

    ax2.set_ylim(0.0, max_lat * 1.25)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stages, fontsize=12.5, fontweight="bold")
    ax2.set_xlabel("Pipeline Stage (1M Flows)", fontsize=15.0, labelpad=8)
    ax2.set_ylabel("Latency (s)", fontsize=15.0, labelpad=8)
    ax2.set_title("(b) 1M-Flow End-to-End Latency Breakdown", fontsize=15.0, fontweight="bold", pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.legend(frameon=False, loc="upper left", fontsize=12.0)
    _style_axis(ax2)

    dual_svg = _save_and_show(fig, fig_dir / "fig_efficiency_dual_platform.svg")
    return md_path, csv_path, dual_svg


def run_efficiency_study() -> Tuple[Dict[str, Any], List[Path]]:
    cfg = load_config("configs/full_eval.yaml")
    dirs = get_profile_dirs(cfg)
    run_dir = dirs["run_dir"]
    dataset = load_efficiency_dataset(dirs["processed_dir"], cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model = CCTBNet(cfg, dataset.meta).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    criterion = CCTBJointLoss(cfg, dataset.meta["train_class_counts"]).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    gpu_bench = _benchmark_gpu_configs(cfg, dataset, model, criterion, device)
    dec_bench = _benchmark_decision_backend(cfg, run_dir)

    # Pull 1M-flow inference sec for eval_batch_size=1024, AMP=True
    infer_1024 = [r for r in gpu_bench["infer_sweep"] if r["batch_size"] == 1024 and r["amp"]][0]
    backbone_sec = float(infer_1024["est_1m_sec"])
    aci_sec = float(dec_bench["aci_sec"])
    jev_sec = float(dec_bench["jev_sec"])
    total_sec = round(backbone_sec + aci_sec + jev_sec, 2)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_slug = _slugify_device(gpu_name)

    payload = {
        "num_parameters": int(n_params),
        "model_size_mb_fp32": round((n_params * 4) / (1024.0 ** 2), 3),
        "gpu_device": gpu_name,
        "latency_1m_flows": {
            "backbone_gpu_sec": backbone_sec,
            "aci_numba_sec": aci_sec,
            "jev_b_sec": jev_sec,
            "end_to_end_sec": total_sec,
            "per_flow_ms": round((total_sec / 1_000_000.0) * 1000.0, 4),
            "throughput_kflows_s": round(1000.0 / max(1e-6, total_sec), 2),
        },
        **gpu_bench,
        "decision_backend": dec_bench,
    }

    tables_dir = REPO_ROOT / "outputs" / "summary_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_json = tables_dir / "efficiency_benchmark.json"
    tagged_json = tables_dir / f"efficiency_benchmark_{gpu_slug}.json"
    for p_json in (out_json, tagged_json):
        with p_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    fig_dir = REPO_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved_svgs: List[Path] = []

    c_blue = "#2B5C8F"
    c_teal = "#3B9A9C"
    c_orange = "#D96B27"
    c_slate = "#6C7A89"
    c_dark = "#34495E"

    # Figure 1: 1M-Flow End-to-End Latency Breakdown
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    fig.patch.set_facecolor("white")
    stage_labels = ["Backbone\n(GPU AMP)", "ACI\n(Numba)", "Jev-B\nAdjudicator", "End-to-End\nTotal"]
    stage_vals = [backbone_sec, aci_sec, jev_sec, total_sec]
    stage_colors = [c_blue, c_teal, c_orange, c_dark]
    bars = ax.bar(np.arange(4), stage_vals, width=0.50, color=stage_colors, edgecolor="white")
    for idx_b, (b, val) in enumerate(zip(bars, stage_vals)):
        pct = (val / total_sec) * 100.0
        txt = f"{val:.2f} s\n({pct:.1f}%)" if idx_b < 3 else f"{val:.2f} s\n(100%)"
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            val + total_sec * 0.025,
            txt,
            ha="center",
            va="bottom",
            fontsize=12.0,
            fontweight="bold" if idx_b == 3 else "normal",
            color="#222222",
        )
    ax.set_ylim(0.0, total_sec * 1.28)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(stage_labels, fontsize=14.0, fontweight="bold")
    ax.set_xlabel("Pipeline Stage", fontsize=18.0, labelpad=8)
    ax.set_ylabel("Latency (s)", fontsize=18.0, labelpad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    _style_axis(ax)
    saved_svgs.append(_save_and_show(fig, fig_dir / "fig_efficiency_latency.svg"))

    # Figure 2: Backbone Throughput vs. Batch Size (AMP vs FP32)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    fig.patch.set_facecolor("white")
    amp_rows = [r for r in gpu_bench["infer_sweep"] if r["amp"]]
    fp32_rows = [r for r in gpu_bench["infer_sweep"] if not r["amp"]]
    batch_sizes = [r["batch_size"] for r in amp_rows]
    bs_x = np.arange(len(batch_sizes))
    ax.plot(
        bs_x,
        [r["throughput_kflows_s"] for r in amp_rows],
        marker="o",
        linewidth=2.2,
        markersize=7.0,
        color=c_blue,
        label="AMP (FP16)",
    )
    for xi, r in zip(bs_x, amp_rows):
        ax.text(xi, r["throughput_kflows_s"] - 4.5, f"{r['throughput_kflows_s']:.1f}k", ha="center", va="top", fontsize=12.0, color=c_blue)

    if fp32_rows:
        ax.plot(
            bs_x,
            [r["throughput_kflows_s"] for r in fp32_rows],
            marker="s",
            linestyle="--",
            linewidth=2.0,
            markersize=7.0,
            color=c_slate,
            label="FP32",
        )
        for xi, r in zip(bs_x, fp32_rows):
            ax.text(xi, r["throughput_kflows_s"] + 2.8, f"{r['throughput_kflows_s']:.1f}k", ha="center", va="bottom", fontsize=12.0, color=c_slate)

    max_tp = max([r["throughput_kflows_s"] for r in gpu_bench["infer_sweep"]])
    ax.set_ylim(0.0, max_tp * 1.24)
    ax.set_xticks(bs_x)
    ax.set_xticklabels([str(b) for b in batch_sizes], fontsize=16.0, fontweight="bold")
    ax.set_xlabel("Batch Size", fontsize=18.0, labelpad=8)
    ax.set_ylabel("Throughput (k flows/s)", fontsize=18.0, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="upper left", fontsize=16.0)
    _style_axis(ax)
    saved_svgs.append(_save_and_show(fig, fig_dir / "fig_efficiency_throughput.svg"))

    # Figure 3: Peak GPU Memory Footprint vs. Hardware Budget
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    fig.patch.set_facecolor("white")
    v_labels = [v["label"] for v in gpu_bench["vram_profiles"]]
    v_vals = [v["peak_vram_mb"] for v in gpu_bench["vram_profiles"]]
    v_colors = [c_teal, c_blue, c_slate, c_orange]
    v_bars = ax.bar(np.arange(len(v_labels)), v_vals, width=0.50, color=v_colors, edgecolor="white")
    max_v = max(v_vals) if v_vals else 100.0
    for b, val in zip(v_bars, v_vals):
        pct_4g = (val / 4096.0) * 100.0
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            val + max_v * 0.025,
            f"{val:.1f} MB\n({pct_4g:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=12.0,
            color="#222222",
        )
    ax.set_ylim(0.0, max_v * 1.28)
    ax.set_xticks(np.arange(len(v_labels)))
    ax.set_xticklabels(v_labels, fontsize=13.0, fontweight="bold")
    ax.set_xlabel("Execution Mode", fontsize=18.0, labelpad=8)
    ax.set_ylabel("Peak VRAM (MB)", fontsize=18.0, labelpad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.text(
        0.04,
        0.95,
        f"Params: {n_params / 1e3:.1f}K\nSize: {payload['model_size_mb_fp32']:.2f} MB",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13.0,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F4F6F7", edgecolor="#BDC3C7", linewidth=0.8),
    )
    _style_axis(ax)
    saved_svgs.append(_save_and_show(fig, fig_dir / "fig_efficiency_vram.svg"))

    # Automatically generate dual-platform comparison if edge baseline is present and current device is different
    edge_json = tables_dir / "efficiency_benchmark_gtx1650.json"
    if edge_json.exists() and gpu_slug != "gtx1650":
        try:
            _, _, dual_svg = generate_dual_platform_comparison(edge_json, tagged_json)
            saved_svgs.append(dual_svg)
        except Exception as exc:
            print(f"[Warn] Dual-platform comparison skipped: {exc}", flush=True)

    return payload, saved_svgs


def main() -> None:
    parser = argparse.ArgumentParser(description="CCTB-Net computational efficiency benchmark & dual-platform comparison.")
    parser.add_argument(
        "--export-bundle",
        action="store_true",
        help="Extract 50,000 test windows + 6,144 train windows and package colab_efficiency_bundle.zip",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate Edge (GTX 1650) vs. Server (Colab T4/L4) comparison tables and figure",
    )
    parser.add_argument("--edge-json", type=str, default=None, help="Path to edge platform JSON")
    parser.add_argument("--server-json", type=str, default=None, help="Path to server platform JSON")
    args = parser.parse_args()

    if args.export_bundle:
        export_colab_bundle()
        return

    if args.compare:
        edge_p = Path(args.edge_json) if args.edge_json else None
        srv_p = Path(args.server_json) if args.server_json else None
        md_p, csv_p, svg_p = generate_dual_platform_comparison(edge_p, srv_p)
        print(f"Saved dual-platform Markdown table: {md_p}")
        print(f"Saved dual-platform CSV table:      {csv_p}")
        print(f"Saved dual-platform SVG figure:     {svg_p}")
        return

    payload, svg_paths = run_efficiency_study()
    print(json.dumps(payload, indent=2))
    print("Saved SVG figures:")
    for p_svg in svg_paths:
        print(f"  {p_svg}")


if __name__ == "__main__":
    main()
