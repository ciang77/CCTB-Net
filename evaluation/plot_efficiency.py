"""Benchmark CCTB-Net computational efficiency (latency breakdown, throughput, peak VRAM) for Section 3.5.2."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_pipeline.dataset import CCTBDataset
from decision.aci import run_aci_stream, run_static_cp
from decision.secondary_judge import EvidenceBatch, StructuredRuleJudge, run_decision_pipeline
from engine.config import REPO_ROOT, get_profile_dirs, load_config
from models.cctb_net import CCTBJointLoss, CCTBNet

# Global Times New Roman font configuration
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"


def _benchmark_gpu_configs(
    cfg: Dict[str, Any],
    dataset: CCTBDataset,
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


def _benchmark_decision_backend(cfg: Dict[str, Any], run_dir: Path) -> Dict[str, float]:
    """Measure exact 1M-flow runtime for Numba ACI and Jev-B secondary adjudication."""
    pred_dir = run_dir / "preds"
    val_npz = np.load(pred_dir / "val_preds.npz")
    cal_npz = np.load(pred_dir / "cal_preds.npz")
    test_npz = np.load(pred_dir / "test_preds.npz")

    alpha = float(cfg["decision"].get("alpha", 0.05))
    eta = float(cfg["decision"].get("eta", 0.005))
    aci_win = int(cfg["decision"].get("aci_window", 20000))
    aci_bins = int(cfg["decision"].get("aci_num_bins", 4096))
    score_type = str(cfg["decision"].get("score_type", "aps"))
    seed = int(cfg["project"].get("seed", 42))

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
    plt.show()
    plt.close(fig)
    return svg_path


def run_efficiency_study() -> Tuple[Dict[str, Any], List[Path]]:
    cfg = load_config("configs/full_eval.yaml")
    dirs = get_profile_dirs(cfg)
    run_dir = dirs["run_dir"]
    dataset = CCTBDataset(dirs["processed_dir"], cfg)

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

    payload = {
        "num_parameters": int(n_params),
        "model_size_mb_fp32": round((n_params * 4) / (1024.0 ** 2), 3),
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
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

    out_json = REPO_ROOT / "outputs" / "summary_tables" / "efficiency_benchmark.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
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

    return payload, saved_svgs


if __name__ == "__main__":
    payload, svg_paths = run_efficiency_study()
    print(json.dumps(payload, indent=2))
    print("Saved SVG figures:")
    for p_svg in svg_paths:
        print(f"  {p_svg}")