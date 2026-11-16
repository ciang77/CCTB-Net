"""Evaluate Static CP vs. ACI under label feedback delays (D=0, 1k, 10k), measure exact latency, and plot 3.5.1 figure."""
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
from evaluation.metrics import compute_all_metrics
from models.cctb_net import CCTBNet

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def _measure_gpu_inference_speed(cfg: Dict[str, Any], run_dir: Path, max_samples: int = 100000) -> Dict[str, float]:
    """Measure steady-state GPU forward inference throughput on GTX 1650."""
    proc_dir = get_profile_dirs(cfg)["processed_dir"]
    dataset = CCTBDataset(proc_dir, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model = CCTBNet(cfg, dataset.meta).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    bs = int(cfg["train"].get("eval_batch_size", 1024))
    batches = list(dataset.iter_batches("test", batch_size=bs, training=False, max_eval_samples=max_samples))
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"

    with torch.no_grad():
        for b in batches[:5]:
            db = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            with torch.amp.autocast("cuda", enabled=use_amp):
                _ = model(db)
    if device.type == "cuda":
        torch.cuda.synchronize()

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
    per_flow_ms = (elapsed / max(1, n_flows)) * 1000.0
    return {
        "bench_flows": int(n_flows),
        "bench_sec": round(elapsed, 3),
        "est_1m_infer_sec": round((elapsed / max(1, n_flows)) * 1_000_000, 2),
        "per_flow_infer_ms": round(per_flow_ms, 4),
    }


def _bin_mean(arr: np.ndarray, bin_size: int = 2000) -> np.ndarray:
    m = (len(arr) // bin_size) * bin_size
    return arr[:m].reshape(-1, bin_size).mean(axis=1)


def run_delay_and_timing_study() -> Tuple[Dict[str, Any], Path, Path]:
    cfg = load_config("configs/full_eval.yaml")
    run_dir = get_profile_dirs(cfg)["run_dir"]
    pred_dir = run_dir / "preds"

    val_npz = np.load(pred_dir / "val_preds.npz")
    cal_npz = np.load(pred_dir / "cal_preds.npz")
    test_npz = np.load(pred_dir / "test_preds.npz")
    with (pred_dir / "calibration_meta.json").open("r", encoding="utf-8") as f:
        cal_meta = json.load(f)

    alpha = float(cfg["decision"].get("alpha", 0.05))
    base_eta = float(cfg["decision"].get("eta", 0.005))
    aci_win = int(cfg["decision"].get("aci_window", 20000))
    aci_bins = int(cfg["decision"].get("aci_num_bins", 4096))
    score_type = str(cfg["decision"].get("score_type", "aps"))
    seed = int(cfg["project"].get("seed", 42))
    high_thresh = float(cal_meta["high_load_threshold"])

    val_cp = run_static_cp(cal_npz["probs"], cal_npz["y_cls"], val_npz["probs"], alpha=alpha, score_type=score_type, seed=seed)
    val_ev = EvidenceBatch(val_npz["probs"], val_cp["pred_sets"], val_npz["s_anom"], val_npz["p_anom"], val_npz["ctx_load"])
    judge = StructuredRuleJudge(cfg).fit(val_ev, val_npz["y_cls"])

    # Warmup numba JIT before timing
    _ = run_aci_stream(
        cal_npz["probs"][:1000], cal_npz["y_cls"][:1000],
        test_npz["probs"][:2000], test_npz["y_cls"][:2000],
        alpha=alpha, eta=base_eta, window_size=500, num_bins=aci_bins, label_delay=0, score_type=score_type, seed=seed
    )

    # Delay-compensated step size eta_D = min(base_eta, 0.2 / D) prevents delay-loop bang-bang oscillation
    configs = [
        ("Static CP", "static", 0, 0.0),
        ("ACI (D=0)", "aci", 0, base_eta),
        ("ACI (D=1,000)", "aci", 1000, min(base_eta, 0.2 / 1000)),
        ("ACI (D=10,000)", "aci", 10000, min(base_eta, 0.2 / 10000)),
    ]

    summary_rows: List[Dict[str, Any]] = []
    roll_curves: Dict[str, np.ndarray] = {}
    alpha_curves: Dict[str, np.ndarray] = {}
    size_dists: Dict[str, List[float]] = {}

    y_test = test_npz["y_cls"]
    n_test = len(y_test)
    roll_win = 20000
    kernel = np.ones(roll_win, dtype=np.float64) / float(roll_win)

    for label, mode, delay, eta_eff in configs:
        t0 = time.perf_counter()
        if mode == "static":
            res = run_static_cp(cal_npz["probs"], cal_npz["y_cls"], test_npz["probs"], alpha=alpha, score_type=score_type, seed=seed)
        else:
            res = run_aci_stream(
                cal_npz["probs"], cal_npz["y_cls"], test_npz["probs"], y_test,
                alpha=alpha, eta=eta_eff, window_size=aci_win, num_bins=aci_bins,
                label_delay=delay, score_type=score_type, seed=seed,
            )
        t_cp = time.perf_counter() - t0

        t1 = time.perf_counter()
        test_ev = EvidenceBatch(test_npz["probs"], res["pred_sets"], test_npz["s_anom"], test_npz["p_anom"], test_npz["ctx_load"])
        dec_out = run_decision_pipeline(test_ev, judge)
        t_jev = time.perf_counter() - t1

        metrics = compute_all_metrics(
            y_cls=y_test,
            probs=test_npz["probs"],
            s_anom=test_npz["s_anom"],
            ctx_load=test_npz["ctx_load"],
            high_load_threshold=high_thresh,
            pred_sets=res["pred_sets"],
            decision_out=dec_out,
        )
        covered = res["pred_sets"][np.arange(n_test), y_test].astype(np.float64)
        roll_cov = np.convolve(covered, kernel, mode="valid")
        roll_curves[label] = _bin_mean(roll_cov, bin_size=2000)
        alpha_curves[label] = _bin_mean(res["alpha_traj"], bin_size=2000)

        sz = res["pred_sets"].sum(axis=1)
        size_dists[label] = [
            float((sz == 0).mean()) * 100.0,
            float((sz == 1).mean()) * 100.0,
            float((sz == 2).mean()) * 100.0,
            float((sz >= 3).mean()) * 100.0,
        ]

        jev = metrics["cctb_aci_jev"]
        conf = metrics["conformal"]
        summary_rows.append(
            {
                "method": label,
                "delay": delay,
                "eta": eta_eff,
                "empirical_coverage": round(float(conf["empirical_coverage"]), 4),
                "rolling_cov_mean": round(float(roll_cov.mean()), 4),
                "rolling_cov_min": round(float(roll_cov.min()), 4),
                "avg_set_size": round(float(conf["avg_set_size"]), 3),
                "singleton_rate": round(float(conf["singleton_rate"]), 4),
                "empty_rate": round(float(conf["empty_rate"]), 4),
                "ambiguous_rate": round(float(conf["ambiguous_rate"]), 4),
                "macro_f1": round(float(jev["macro_f1"]), 4),
                "fpr_cons": round(float(jev["fpr"]), 4),
                "fpr_high": round(float(jev["high_load_normal_fpr"]), 4),
                "fnr_select": round(float(jev["fnr_selective"]), 4),
                "cp_sec": round(t_cp, 3),
                "jev_sec": round(t_jev, 3),
                "decision_total_sec": round(t_cp + t_jev, 3),
            }
        )

    gpu_bench = _measure_gpu_inference_speed(cfg, run_dir, max_samples=100000)

    out_json = REPO_ROOT / "outputs" / "summary_tables" / "aci_delay_sensitivity.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"gpu_inference_bench": gpu_bench, "methods": summary_rows}
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Plot 1x3 publication figure
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), dpi=300)
    fig.patch.set_facecolor("white")

    colors = {
        "Static CP": "#7F8C8D",
        "ACI (D=0)": "#2B5C8F",
        "ACI (D=1,000)": "#3B9A9C",
        "ACI (D=10,000)": "#D96B27",
    }
    styles = {
        "Static CP": "--",
        "ACI (D=0)": "-",
        "ACI (D=1,000)": "-.",
        "ACI (D=10,000)": ":",
    }

    # Panel (a): Rolling coverage trajectory (window = 20,000)
    ax = axes[0]
    for label, _, _, _ in configs:
        rc = roll_curves[label]
        x_m = np.linspace(0.02, 1.0, len(rc))
        ax.plot(x_m, rc, label=label, color=colors[label], linestyle=styles[label], linewidth=1.7)
    ax.axhline(0.95, color="#C0392B", linestyle=":", linewidth=1.2, label="目标覆盖率 (0.95)")
    ax.set_ylim(0.55, 1.02)
    ax.set_xlim(0.0, 1.0)
    ax.set_title("(a) 在线滚动经验覆盖率轨迹（W=20,000）", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(r"测试流进度 ($\times 10^6$ 条)", fontsize=10)
    ax.set_ylabel("滚动覆盖率", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="lower left", fontsize=8.0)

    # Panel (b): Adaptive alpha_t trajectory (bin-smoothed mean)
    ax = axes[1]
    for label, mode, _, _ in configs:
        if mode == "static":
            continue
        at = alpha_curves[label]
        x_m = np.linspace(0.0, 1.0, len(at))
        ax.plot(x_m, at, label=label, color=colors[label], linestyle=styles[label], linewidth=1.6)
    ax.axhline(0.05, color="#C0392B", linestyle=":", linewidth=1.2, label=r"目标 $\alpha=0.05$")
    ax.set_ylim(-0.06, 0.26)
    ax.set_xlim(0.0, 1.0)
    ax.set_title(r"(b) 自适应显著性水平 $\alpha_t$ 演化轨迹", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(r"测试流进度 ($\times 10^6$ 条)", fontsize=10)
    ax.set_ylabel(r"$\alpha_t$", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="upper left", fontsize=8.2)

    # Panel (c): Stacked prediction set size distribution
    ax = axes[2]
    m_labels = [r["method"].replace(" (", "\n(") for r in summary_rows]
    x_idx = np.arange(len(summary_rows))
    bar_w = 0.50
    d_mat = np.array([size_dists[r["method"]] for r in summary_rows])  # [4, 4]
    seg_colors = ["#95A5A6", "#2B5C8F", "#3B9A9C", "#D96B27"]
    seg_names = ["|C|=0 (空集)", "|C|=1 (直通)", "|C|=2 (双类)", "|C|≥3 (多类)"]
    bottom = np.zeros(len(summary_rows))
    for k in range(4):
        vals = d_mat[:, k]
        ax.bar(x_idx, vals, bar_w, bottom=bottom, label=seg_names[k], color=seg_colors[k], edgecolor="white")
        for i_b, v in enumerate(vals):
            if v >= 5.5:
                ax.text(
                    x_idx[i_b],
                    bottom[i_b] + v / 2.0,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7.8,
                    fontweight="bold",
                )
        bottom += vals

    for i_b, row in enumerate(summary_rows):
        ax.text(
            x_idx[i_b],
            101.8,
            f"Cov={row['empirical_coverage']:.3f}\nSz={row['avg_set_size']:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color="#222222",
        )

    ax.set_ylim(0, 134)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(m_labels, fontsize=8.6, fontweight="bold")
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("(c) 不同反馈延迟下预测集构成与覆盖率", fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("样本占比 (%)", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=7.6)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    plt.tight_layout(w_pad=2.0)
    fig_dir = REPO_ROOT / "outputs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / "fig_aci_dynamics.png"
    pdf_path = fig_dir / "fig_aci_dynamics.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return payload, png_path, pdf_path


if __name__ == "__main__":
    payload, png_p, pdf_p = run_delay_and_timing_study()
    print("GPU Benchmark:", json.dumps(payload["gpu_inference_bench"]))
    for row in payload["methods"]:
        print(row)
    print(f"Saved figures:\n  {png_p}\n  {pdf_p}")
