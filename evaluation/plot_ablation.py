"""Generate three publication-quality ablation comparison SVG figures directly from metrics.json files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from engine.config import REPO_ROOT

# Global Times New Roman font configuration
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"


RUN_SPECS: List[Tuple[str, str]] = [
    ("M0", "cctb_net_full_eval"),
    ("A1", "cctb_net_full_eval_A11_jev_a"),
    ("A2", "cctb_net_full_eval_A1_wo_ctx_film"),
    ("A3", "cctb_net_full_eval_A5_wo_dynamic_center"),
    ("A4", "cctb_net_full_eval_A7_wo_dual_cons"),
    ("C1", "cctb_net_full_eval_C1_shuffled_ctx"),
]


def load_ablation_metrics() -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for alias, folder in RUN_SPECS:
        m_path = REPO_ROOT / "outputs" / "runs" / folder / "seed_42" / "metrics.json"
        with m_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        j = data["cctb_aci_jev"]
        c = data["conformal"]
        rows.append(
            {
                "alias": alias,
                "macro_f1": float(j["macro_f1"]),
                "mitm_f1": float(j["per_class_f1"]["mitm"]),
                "fpr_select": float(j["fpr_selective"]),
                "fpr_high": float(j["high_load_normal_fpr"]),
                "fnr_select": float(j["fnr_selective"]),
                "r_defer": float(j["defer_rate"]),
                "auroc_anom": float(j["auroc_anom"]),
                "set_size": float(c["avg_set_size"]),
            }
        )
    return rows


def _annotate_bars(
    ax: plt.Axes,
    bars,
    fmt: str = "{:.4f}",
    fontsize: float = 10.5,
    y_offset: float = 0.015,
) -> None:
    ylim = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + ylim * y_offset,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#222222",
        )


def _plot_single_pair(
    labels: List[str],
    vals_1: List[float],
    vals_2: List[float],
    label_1: str,
    label_2: str,
    color_1: str,
    color_2: str,
    out_stem: Path,
    legend_loc: str = "upper right",
    ylim_top: float | None = None,
) -> Path:
    x = np.arange(len(labels))
    width = 0.4
    offset = 0.25

    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    fig.patch.set_facecolor("white")

    b1 = ax.bar(x - offset, vals_1, width, label=label_1, color=color_1, edgecolor="white")
    b2 = ax.bar(x + offset, vals_2, width, label=label_2, color=color_2, edgecolor="white")

    max_val = max(max(vals_1), max(vals_2))
    top = ylim_top if ylim_top is not None else (max_val * 1.24 if max_val > 0 else 0.1)
    ax.set_ylim(0.0, top)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    ax.set_xlabel("Experimental Group", fontsize=18.0, labelpad=8)
    ax.set_ylabel("Value", fontsize=18.0, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16.0, fontweight="bold")
    ax.get_xticklabels()[0].set_color("#B03A2E")
    ax.tick_params(axis="y", labelsize=14)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc=legend_loc, fontsize=16.0, ncol=2)

    _annotate_bars(ax, b1, fmt="{:.4f}", fontsize=12)
    _annotate_bars(ax, b2, fmt="{:.4f}", fontsize=12)

    plt.tight_layout()
    svg_path = out_stem.with_suffix(".svg")
    fig.savefig(svg_path, format="svg", dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return svg_path


def plot_ablation_figures(out_dir: Path | None = None) -> List[Path]:
    rows = load_ablation_metrics()
    labels = [str(r["alias"]) for r in rows]

    save_dir = out_dir or (REPO_ROOT / "outputs" / "figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Colorblind-safe academic tones
    c_blue = "#2B5C8F"
    c_teal = "#3B9A9C"
    c_slate = "#6C7A89"
    c_orange = "#D96B27"
    c_red = "#C0392B"
    c_purple = "#7D5BA6"

    saved: List[Path] = []

    # Figure 1: Macro-F1 vs MITM-F1
    saved.append(
        _plot_single_pair(
            labels=labels,
            vals_1=[float(r["macro_f1"]) for r in rows],
            vals_2=[float(r["mitm_f1"]) for r in rows],
            label_1="Macro-F1",
            label_2="MITM-F1",
            color_1=c_blue,
            color_2=c_teal,
            out_stem=save_dir / "fig_ablation_f1",
            legend_loc="upper right",
            ylim_top=1.15,
        )
    )

    # Figure 2: FPR_select vs FPR_high (decimal)
    saved.append(
        _plot_single_pair(
            labels=labels,
            vals_1=[float(r["fpr_select"]) for r in rows],
            vals_2=[float(r["fpr_high"]) for r in rows],
            label_1=r"$\mathrm{FPR}_{\mathrm{select}}$",
            label_2=r"$\mathrm{FPR}_{\mathrm{high}}$",
            color_1=c_slate,
            color_2=c_orange,
            out_stem=save_dir / "fig_ablation_fpr",
            legend_loc="upper left",
        )
    )

    # Figure 3: FNR_select vs R_defer (decimal)
    saved.append(
        _plot_single_pair(
            labels=labels,
            vals_1=[float(r["fnr_select"]) for r in rows],
            vals_2=[float(r["r_defer"]) for r in rows],
            label_1=r"$\mathrm{FNR}_{\mathrm{select}}$",
            label_2=r"$\mathrm{R}_{\mathrm{defer}}$",
            color_1=c_red,
            color_2=c_purple,
            out_stem=save_dir / "fig_ablation_fnr_defer",
            legend_loc="upper left",
        )
    )

    return saved


if __name__ == "__main__":
    outputs = plot_ablation_figures()
    print("Saved ablation SVG figures:")
    for p_svg in outputs:
        print(f"  {p_svg}")