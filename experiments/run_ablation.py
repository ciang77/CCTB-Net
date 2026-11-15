"""Automated ablation matrix runner and paper summary table exporter (Phase 4 & 5)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from data_pipeline.dataset import CCTBDataset, build_processed_dataset
from data_pipeline.schema import CLASS_NAMES
from engine.config import apply_overrides, get_profile_dirs, load_config, resolve_path
from engine.trainer import Trainer
from experiments.run_decision import execute_decision_stage

# Each entry: (label_for_paper, type, overrides)
# type in {"train", "reuse_a0", "argmax_a0", "decision_a0"}
ABLATION_SPECS: Dict[str, Dict[str, Any]] = {
    "A0_full_cctb": {
        "label": "CCTB-Net (Full: FiLM + TCN + Dual-Task + ACI + Jev-B)",
        "mode": "reuse_a0",
        "overrides": [],
    },
    "A8_wo_aci_jev": {
        "label": "w/o ACI + Jev (Argmax Baseline)",
        "mode": "argmax_a0",
        "overrides": [],
    },
    "A11_jev_a": {
        "label": "CCTB-Net + ACI + Jev-A (Bayesian CPT)",
        "mode": "decision_a0",
        "overrides": ["decision.judge_type=jev_a"],
    },
    "A1_wo_ctx_film": {
        "label": "w/o Context & FiLM",
        "mode": "train",
        "overrides": ["model.use_ctx=false"],
    },
    "A5_wo_dynamic_center": {
        "label": "w/o Dynamic Center Anomaly Branch",
        "mode": "train",
        "overrides": ["model.use_anomaly_branch=false"],
    },
    "A7_wo_dual_cons": {
        "label": "w/o Dual-Task Consistency Loss",
        "mode": "train",
        "overrides": ["loss.lambda_cons=0.0"],
    },
    "C1_shuffled_ctx": {
        "label": "Shuffled Context Control",
        "mode": "train",
        "overrides": ["model.shuffle_ctx=true"],
    },
    "A2_concat_fusion": {
        "label": "Concat Fusion (instead of FiLM)",
        "mode": "train",
        "overrides": ["model.fusion=concat"],
    },
}


def _extract_row(variant: str, label: str, metrics: Dict[str, Any], use_argmax: bool = False) -> Dict[str, Any]:
    det = metrics["argmax_baseline"] if use_argmax else metrics["cctb_aci_jev"]
    conf = metrics["conformal"]
    return {
        "variant": variant,
        "description": label,
        "macro_f1": round(float(det["macro_f1"]), 4),
        "binary_f1": round(float(det["binary_f1"]), 4),
        "auroc_prob": round(float(det["auroc_prob"]), 4),
        "auroc_anom": round(float(det["auroc_anom"]), 4),
        "fpr_cons": round(float(det["fpr"]), 4),
        "fpr_select": round(float(det["fpr_selective"]), 4),
        "fnr_select": round(float(det["fnr_selective"]), 4),
        "high_load_normal_fpr": round(float(det["high_load_normal_fpr"]), 4),
        "defer_rate": round(float(det["defer_rate"]), 4),
        "empirical_coverage": None if use_argmax else round(float(conf["empirical_coverage"]), 4),
        "avg_set_size": None if use_argmax else round(float(conf["avg_set_size"]), 3),
        "per_class_f1": det.get("per_class_f1", {}),
    }


def _save_summary_tables(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    main_cols = [
        "variant",
        "description",
        "macro_f1",
        "binary_f1",
        "auroc_prob",
        "auroc_anom",
        "fpr_cons",
        "fpr_select",
        "fnr_select",
        "high_load_normal_fpr",
        "defer_rate",
        "empirical_coverage",
        "avg_set_size",
    ]
    csv_path = out_dir / "main_and_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=main_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in main_cols})

    md_path = out_dir / "main_and_ablation.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# CCTB-Net 主实验与消融实验汇总表（P1 主协议）\n\n")
        f.write(
            "| Variant | 说明 | Macro-F1 | Binary-F1 | AUROC(Prob) | AUROC(Anom) | "
            "FPR(Select) | FPR(Cons) | FNR(Select) | High-Load FPR | Coverage | Set Size |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            cov = "-" if r["empirical_coverage"] is None else f"{r['empirical_coverage']:.4f}"
            sz = "-" if r["avg_set_size"] is None else f"{r['avg_set_size']:.3f}"
            f.write(
                f"| {r['variant']} | {r['description']} | {r['macro_f1']:.4f} | {r['binary_f1']:.4f} | "
                f"{r['auroc_prob']:.4f} | {r['auroc_anom']:.4f} | {r['fpr_select']:.4f} | {r['fpr_cons']:.4f} | "
                f"{r['fnr_select']:.4f} | {r['high_load_normal_fpr']:.4f} | {cov} | {sz} |\n"
            )

    # Per-class F1 table
    cls_csv = out_dir / "per_class_f1.csv"
    with cls_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant"] + CLASS_NAMES)
        w.writeheader()
        for r in rows:
            row_d = {"variant": r["variant"]}
            for c in CLASS_NAMES:
                v = r["per_class_f1"].get(c)
                row_d[c] = "" if v is None else f"{float(v):.4f}"
            w.writerow(row_d)

    cls_md = out_dir / "per_class_f1.md"
    with cls_md.open("w", encoding="utf-8") as f:
        f.write("# 各类别 F1 对比表\n\n")
        f.write("| Variant | " + " | ".join(CLASS_NAMES) + " |\n")
        f.write("|---|" + "|".join(["---:"] * len(CLASS_NAMES)) + "|\n")
        for r in rows:
            vals = [
                f"{float(r['per_class_f1'][c]):.4f}" if c in r["per_class_f1"] else "-"
                for c in CLASS_NAMES
            ]
            f.write(f"| {r['variant']} | " + " | ".join(vals) + " |\n")

    with (out_dir / "ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCTB-Net ablation suite and export summary tables.")
    parser.add_argument("--config", type=str, default="configs/full_eval.yaml")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[
            "A0_full_cctb",
            "A8_wo_aci_jev",
            "A11_jev_a",
            "A1_wo_ctx_film",
            "A5_wo_dynamic_center",
            "A7_wo_dual_cons",
            "C1_shuffled_ctx",
        ],
    )
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for newly trained ablation variants.")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience for ablation variants.")
    parser.add_argument("--summarize-only", action="store_true", help="Only aggregate existing metrics.json files.")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    a0_dirs = get_profile_dirs(base_cfg)
    a0_run_dir = a0_dirs["run_dir"]
    a0_metrics_path = a0_run_dir / "metrics.json"

    dataset = None
    if not args.summarize_only:
        proc_dir = build_processed_dataset(base_cfg, force=False)
        dataset = CCTBDataset(proc_dir, base_cfg)

    for name in args.variants:
        if name not in ABLATION_SPECS:
            continue
        spec = ABLATION_SPECS[name]
        mode = spec["mode"]
        if args.summarize_only:
            continue

        if mode == "reuse_a0":
            if not a0_metrics_path.exists():
                raise FileNotFoundError(f"A0 metrics not found at {a0_metrics_path}. Run A0 first.")
            print(f"[Ablation] Reusing A0 metrics from {a0_metrics_path}", flush=True)
        elif mode == "argmax_a0":
            print("[Ablation] A8_wo_aci_jev uses argmax_baseline inside A0 metrics.json", flush=True)
        elif mode == "decision_a0":
            overrides = spec["overrides"] + [f"project.name={base_cfg['project']['name']}_{name}"]
            cfg = apply_overrides(base_cfg, overrides)
            print(f"[Ablation] Running decision-only variant {name} on A0 predictions...", flush=True)
            execute_decision_stage(cfg, pred_dir=a0_run_dir / "preds")
        elif mode == "train":
            overrides = spec["overrides"] + [
                f"project.name={base_cfg['project']['name']}_{name}",
                f"train.epochs={args.epochs}",
                f"train.patience={args.patience}",
            ]
            cfg = apply_overrides(base_cfg, overrides)
            v_run_dir = get_profile_dirs(cfg)["run_dir"]
            if (v_run_dir / "metrics.json").exists():
                print(f"[Ablation] Skipping {name}, metrics already exist at {v_run_dir / 'metrics.json'}", flush=True)
                continue
            print(f"\n[Ablation] Training {name}: {spec['label']} (epochs={args.epochs})", flush=True)
            trainer = Trainer(cfg, dataset)
            trainer.fit(resume=(v_run_dir / "last.pt").exists())
            pred_dir = trainer.export_predictions()
            execute_decision_stage(cfg, pred_dir=pred_dir)

    # Collect all available variants into summary tables
    summary_rows: List[Dict[str, Any]] = []
    if a0_metrics_path.exists():
        with a0_metrics_path.open("r", encoding="utf-8") as f:
            a0_m = json.load(f)
        summary_rows.append(_extract_row("A0_full_cctb", ABLATION_SPECS["A0_full_cctb"]["label"], a0_m))
        summary_rows.append(
            _extract_row("A8_wo_aci_jev", ABLATION_SPECS["A8_wo_aci_jev"]["label"], a0_m, use_argmax=True)
        )

    for name, spec in ABLATION_SPECS.items():
        if spec["mode"] in ("reuse_a0", "argmax_a0"):
            continue
        cfg = apply_overrides(base_cfg, spec["overrides"] + [f"project.name={base_cfg['project']['name']}_{name}"])
        m_path = get_profile_dirs(cfg)["run_dir"] / "metrics.json"
        if m_path.exists():
            with m_path.open("r", encoding="utf-8") as f:
                m_data = json.load(f)
            summary_rows.append(_extract_row(name, spec["label"], m_data))

    out_tables_dir = resolve_path(base_cfg["paths"]["output_root"]) / "summary_tables"
    _save_summary_tables(summary_rows, out_tables_dir)
    print(f"[Ablation] Summary tables saved to {out_tables_dir}", flush=True)


if __name__ == "__main__":
    main()
