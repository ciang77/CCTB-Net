"""CLI entry point for Phase 3 ACI conformal calibration and Jev secondary adjudication."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from decision.aci import run_aci_stream, run_static_cp
from decision.secondary_judge import (
    BayesianJudgeA,
    EvidenceBatch,
    StructuredRuleJudge,
    run_decision_pipeline,
)
from engine.config import get_profile_dirs, load_config
from evaluation.metrics import compute_all_metrics


def execute_decision_stage(cfg: Dict[str, Any], pred_dir: Path | None = None) -> Dict[str, Any]:
    dirs = get_profile_dirs(cfg)
    run_dir = dirs["run_dir"]
    pred_dir = pred_dir or (run_dir / "preds")

    val_npz = np.load(pred_dir / "val_preds.npz")
    cal_npz = np.load(pred_dir / "cal_preds.npz")
    test_npz = np.load(pred_dir / "test_preds.npz")
    with (pred_dir / "calibration_meta.json").open("r", encoding="utf-8") as f:
        cal_meta = json.load(f)

    dcfg = cfg["decision"]
    alpha = float(dcfg.get("alpha", 0.05))
    eta = float(dcfg.get("eta", 0.005))
    aci_window = int(dcfg.get("aci_window", 20000))
    aci_bins = int(dcfg.get("aci_num_bins", 4096))
    label_delay = int(dcfg.get("label_delay", 0))
    score_type = str(dcfg.get("score_type", "aps"))
    seed = int(cfg["project"].get("seed", 42))

    # 1. Construct prediction sets on Val (static CP from Cal) to fit SecondaryJudge on Val uncertain samples
    val_cp = run_static_cp(
        cal_probs=cal_npz["probs"],
        cal_y_cls=cal_npz["y_cls"],
        test_probs=val_npz["probs"],
        alpha=alpha,
        score_type=score_type,
        seed=seed,
    )
    val_ev = EvidenceBatch(
        probs=val_npz["probs"],
        pred_sets=val_cp["pred_sets"],
        s_anom=val_npz["s_anom"],
        p_anom=val_npz["p_anom"],
        ctx_load=val_npz["ctx_load"],
    )

    judge_type = str(dcfg.get("judge_type", "jev_b")).lower()
    judge = BayesianJudgeA(cfg) if judge_type == "jev_a" else StructuredRuleJudge(cfg)
    judge.fit(val_ev, val_npz["y_cls"])

    # 2. Run online ACI stream on Test (sorted by ts_end)
    test_aci = run_aci_stream(
        cal_probs=cal_npz["probs"],
        cal_y_cls=cal_npz["y_cls"],
        test_probs=test_npz["probs"],
        test_y_cls=test_npz["y_cls"],
        alpha=alpha,
        eta=eta,
        window_size=aci_window,
        num_bins=aci_bins,
        label_delay=label_delay,
        score_type=score_type,
        seed=seed,
    )
    test_ev = EvidenceBatch(
        probs=test_npz["probs"],
        pred_sets=test_aci["pred_sets"],
        s_anom=test_npz["s_anom"],
        p_anom=test_npz["p_anom"],
        ctx_load=test_npz["ctx_load"],
    )
    decision_out = run_decision_pipeline(test_ev, judge)

    metrics = compute_all_metrics(
        y_cls=test_npz["y_cls"],
        probs=test_npz["probs"],
        s_anom=test_npz["s_anom"],
        ctx_load=test_npz["ctx_load"],
        high_load_threshold=float(cal_meta["high_load_threshold"]),
        pred_sets=test_aci["pred_sets"],
        decision_out=decision_out,
    )
    metrics["temperature"] = float(cal_meta["temperature"])
    metrics["alpha_target"] = alpha
    metrics["alpha_final"] = float(test_aci["alpha_traj"][-1]) if len(test_aci["alpha_traj"]) > 0 else alpha

    out_path = run_dir / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ACI + Jev secondary decision pipeline.")
    parser.add_argument("--config", type=str, default="configs/fast_dev.yaml", help="Path to YAML config.")
    parser.add_argument("overrides", nargs="*", help="Optional dotted key=value overrides.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    t0 = time.time()
    metrics = execute_decision_stage(cfg)
    elapsed = time.time() - t0

    jev = metrics["cctb_aci_jev"]
    conf = metrics["conformal"]
    print(f"[Decision] Completed in {elapsed:.2f}s")
    print(
        f"  Macro-F1={jev['macro_f1']:.4f} | AUROC={jev['auroc_prob']:.4f} | "
        f"FPR={jev['fpr_selective']:.4f} | FNR={jev['fnr_selective']:.4f} | "
        f"HighLoad-FPR={jev['high_load_normal_fpr']:.4f}"
    )
    print(
        f"  Empirical Coverage={conf['empirical_coverage']:.4f} | "
        f"Avg Set Size={conf['avg_set_size']:.3f} | Uncertain Rate={metrics['uncertain_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
