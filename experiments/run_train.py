"""End-to-end training, prediction export, and ACI+Jev evaluation entry point."""
from __future__ import annotations

import argparse
import json
import time

from data_pipeline.dataset import CCTBDataset, build_processed_dataset
from engine.config import get_profile_dirs, load_config
from engine.trainer import Trainer
from experiments.run_decision import execute_decision_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CCTB-Net and evaluate ACI+Jev pipeline.")
    parser.add_argument("--config", type=str, default="configs/fast_dev.yaml", help="Path to YAML config.")
    parser.add_argument("--force-preprocess", action="store_true", help="Rebuild preprocessed dataset.")
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt if present.")
    parser.add_argument("--skip-decision", action="store_true", help="Skip Phase 3 ACI+Jev evaluation.")
    parser.add_argument("overrides", nargs="*", help="Optional dotted key=value overrides.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    t0 = time.time()

    proc_dir = build_processed_dataset(cfg, force=args.force_preprocess)
    dataset = CCTBDataset(proc_dir, cfg)
    trainer = Trainer(cfg, dataset)

    print(
        f"[Train] Starting {cfg['project']['name']} on device={trainer.device} "
        f"(amp={trainer.use_amp}, epochs={trainer.epochs})"
    )
    fit_summary = trainer.fit(resume=args.resume)
    pred_dir = trainer.export_predictions()
    print(
        f"[Train] Finished in {time.time() - t0:.2f}s | "
        f"best_val_macro_f1={fit_summary['best_val_macro_f1']:.4f} | preds -> {pred_dir}"
    )

    if not args.skip_decision:
        metrics = execute_decision_stage(cfg, pred_dir=pred_dir)
        jev = metrics["cctb_aci_jev"]
        conf = metrics["conformal"]
        dirs = get_profile_dirs(cfg)
        print(f"[Eval] Final Test Metrics ({dirs['run_dir'] / 'metrics.json'}):")
        print(
            f"  Macro-F1          : {jev['macro_f1']:.4f}\n"
            f"  AUROC (prob/anom) : {jev['auroc_prob']:.4f} / {jev['auroc_anom']:.4f}\n"
            f"  FPR / FNR (select): {jev['fpr_selective']:.4f} / {jev['fnr_selective']:.4f}\n"
            f"  High-Load Norm FPR: {jev['high_load_normal_fpr']:.4f}\n"
            f"  Empirical Coverage: {conf['empirical_coverage']:.4f}\n"
            f"  Avg Pred Set Size : {conf['avg_set_size']:.3f}"
        )


if __name__ == "__main__":
    main()
