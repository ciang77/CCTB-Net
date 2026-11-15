"""CLI entry point for Phase 1 preprocessing and multi-source alignment."""
from __future__ import annotations

import argparse
import json
import time

from data_pipeline.dataset import build_processed_dataset
from engine.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CCTB-Net Phase 1 preprocessing pipeline.")
    parser.add_argument("--config", type=str, default="configs/fast_dev.yaml", help="Path to YAML config.")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if cached arrays exist.")
    parser.add_argument("overrides", nargs="*", help="Optional dotted key=value overrides.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    t0 = time.time()
    proc_dir = build_processed_dataset(cfg, force=args.force)
    elapsed = time.time() - t0

    with (proc_dir / "metadata.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"[Preprocess] Completed in {elapsed:.2f}s -> {proc_dir}")
    print(f"  Total flows: {meta['n_flows']} | num_features={meta['num_features']} | cat_features={meta['cat_features']} | ctx_features={meta['ctx_features']}")
    for split_name, counts in meta["split_counts"].items():
        print(f"  Split [{split_name:5s}]: total={counts['_total']}")


if __name__ == "__main__":
    main()
