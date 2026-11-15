"""Training engine for CCTB-Net with AMP, checkpointing, temperature scaling, and prediction export."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from data_pipeline.dataset import CCTBDataset
from engine.config import get_profile_dirs
from models.cctb_net import CCTBJointLoss, CCTBNet

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


class Trainer:
    """Manages CCTB-Net training, validation, temperature calibration, and split inference."""

    def __init__(self, cfg: Dict[str, Any], dataset: CCTBDataset) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.dirs = get_profile_dirs(cfg)
        self.run_dir = self.dirs["run_dir"]
        self.run_dir.mkdir(parents=True, exist_ok=True)

        tcfg = cfg["train"]
        req_dev = str(tcfg.get("device", "cuda"))
        if req_dev.startswith("cuda") and not torch.cuda.is_available():
            req_dev = "cpu"
        self.device = torch.device(req_dev)
        self.use_amp = bool(tcfg.get("amp", True)) and self.device.type == "cuda"

        self.batch_size = int(tcfg.get("batch_size", 512))
        self.eval_batch_size = int(tcfg.get("eval_batch_size", 1024))
        self.epochs = int(tcfg.get("epochs", 20))
        self.grad_clip = float(tcfg.get("grad_clip", 1.0))
        self.patience = int(tcfg.get("patience", 5))
        self.seed = int(cfg["project"].get("seed", 42))

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.model = CCTBNet(cfg, dataset.meta).to(self.device)
        self.criterion = CCTBJointLoss(cfg, dataset.meta["train_class_counts"]).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(tcfg.get("lr", 1e-3)),
            weight_decay=float(tcfg.get("weight_decay", 1e-4)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, self.epochs)
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.csv_log_path = self.run_dir / "train_log.csv"
        self.tb_writer = SummaryWriter(log_dir=str(self.run_dir / "tb")) if SummaryWriter else None

    def _move_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.device.type == "cuda":
            return {
                k: v.pin_memory().to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "loss_cls": 0.0, "loss_bin": 0.0, "loss_sad": 0.0, "loss_cons": 0.0}
        n_batches = 0

        for b_idx, batch in enumerate(
            self.dataset.iter_batches("train", batch_size=self.batch_size, training=True, seed=self.seed + epoch),
            start=1,
        ):
            dev_batch = self._move_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(dev_batch)
                loss_dict = self.criterion(outputs, dev_batch["y_cls"], dev_batch["y_bin"])
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for k in totals:
                totals[k] += float(loss_dict[k].detach().cpu().item())
            n_batches += 1
            if b_idx % 100 == 0:
                print(f"    epoch {epoch} batch {b_idx} | loss={totals['loss']/n_batches:.4f}", flush=True)

        self.scheduler.step()
        denom = max(1, n_batches)
        return {k: v / denom for k, v in totals.items()}

    @torch.no_grad()
    def predict_split(self, split_name: str, max_eval_samples: int | None = None) -> Dict[str, np.ndarray]:
        self.model.eval()
        logits_l, s_anom_l, p_anom_l = [], [], []
        y_cls_l, y_bin_l, load_l, ts_l, fid_l = [], [], [], [], []

        for b_idx, batch in enumerate(
            self.dataset.iter_batches(
                split_name,
                batch_size=self.eval_batch_size,
                training=False,
                max_eval_samples=max_eval_samples,
            ),
            start=1,
        ):
            dev_batch = self._move_batch(batch)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                out = self.model(dev_batch)

            logits_l.append(out["logits"].float().cpu().numpy())
            s_anom_l.append(out["s_anom"].float().cpu().numpy())
            p_anom_l.append(out["p_anom"].float().cpu().numpy())
            y_cls_l.append(batch["y_cls"].numpy())
            y_bin_l.append(batch["y_bin"].numpy())
            load_l.append(batch["ctx_load"].numpy())
            ts_l.append(batch["ts_end"].numpy())
            fid_l.append(batch["flow_id"].numpy())
            if max_eval_samples is None or max_eval_samples > 300000:
                if b_idx % 250 == 0:
                    print(f"    infer {split_name} batch {b_idx}", flush=True)

        if not logits_l:
            return {
                "logits": np.zeros((0, 10), dtype=np.float32),
                "s_anom": np.zeros(0, dtype=np.float32),
                "p_anom": np.zeros(0, dtype=np.float32),
                "y_cls": np.zeros(0, dtype=np.int64),
                "y_bin": np.zeros(0, dtype=np.float32),
                "ctx_load": np.zeros(0, dtype=np.float32),
                "ts_end": np.zeros(0, dtype=np.float64),
                "flow_id": np.zeros(0, dtype=np.int64),
            }

        return {
            "logits": np.concatenate(logits_l, axis=0),
            "s_anom": np.concatenate(s_anom_l, axis=0),
            "p_anom": np.concatenate(p_anom_l, axis=0),
            "y_cls": np.concatenate(y_cls_l, axis=0),
            "y_bin": np.concatenate(y_bin_l, axis=0),
            "ctx_load": np.concatenate(load_l, axis=0),
            "ts_end": np.concatenate(ts_l, axis=0),
            "flow_id": np.concatenate(fid_l, axis=0),
        }

    def evaluate_validation(self) -> Dict[str, float]:
        preds = self.predict_split("val", max_eval_samples=200000)
        y_cls = preds["y_cls"]
        y_bin = preds["y_bin"]
        if len(y_cls) == 0:
            return {"val_macro_f1": 0.0, "val_auroc": 0.5}

        y_hat = np.argmax(preds["logits"], axis=1)
        present = np.unique(y_cls)
        macro_f1 = float(f1_score(y_cls, y_hat, labels=present, average="macro", zero_division=0))

        probs = torch.softmax(torch.from_numpy(preds["logits"]), dim=-1).numpy()
        p_atk = 1.0 - probs[:, 0]
        if len(np.unique(y_bin)) > 1:
            auroc = float(roc_auc_score(y_bin, p_atk))
        else:
            auroc = 0.5
        return {"val_macro_f1": macro_f1, "val_auroc": auroc}

    def save_checkpoint(self, name: str, epoch: int, best_metric: float) -> Path:
        ckpt_path = self.run_dir / name
        torch.save(
            {
                "epoch": epoch,
                "best_metric": best_metric,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "cfg": self.cfg,
            },
            ckpt_path,
        )
        return ckpt_path

    def load_checkpoint(self, name: str = "best.pt") -> None:
        ckpt_path = self.run_dir / name
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])

    def fit_temperature(self, val_logits: np.ndarray, val_y_cls: np.ndarray) -> float:
        """Fit scalar temperature T > 0 on Validation logits via NLL minimization."""
        if len(val_y_cls) == 0:
            return 1.0
        logits_t = torch.from_numpy(val_logits).float().to(self.device)
        y_t = torch.from_numpy(val_y_cls).long().to(self.device)
        log_temp = torch.zeros(1, device=self.device, requires_grad=True)
        opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)

        def closure() -> torch.Tensor:
            opt.zero_grad()
            temp = torch.exp(log_temp).clamp(min=0.05, max=10.0)
            loss = F.cross_entropy(logits_t / temp, y_t)
            loss.backward()
            return loss

        opt.step(closure)
        return float(torch.exp(log_temp).clamp(min=0.05, max=10.0).detach().cpu().item())

    def fit(self, resume: bool = False) -> Dict[str, Any]:
        start_epoch = 1
        best_f1 = -1.0
        bad_epochs = 0
        history = []

        last_ckpt = self.run_dir / "last.pt"
        if resume and last_ckpt.exists():
            ckpt = torch.load(last_ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_f1 = float(ckpt.get("best_metric", -1.0))
            if "scheduler_state" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])
            else:
                for _ in range(start_epoch - 1):
                    self.scheduler.step()
            print(f"  [Resume] Resuming from epoch {start_epoch - 1} (best_val_macro_f1={best_f1:.4f})", flush=True)

        file_mode = "a" if (resume and start_epoch > 1 and self.csv_log_path.exists()) else "w"
        with self.csv_log_path.open(file_mode, newline="", encoding="utf-8") as csv_f:
            writer = csv.writer(csv_f)
            if file_mode == "w":
                writer.writerow(
                    ["epoch", "loss", "loss_cls", "loss_bin", "loss_sad", "loss_cons", "val_macro_f1", "val_auroc", "sec"]
                )

            for epoch in range(start_epoch, self.epochs + 1):
                t0 = time.time()
                train_metrics = self.train_one_epoch(epoch)
                val_metrics = self.evaluate_validation()
                elapsed = time.time() - t0

                row = {
                    "epoch": epoch,
                    **train_metrics,
                    **val_metrics,
                    "sec": round(elapsed, 2),
                }
                history.append(row)
                writer.writerow(
                    [
                        epoch,
                        f"{row['loss']:.5f}",
                        f"{row['loss_cls']:.5f}",
                        f"{row['loss_bin']:.5f}",
                        f"{row['loss_sad']:.5f}",
                        f"{row['loss_cons']:.5f}",
                        f"{row['val_macro_f1']:.5f}",
                        f"{row['val_auroc']:.5f}",
                        f"{elapsed:.2f}",
                    ]
                )
                csv_f.flush()
                print(
                    f"  [Epoch {epoch:02d}/{self.epochs}] loss={row['loss']:.4f} "
                    f"(cls={row['loss_cls']:.4f}, bin={row['loss_bin']:.4f}, sad={row['loss_sad']:.4f}) "
                    f"| val_macro_f1={row['val_macro_f1']:.4f} val_auroc={row['val_auroc']:.4f} ({elapsed:.1f}s)",
                    flush=True,
                )

                if self.tb_writer:
                    for k, v in train_metrics.items():
                        self.tb_writer.add_scalar(f"train/{k}", v, epoch)
                    for k, v in val_metrics.items():
                        self.tb_writer.add_scalar(f"val/{k}", v, epoch)

                if val_metrics["val_macro_f1"] >= best_f1:
                    best_f1 = val_metrics["val_macro_f1"]
                    self.save_checkpoint("best.pt", epoch, best_f1)
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                self.save_checkpoint("last.pt", epoch, best_f1)
                if bad_epochs >= self.patience:
                    break

        if self.tb_writer:
            self.tb_writer.close()

        self.load_checkpoint("best.pt")
        return {"best_val_macro_f1": best_f1, "history": history}

    def export_predictions(self) -> Path:
        """Export calibrated probabilities and anomaly scores for val, cal, and test splits."""
        pred_dir = self.run_dir / "preds"
        pred_dir.mkdir(parents=True, exist_ok=True)

        val_preds = self.predict_split("val")
        temperature = self.fit_temperature(val_preds["logits"], val_preds["y_cls"])

        for split_name in ["val", "cal", "test"]:
            print(f"  [Export] Inferring {split_name} split...", flush=True)
            raw = val_preds if split_name == "val" else self.predict_split(split_name)
            scaled_logits = raw["logits"] / max(1e-3, temperature)
            probs = torch.softmax(torch.from_numpy(scaled_logits), dim=-1).numpy().astype(np.float32)
            np.savez_compressed(
                pred_dir / f"{split_name}_preds.npz",
                logits=raw["logits"],
                probs=probs,
                s_anom=raw["s_anom"],
                p_anom=raw["p_anom"],
                y_cls=raw["y_cls"],
                y_bin=raw["y_bin"],
                ctx_load=raw["ctx_load"],
                ts_end=raw["ts_end"],
                flow_id=raw["flow_id"],
                temperature=np.array([temperature], dtype=np.float32),
            )

        with (pred_dir / "calibration_meta.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "temperature": temperature,
                    "high_load_threshold": self.dataset.meta["high_load_threshold"],
                },
                f,
                indent=2,
            )
        return pred_dir
