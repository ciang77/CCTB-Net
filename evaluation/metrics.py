"""Comprehensive detection, conformal, and context-shift evaluation metrics for CCTB-Net."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score

from data_pipeline.schema import CLASS_NAMES, NUM_CLASSES
from decision.secondary_judge import ACTION_DEFER, DecisionOutput


def _binary_fpr_fnr(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Tuple[float, float]:
    is_norm = y_true_bin == 0
    is_atk = y_true_bin == 1
    fpr = float((y_pred_bin[is_norm] == 1).mean()) if is_norm.any() else 0.0
    fnr = float((y_pred_bin[is_atk] == 0).mean()) if is_atk.any() else 0.0
    return fpr, fnr


def compute_conformal_metrics(
    pred_sets: np.ndarray,
    y_cls: np.ndarray,
    rolling_window: int = 2000,
) -> Dict[str, Any]:
    """Compute Empirical Coverage, Rolling Coverage, Average Set Size, and Class-Conditional Coverage."""
    n = len(y_cls)
    if n == 0:
        return {
            "empirical_coverage": 0.0,
            "avg_set_size": 0.0,
            "singleton_rate": 0.0,
            "empty_rate": 0.0,
            "ambiguous_rate": 0.0,
            "rolling_coverage_mean": 0.0,
            "rolling_coverage_min": 0.0,
            "per_class_coverage": {},
        }

    covered = pred_sets[np.arange(n), y_cls].astype(np.float64)
    set_sizes = pred_sets.sum(axis=1)

    win = min(n, max(1, int(rolling_window)))
    kernel = np.ones(win, dtype=np.float64) / float(win)
    roll_cov = np.convolve(covered, kernel, mode="valid")

    per_cls_cov: Dict[str, float] = {}
    for c in range(NUM_CLASSES):
        m = y_cls == c
        if m.any():
            per_cls_cov[CLASS_NAMES[c]] = float(covered[m].mean())

    return {
        "empirical_coverage": float(covered.mean()),
        "avg_set_size": float(set_sizes.mean()),
        "singleton_rate": float((set_sizes == 1).mean()),
        "empty_rate": float((set_sizes == 0).mean()),
        "ambiguous_rate": float((set_sizes >= 2).mean()),
        "rolling_coverage_mean": float(roll_cov.mean()),
        "rolling_coverage_min": float(roll_cov.min()),
        "per_class_coverage": per_cls_cov,
    }


def compute_detection_metrics(
    y_cls: np.ndarray,
    pred_cls: np.ndarray,
    probs: np.ndarray,
    s_anom: np.ndarray,
    ctx_load: np.ndarray,
    high_load_threshold: float,
    actions: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compute Macro-F1, AUROC, FPR, FNR, and High-Load Normal FPR."""
    n = len(y_cls)
    if n == 0:
        return {
            "macro_f1": 0.0,
            "binary_f1": 0.0,
            "auroc_prob": 0.5,
            "auroc_anom": 0.5,
            "auprc_attack": 0.0,
            "auprc_normal": 0.0,
            "fpr": 0.0,
            "fnr": 0.0,
            "fpr_selective": 0.0,
            "fnr_selective": 0.0,
            "defer_rate": 0.0,
            "high_load_normal_fpr": 0.0,
            "per_class_f1": {},
        }

    y_bin = (y_cls != 0).astype(np.int32)
    # Conservative binary view: UNKNOWN_ATTACK (10) or DEFER counts as attack/alert
    pred_bin_cons = ((pred_cls != 0) | ((actions == ACTION_DEFER) if actions is not None else False)).astype(
        np.int32
    )
    fpr_cons, fnr_cons = _binary_fpr_fnr(y_bin, pred_bin_cons)

    # Selective view: evaluate on non-deferred samples
    if actions is not None:
        keep = actions != ACTION_DEFER
        defer_rate = float((~keep).mean())
    else:
        keep = np.ones(n, dtype=np.bool_)
        defer_rate = 0.0

    if keep.any():
        fpr_sel, fnr_sel = _binary_fpr_fnr(y_bin[keep], (pred_cls[keep] != 0).astype(np.int32))
        present = np.unique(y_cls[keep])
        macro_f1 = float(
            f1_score(y_cls[keep], pred_cls[keep], labels=present, average="macro", zero_division=0)
        )
        bin_f1 = float(f1_score(y_bin[keep], (pred_cls[keep] != 0).astype(np.int32), zero_division=0))
    else:
        fpr_sel, fnr_sel, macro_f1, bin_f1 = fpr_cons, fnr_cons, 0.0, 0.0

    # Core paper metric: FPR on normal traffic during high host context load periods
    high_load_norm = (y_cls == 0) & (ctx_load >= high_load_threshold) & keep
    if high_load_norm.any():
        high_load_fpr = float((pred_cls[high_load_norm] != 0).mean())
    else:
        norm_keep = (y_cls == 0) & keep
        if norm_keep.any():
            top_q = np.quantile(ctx_load[norm_keep], 0.90)
            sub = norm_keep & (ctx_load >= top_q)
            high_load_fpr = float((pred_cls[sub] != 0).mean()) if sub.any() else fpr_sel
        else:
            high_load_fpr = 0.0

    p_atk = 1.0 - probs[:, 0]
    if len(np.unique(y_bin)) > 1:
        auroc_prob = float(roc_auc_score(y_bin, p_atk))
        auroc_anom = float(roc_auc_score(y_bin, s_anom))
        auprc_atk = float(average_precision_score(y_bin, p_atk))
        auprc_norm = float(average_precision_score(1 - y_bin, probs[:, 0]))
    else:
        auroc_prob, auroc_anom, auprc_atk, auprc_norm = 0.5, 0.5, 0.0, 0.0

    _, _, f1_arr, _ = precision_recall_fscore_support(
        y_cls[keep] if keep.any() else y_cls,
        pred_cls[keep] if keep.any() else pred_cls,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    per_cls_f1 = {
        CLASS_NAMES[i]: float(f1_arr[i]) for i in range(NUM_CLASSES) if (y_cls == i).any()
    }

    return {
        "macro_f1": macro_f1,
        "binary_f1": bin_f1,
        "auroc_prob": auroc_prob,
        "auroc_anom": auroc_anom,
        "auprc_attack": auprc_atk,
        "auprc_normal": auprc_norm,
        "fpr": fpr_cons,
        "fnr": fnr_cons,
        "fpr_selective": fpr_sel,
        "fnr_selective": fnr_sel,
        "defer_rate": defer_rate,
        "high_load_normal_fpr": high_load_fpr,
        "per_class_f1": per_cls_f1,
    }


def compute_all_metrics(
    y_cls: np.ndarray,
    probs: np.ndarray,
    s_anom: np.ndarray,
    ctx_load: np.ndarray,
    high_load_threshold: float,
    pred_sets: np.ndarray,
    decision_out: DecisionOutput,
) -> Dict[str, Any]:
    """Compute baseline argmax metrics, conformal metrics, and post-Jev decision metrics."""
    argmax_cls = np.argmax(probs, axis=1)
    base_det = compute_detection_metrics(
        y_cls=y_cls,
        pred_cls=argmax_cls,
        probs=probs,
        s_anom=s_anom,
        ctx_load=ctx_load,
        high_load_threshold=high_load_threshold,
        actions=None,
    )
    jev_det = compute_detection_metrics(
        y_cls=y_cls,
        pred_cls=decision_out.final_label,
        probs=probs,
        s_anom=s_anom,
        ctx_load=ctx_load,
        high_load_threshold=high_load_threshold,
        actions=decision_out.action,
    )
    conf_metrics = compute_conformal_metrics(pred_sets, y_cls)

    unc_mask = decision_out.is_uncertain
    if unc_mask.any():
        jev_unc_acc = float((decision_out.final_label[unc_mask] == y_cls[unc_mask]).mean())
        base_unc_acc = float((argmax_cls[unc_mask] == y_cls[unc_mask]).mean())
    else:
        jev_unc_acc, base_unc_acc = 1.0, 1.0

    return {
        "cctb_aci_jev": jev_det,
        "argmax_baseline": base_det,
        "conformal": conf_metrics,
        "uncertain_rate": float(unc_mask.mean()) if len(y_cls) > 0 else 0.0,
        "uncertain_subset_acc_before_jev": base_unc_acc,
        "uncertain_subset_acc_after_jev": jev_unc_acc,
    }
