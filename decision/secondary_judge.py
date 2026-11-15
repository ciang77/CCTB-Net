"""Jev-style secondary decision interface, routing logic, and structured adjudicators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from data_pipeline.schema import NUM_CLASSES, UNKNOWN_ATTACK_ID

ACTION_ACCEPT = 0
ACTION_ALERT = 1
ACTION_DEFER = 2


@dataclass
class EvidenceBatch:
    probs: np.ndarray      # [N, K] calibrated class probabilities
    pred_sets: np.ndarray  # [N, K] bool conformal prediction sets C(x_t)
    s_anom: np.ndarray     # [N] continuous anomaly deviation score
    p_anom: np.ndarray     # [N] anomaly branch probability
    ctx_load: np.ndarray   # [N] host context load summary


@dataclass
class DecisionOutput:
    final_label: np.ndarray   # [N] int64 in {0..K-1} U {UNKNOWN_ATTACK_ID=10}
    action: np.ndarray        # [N] int8 in {ACCEPT=0, ALERT=1, DEFER=2}
    is_uncertain: np.ndarray  # [N] bool (True where |C(x_t)| != 1)
    set_sizes: np.ndarray     # [N] int32 |C(x_t)|
    posterior_max: np.ndarray # [N] float32 confidence of final decision
    z_anom: np.ndarray        # [N] float32 context-stratified standardized anomaly score


class SecondaryJudge(ABC):
    """Unified pluggable interface for Jev-style secondary adjudication on uncertain samples."""

    @abstractmethod
    def fit(self, val_ev: EvidenceBatch, val_y_cls: np.ndarray) -> "SecondaryJudge":
        """Fit secondary judge parameters strictly on Validation split."""

    @abstractmethod
    def adjudicate_uncertain(
        self,
        ev: EvidenceBatch,
        uncertain_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (labels, actions, max_posteriors, z_anom_all) for uncertain samples."""


class ContextAnomalyStandardizer:
    """Computes z_anom = (log(1+s_anom) - mu_norm(ctx_stratum)) / sigma_norm(ctx_stratum)."""

    def __init__(self, num_strata: int = 5) -> None:
        self.num_strata = max(1, int(num_strata))
        self.quantiles: np.ndarray = np.array([], dtype=np.float64)
        self.mu: np.ndarray = np.zeros(1, dtype=np.float64)
        self.sigma: np.ndarray = np.ones(1, dtype=np.float64)

    def fit(self, ctx_load: np.ndarray, s_anom: np.ndarray, y_cls: np.ndarray) -> "ContextAnomalyStandardizer":
        log_s = np.log1p(np.maximum(0.0, s_anom.astype(np.float64)))
        if self.num_strata > 1 and len(ctx_load) > self.num_strata:
            qs = np.linspace(0.0, 1.0, self.num_strata + 1)[1:-1]
            self.quantiles = np.quantile(ctx_load, qs)
        else:
            self.quantiles = np.array([], dtype=np.float64)

        strata = np.digitize(ctx_load, self.quantiles)
        n_bins = len(self.quantiles) + 1
        norm_mask = y_cls == 0
        global_mu = float(log_s[norm_mask].mean()) if norm_mask.any() else float(log_s.mean())
        global_sd = float(log_s[norm_mask].std()) if norm_mask.sum() > 1 else float(log_s.std() + 1.0)
        global_sd = max(global_sd, 1e-3)

        self.mu = np.full(n_bins, global_mu, dtype=np.float64)
        self.sigma = np.full(n_bins, global_sd, dtype=np.float64)
        for b in range(n_bins):
            m = (strata == b) & norm_mask
            if m.sum() >= 5:
                self.mu[b] = float(log_s[m].mean())
                self.sigma[b] = max(float(log_s[m].std()), 1e-3)
        return self

    def transform(self, ctx_load: np.ndarray, s_anom: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        log_s = np.log1p(np.maximum(0.0, s_anom.astype(np.float64)))
        strata = np.digitize(ctx_load, self.quantiles)
        z = (log_s - self.mu[strata]) / self.sigma[strata]
        return z.astype(np.float32), strata.astype(np.int32)


class StructuredRuleJudge(SecondaryJudge):
    """Scheme B (Main): fuses calibrated P_cls, context-relative z_anom, ctx prior, and rules R1-R4."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        dcfg = cfg["decision"]
        self.num_strata = int(dcfg.get("num_load_strata", 5))
        self.z_low = float(dcfg.get("z_low", -0.5))
        self.z_high = float(dcfg.get("z_high", 2.0))
        self.cost_fn = float(dcfg.get("cost_fn", 5.0))
        self.cost_fp = float(dcfg.get("cost_fp", 1.0))
        self.cost_miscls = float(dcfg.get("cost_miscls", 1.5))
        self.defer_budget = float(dcfg.get("defer_budget", 0.05))
        self.unk_conf_thresh = float(dcfg.get("unk_conf_thresh", 0.80))
        self.max_defer_thresh = float(dcfg.get("max_defer_thresh", 0.55))

        self.standardizer = ContextAnomalyStandardizer(self.num_strata)
        self.stratum_log_priors: np.ndarray = np.full((self.num_strata, NUM_CLASSES), -np.log(NUM_CLASSES))
        # Weights [w_prob, w_anom, w_prior, w_rule], learned on Val uncertain samples
        self.weights = np.array([1.0, 0.35, 0.10, 0.80], dtype=np.float64)
        self.defer_threshold: float = 0.0

    def _fit_weights_on_val(
        self,
        ev: EvidenceBatch,
        z_val: np.ndarray,
        strata_val: np.ndarray,
        unc_mask: np.ndarray,
        y_cls: np.ndarray,
    ) -> None:
        """Fit [w_prob, w_anom, w_prior, w_rule] via L2-regularized NLL on Val uncertain samples."""
        if unc_mask.sum() < 50:
            return
        idx = np.flatnonzero(unc_mask)
        if len(idx) > 50000:
            sel = np.linspace(0, len(idx) - 1, num=50000, dtype=np.int64)
            idx = idx[sel]

        sub_mask = np.zeros(len(y_cls), dtype=np.bool_)
        sub_mask[idx] = True
        probs = ev.probs[sub_mask].astype(np.float64)
        c_sets = ev.pred_sets[sub_mask]
        z = np.clip(z_val[sub_mask].astype(np.float64), -5.0, 5.0)
        st = np.clip(strata_val[sub_mask], 0, len(self.stratum_log_priors) - 1)
        y = y_cls[sub_mask]
        is_empty = c_sets.sum(axis=1) == 0
        cand_mask = np.where(is_empty[:, None], True, c_sets)

        f0 = np.log(np.clip(probs, 1e-6, 1.0))
        f1 = np.broadcast_to(z[:, None], probs.shape).copy()
        f1[:, 0] = -z
        f2 = self.stratum_log_priors[st]
        f3 = np.zeros_like(f0)
        f3[(z < self.z_low) & c_sets[:, 0], 0] = 1.0
        all_atk = (~is_empty) & (~c_sets[:, 0])
        if all_atk.any():
            top_atk = np.argmax(np.where(c_sets, probs, -1.0), axis=1)
            f3[all_atk, top_atk[all_atk]] += 0.5

        # Keep only samples whose true class is in candidate mask for clean likelihood fitting
        valid_y = cand_mask[np.arange(len(y)), y]
        if valid_y.sum() < 30:
            return
        f0, f1, f2, f3 = f0[valid_y], f1[valid_y], f2[valid_y], f3[valid_y]
        cm = cand_mask[valid_y]
        yv = y[valid_y]

        # Simple projected gradient ascent / finite-step Newton on 4 scalar weights
        w = np.array([1.0, 0.30, 0.10, 0.50], dtype=np.float64)
        lr = 0.25
        for _ in range(30):
            s = w[0] * f0 + w[1] * f1 + w[2] * f2 + w[3] * f3
            s = np.where(cm, s, -1e9)
            s -= s.max(axis=1, keepdims=True)
            p = np.exp(s)
            p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
            grad = np.array(
                [
                    np.mean(f0[np.arange(len(yv)), yv] - np.sum(p * f0, axis=1)),
                    np.mean(f1[np.arange(len(yv)), yv] - np.sum(p * f1, axis=1)),
                    np.mean(f2[np.arange(len(yv)), yv] - np.sum(p * f2, axis=1)),
                    np.mean(f3[np.arange(len(yv)), yv] - np.sum(p * f3, axis=1)),
                ]
            ) - 0.05 * w
            w = np.clip(w + lr * grad, [0.8, 0.0, 0.0, 0.0], [5.0, 2.0, 0.15, 2.0])
        self.weights = w

    def _calibrate_cost_for_low_fpr(
        self,
        val_ev: EvidenceBatch,
        z_val: np.ndarray,
        strata_val: np.ndarray,
        unc_mask: np.ndarray,
        val_y_cls: np.ndarray,
    ) -> None:
        """Grid-tune (cost_fp, z_low) on Val to keep FPR low without hurting Macro-F1 / FNR."""
        if unc_mask.sum() < 50:
            return
        idx = np.flatnonzero(unc_mask)
        if len(idx) > 80000:
            sel = np.linspace(0, len(idx) - 1, num=80000, dtype=np.int64)
            idx = idx[sel]
        sub_mask = np.zeros(len(val_y_cls), dtype=np.bool_)
        sub_mask[idx] = True

        y_sub = val_y_cls[sub_mask]
        is_norm = y_sub == 0
        is_atk = y_sub != 0
        if not is_norm.any() or not is_atk.any():
            return

        orig_z_low = self.z_low
        orig_cost_fp = self.cost_fp
        best_score = -1e9
        best_pair = (orig_cost_fp, orig_z_low)

        for z_l in [-0.5, 0.0, 0.5]:
            self.z_low = z_l
            post, _ = self._compute_candidate_posteriors(val_ev, z_val, strata_val, sub_mask)
            p_norm = post[:, 0]
            p_atk_total = 1.0 - p_norm
            atk_post = post[:, 1:]
            best_atk_off = np.argmax(atk_post, axis=1)
            best_atk_cls = 1 + best_atk_off
            best_atk_p = atk_post[np.arange(len(y_sub)), best_atk_off]

            for c_fp in [1.0, 2.0, 4.0, 6.0, 10.0]:
                cost_pred_norm = self.cost_fn * p_atk_total
                cost_pred_atk = c_fp * p_norm + self.cost_miscls * np.maximum(0.0, p_atk_total - best_atk_p)
                pred_cls = np.where(cost_pred_atk <= cost_pred_norm, best_atk_cls, 0)

                fpr = float((pred_cls[is_norm] != 0).mean())
                fnr = float((pred_cls[is_atk] == 0).mean())
                acc = float((pred_cls == y_sub).mean())
                # Objective: prioritize accuracy & low FNR while penalizing FPR > 0.02
                obj = acc - 3.0 * fnr - 2.0 * max(0.0, fpr - 0.015) - 0.2 * fpr
                if obj > best_score:
                    best_score = obj
                    best_pair = (c_fp, z_l)

        self.cost_fp, self.z_low = best_pair

    def fit(self, val_ev: EvidenceBatch, val_y_cls: np.ndarray) -> "StructuredRuleJudge":
        if len(val_y_cls) == 0:
            return self
        self.standardizer.fit(val_ev.ctx_load, val_ev.s_anom, val_y_cls)
        z_val, strata_val = self.standardizer.transform(val_ev.ctx_load, val_ev.s_anom)
        n_bins = len(self.standardizer.mu)
        self.stratum_log_priors = np.zeros((n_bins, NUM_CLASSES), dtype=np.float64)
        for b in range(n_bins):
            m = strata_val == b
            counts = np.bincount(val_y_cls[m], minlength=NUM_CLASSES).astype(np.float64) + 1.0
            self.stratum_log_priors[b] = np.log(counts / counts.sum())

        set_sizes = val_ev.pred_sets.sum(axis=1)
        unc_mask = set_sizes != 1
        if unc_mask.any():
            self._fit_weights_on_val(val_ev, z_val, strata_val, unc_mask, val_y_cls)
            self._calibrate_cost_for_low_fpr(val_ev, z_val, strata_val, unc_mask, val_y_cls)
            post_u, _ = self._compute_candidate_posteriors(val_ev, z_val, strata_val, unc_mask)
            max_post = post_u.max(axis=1)
            unc_rate = float(unc_mask.mean())
            if self.defer_budget <= 0.0 or unc_rate == 0.0:
                self.defer_threshold = 0.0
            else:
                rel_q = min(0.50, self.defer_budget / max(unc_rate, 1e-6))
                self.defer_threshold = min(self.max_defer_thresh, float(np.quantile(max_post, rel_q)))
        return self

    def _compute_candidate_posteriors(
        self,
        ev: EvidenceBatch,
        z_anom: np.ndarray,
        strata: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        probs = ev.probs[mask].astype(np.float64)
        c_sets = ev.pred_sets[mask]
        z = np.clip(z_anom[mask].astype(np.float64), -5.0, 5.0)
        st = np.clip(strata[mask], 0, len(self.stratum_log_priors) - 1)
        n_u = len(probs)

        # Candidate score matrix over K known classes + 1 UNKNOWN_ATTACK_ID (index 10)
        scores = np.full((n_u, NUM_CLASSES + 1), -1e9, dtype=np.float64)
        set_sz = c_sets.sum(axis=1)
        is_empty = set_sz == 0
        p_max = probs.max(axis=1)

        # Candidate mask: if |C| >= 2 restrict to C; if empty allow all K classes
        cand_mask = np.where(is_empty[:, None], True, c_sets)

        log_p = np.log(np.clip(probs, 1e-6, 1.0))
        log_prior = self.stratum_log_priors[st]

        # Directional anomaly deviation psi(normal) = -z_anom, psi(attack) = +z_anom
        psi = np.broadcast_to(z[:, None], (n_u, NUM_CLASSES)).copy()
        psi[:, 0] = -z

        # Rule bonus R1: low anomaly score under current context load & normal in candidate mask -> boost normal
        rule_bonus = np.zeros((n_u, NUM_CLASSES), dtype=np.float64)
        r1_mask = (z < self.z_low) & cand_mask[:, 0]
        rule_bonus[r1_mask, 0] = 1.0

        # Rule bonus R3: all candidates in C are attacks -> boost top attack candidate
        all_atk_cand = (~is_empty) & (~c_sets[:, 0])
        if all_atk_cand.any():
            masked_p = np.where(c_sets, probs, -1.0)
            top_atk = np.argmax(masked_p, axis=1)
            rule_bonus[all_atk_cand, top_atk[all_atk_cand]] += 0.5

        w0, w1, w2, w3 = self.weights
        known_scores = w0 * log_p + w1 * psi + w2 * log_prior + w3 * rule_bonus
        scores[:, :NUM_CLASSES] = np.where(cand_mask, known_scores, -1e9)

        # Rule R2: Only trigger UNKNOWN_ATTACK when |C|=0 AND classifier lacks high confidence AND z > z_high
        unk_active = is_empty & (p_max < self.unk_conf_thresh) & (z > self.z_high)
        scores[unk_active, UNKNOWN_ATTACK_ID] = w1 * z[unk_active] + w3 * 1.5

        # Softmax over active candidates
        max_s = np.max(scores, axis=1, keepdims=True)
        exp_s = np.exp(scores - max_s)
        post = exp_s / np.maximum(exp_s.sum(axis=1, keepdims=True), 1e-12)
        return post, is_empty

    def adjudicate_uncertain(
        self,
        ev: EvidenceBatch,
        uncertain_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z_all, strata_all = self.standardizer.transform(ev.ctx_load, ev.s_anom)
        n_u = int(uncertain_mask.sum())
        if n_u == 0:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int8),
                np.zeros(0, dtype=np.float32),
                z_all,
            )

        post, _ = self._compute_candidate_posteriors(ev, z_all, strata_all, uncertain_mask)
        # Rule R4: Cost-sensitive expected risk decision
        p_norm = post[:, 0]
        p_atk_total = 1.0 - p_norm
        # Expected cost of predicting normal = cost_fn * P(attack)
        cost_pred_norm = self.cost_fn * p_atk_total
        # Expected cost of predicting best attack class c* (in 1..K)
        atk_post = post[:, 1:]
        best_atk_offset = np.argmax(atk_post, axis=1)
        best_atk_cls = 1 + best_atk_offset
        best_atk_p = atk_post[np.arange(n_u), best_atk_offset]
        cost_pred_atk = self.cost_fp * p_norm + self.cost_miscls * np.maximum(0.0, p_atk_total - best_atk_p)

        choose_atk = cost_pred_atk <= cost_pred_norm
        labels = np.where(choose_atk, best_atk_cls, 0).astype(np.int64)
        max_post = post.max(axis=1).astype(np.float32)

        actions = np.where(labels == 0, ACTION_ACCEPT, ACTION_ALERT).astype(np.int8)
        if self.defer_threshold > 0.0:
            actions[max_post < self.defer_threshold] = ACTION_DEFER

        return labels, actions, max_post, z_all


class BayesianJudgeA(SecondaryJudge):
    """Scheme A: Discretized Naive Bayes / CPT graphical inference over uncertain samples."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.num_strata = int(cfg["decision"].get("num_load_strata", 5))
        self.defer_budget = float(cfg["decision"].get("defer_budget", 0.05))
        self.standardizer = ContextAnomalyStandardizer(self.num_strata)
        self.class_priors = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        # CPT P(anom_bin | y) with 3 bins: low(< -0.5), mid, high(> 1.5)
        self.cpt_anom = np.full((NUM_CLASSES, 3), 1.0 / 3.0)
        self.defer_threshold = 0.0

    @staticmethod
    def _bin_z(z: np.ndarray) -> np.ndarray:
        bins = np.ones(len(z), dtype=np.int32)
        bins[z < -0.5] = 0
        bins[z > 1.5] = 2
        return bins

    def fit(self, val_ev: EvidenceBatch, val_y_cls: np.ndarray) -> "BayesianJudgeA":
        if len(val_y_cls) == 0:
            return self
        self.standardizer.fit(val_ev.ctx_load, val_ev.s_anom, val_y_cls)
        z_val, _ = self.standardizer.transform(val_ev.ctx_load, val_ev.s_anom)
        z_bins = self._bin_z(z_val)

        counts = np.bincount(val_y_cls, minlength=NUM_CLASSES).astype(np.float64) + 1.0
        self.class_priors = counts / counts.sum()

        for c in range(NUM_CLASSES):
            m = val_y_cls == c
            b_cnt = np.bincount(z_bins[m], minlength=3).astype(np.float64) + 1.0
            self.cpt_anom[c] = b_cnt / b_cnt.sum()
        return self

    def adjudicate_uncertain(
        self,
        ev: EvidenceBatch,
        uncertain_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z_all, _ = self.standardizer.transform(ev.ctx_load, ev.s_anom)
        n_u = int(uncertain_mask.sum())
        if n_u == 0:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int8),
                np.zeros(0, dtype=np.float32),
                z_all,
            )

        probs = ev.probs[uncertain_mask]
        c_sets = ev.pred_sets[uncertain_mask]
        z_bins = self._bin_z(z_all[uncertain_mask])

        # Posterior proportional to P_cls(y|x) * P(z_bin | y) restricted to C(x) when non-empty
        likelihood = self.cpt_anom[:, z_bins].T  # [n_u, K]
        unnorm = probs * likelihood
        non_empty = c_sets.sum(axis=1) > 0
        unnorm = np.where(non_empty[:, None] & (~c_sets), 1e-9, unnorm)
        post = unnorm / np.maximum(unnorm.sum(axis=1, keepdims=True), 1e-12)

        labels = np.argmax(post, axis=1).astype(np.int64)
        max_post = post.max(axis=1).astype(np.float32)
        actions = np.where(labels == 0, ACTION_ACCEPT, ACTION_ALERT).astype(np.int8)
        return labels, actions, max_post, z_all


def run_decision_pipeline(
    ev: EvidenceBatch,
    judge: SecondaryJudge,
) -> DecisionOutput:
    """Route |C(x_t)| == 1 directly and adjudicate |C(x_t)| != 1 via SecondaryJudge."""
    n = len(ev.probs)
    set_sizes = ev.pred_sets.sum(axis=1).astype(np.int32)
    is_uncertain = set_sizes != 1

    final_label = np.argmax(ev.probs, axis=1).astype(np.int64)
    posterior_max = ev.probs.max(axis=1).astype(np.float32)

    # Singleton sets: directly pick the single element in C(x_t)
    singleton_mask = ~is_uncertain
    if singleton_mask.any():
        single_cls = np.argmax(ev.pred_sets[singleton_mask].astype(np.int8), axis=1)
        final_label[singleton_mask] = single_cls
        posterior_max[singleton_mask] = ev.probs[singleton_mask, single_cls]

    actions = np.where(final_label == 0, ACTION_ACCEPT, ACTION_ALERT).astype(np.int8)

    u_labels, u_actions, u_post, z_all = judge.adjudicate_uncertain(ev, is_uncertain)
    if is_uncertain.any():
        final_label[is_uncertain] = u_labels
        actions[is_uncertain] = u_actions
        posterior_max[is_uncertain] = u_post

    return DecisionOutput(
        final_label=final_label,
        action=actions,
        is_uncertain=is_uncertain,
        set_sizes=set_sizes,
        posterior_max=posterior_max,
        z_anom=z_all,
    )
