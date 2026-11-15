"""Numba-accelerated Adaptive Conformal Inference (ACI) with Fenwick-tree sliding quantiles."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numba import njit


def compute_nonconformity_scores(
    probs: np.ndarray,
    score_type: str = "aps",
    seed: int = 42,
) -> np.ndarray:
    """Compute nonconformity score matrix S(x_i, y) in [0, 1] for all classes y in 0..K-1.

    For randomized APS:
      S(x_i, y) = sum_{c : P(c|x_i) > P(y|x_i)} P(c|x_i) + u_i * P(y|x_i),  u_i ~ U(0, 1).
    For LAC:
      S(x_i, y) = 1 - P(y|x_i).
    """
    n, k = probs.shape
    if score_type.lower() == "lac":
        return np.clip(1.0 - probs, 0.0, 1.0).astype(np.float32)

    rng = np.random.default_rng(seed)
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum_p = np.cumsum(sorted_p, axis=1)
    u = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    sorted_scores = cum_p - (1.0 - u) * sorted_p

    scores = np.empty_like(sorted_scores, dtype=np.float32)
    np.put_along_axis(scores, order, np.clip(sorted_scores, 0.0, 1.0), axis=1)
    return scores


@njit(cache=True)
def _fenwick_add(tree: np.ndarray, idx: int, delta: int) -> None:
    n = len(tree) - 1
    i = idx + 1
    while i <= n:
        tree[i] += delta
        i += i & (-i)


@njit(cache=True)
def _fenwick_kth(tree: np.ndarray, target_rank: int, max_pow2: int) -> int:
    """Return 0-based histogram bin index whose cumulative count reaches target_rank (>=1)."""
    n = len(tree) - 1
    idx = 0
    step = max_pow2
    rem = target_rank
    while step > 0:
        nxt = idx + step
        if nxt <= n and tree[nxt] < rem:
            idx = nxt
            rem -= tree[nxt]
        step >>= 1
    if idx >= n:
        return n - 1
    return idx


@njit(cache=True)
def _aci_streaming_kernel(
    cal_true_scores: np.ndarray,
    test_score_mat: np.ndarray,
    test_y_cls: np.ndarray,
    alpha: float,
    eta: float,
    window_size: int,
    num_bins: int,
    label_delay: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numba O(N log B) streaming ACI loop with sliding calibration buffer and label delay."""
    n_test, k_cls = test_score_mat.shape
    pred_sets = np.zeros((n_test, k_cls), dtype=np.bool_)
    alpha_traj = np.zeros(n_test, dtype=np.float32)
    q_traj = np.zeros(n_test, dtype=np.float32)

    tree = np.zeros(num_bins + 1, dtype=np.int64)
    max_pow2 = 1
    while (max_pow2 << 1) <= num_bins:
        max_pow2 <<= 1

    ring_cap = max(1, window_size)
    ring = np.zeros(ring_cap, dtype=np.int32)
    ring_head = 0
    ring_count = 0

    # Initialize sliding window from Calibration split scores
    n_cal = len(cal_true_scores)
    start_cal = max(0, n_cal - ring_cap)
    for i in range(start_cal, n_cal):
        b = int(cal_true_scores[i] * num_bins)
        if b < 0:
            b = 0
        elif b >= num_bins:
            b = num_bins - 1
        ring[ring_head] = b
        ring_head = (ring_head + 1) % ring_cap
        ring_count += 1
        _fenwick_add(tree, b, 1)

    alpha_t = alpha
    delay = max(0, label_delay)

    for t in range(n_test):
        alpha_traj[t] = alpha_t
        if alpha_t <= 0.0:
            q_t = 1.000001
            for c in range(k_cls):
                pred_sets[t, c] = True
        elif alpha_t >= 1.0:
            q_t = -1.0
        else:
            q_level = (1.0 - alpha_t) * (ring_count + 1.0)
            rank = int(np.ceil(q_level))
            if rank < 1:
                rank = 1
            elif rank > ring_count:
                rank = ring_count
            bin_idx = _fenwick_kth(tree, rank, max_pow2)
            q_t = float(bin_idx + 1) / float(num_bins)
            for c in range(k_cls):
                if test_score_mat[t, c] <= q_t:
                    pred_sets[t, c] = True

        q_traj[t] = q_t

        # Delayed feedback update at stepfb = t - delay
        fb_t = t - delay
        if fb_t >= 0:
            y_fb = test_y_cls[fb_t]
            err_fb = 0.0 if pred_sets[fb_t, y_fb] else 1.0
            alpha_t = alpha_t + eta * (alpha - err_fb)
            # Clip alpha_t slightly outside [0, 1] so extreme excursions recover smoothly
            if alpha_t < -0.05:
                alpha_t = -0.05
            elif alpha_t > 1.05:
                alpha_t = 1.05

            s_fb = test_score_mat[fb_t, y_fb]
            b_new = int(s_fb * num_bins)
            if b_new < 0:
                b_new = 0
            elif b_new >= num_bins:
                b_new = num_bins - 1

            if ring_count == ring_cap:
                b_old = ring[ring_head]
                _fenwick_add(tree, b_old, -1)
            else:
                ring_count += 1
            ring[ring_head] = b_new
            ring_head = (ring_head + 1) % ring_cap
            _fenwick_add(tree, b_new, 1)

    return pred_sets, alpha_traj, q_traj


def run_aci_stream(
    cal_probs: np.ndarray,
    cal_y_cls: np.ndarray,
    test_probs: np.ndarray,
    test_y_cls: np.ndarray,
    alpha: float = 0.05,
    eta: float = 0.005,
    window_size: int = 20000,
    num_bins: int = 4096,
    label_delay: int = 0,
    score_type: str = "aps",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Run online Adaptive Conformal Inference (ACI) over a stream of test flows."""
    cal_scores_all = compute_nonconformity_scores(cal_probs, score_type=score_type, seed=seed)
    cal_true_scores = cal_scores_all[np.arange(len(cal_y_cls)), cal_y_cls].astype(np.float32)

    test_score_mat = compute_nonconformity_scores(test_probs, score_type=score_type, seed=seed + 1)
    if len(test_probs) == 0:
        return {
            "pred_sets": np.zeros((0, cal_probs.shape[1]), dtype=np.bool_),
            "alpha_traj": np.zeros(0, dtype=np.float32),
            "q_traj": np.zeros(0, dtype=np.float32),
            "score_mat": test_score_mat,
        }

    pred_sets, alpha_traj, q_traj = _aci_streaming_kernel(
        cal_true_scores=cal_true_scores,
        test_score_mat=test_score_mat,
        test_y_cls=test_y_cls.astype(np.int64),
        alpha=float(alpha),
        eta=float(eta),
        window_size=int(window_size),
        num_bins=int(num_bins),
        label_delay=int(label_delay),
    )
    return {
        "pred_sets": pred_sets,
        "alpha_traj": alpha_traj,
        "q_traj": q_traj,
        "score_mat": test_score_mat,
    }


def run_static_cp(
    cal_probs: np.ndarray,
    cal_y_cls: np.ndarray,
    test_probs: np.ndarray,
    alpha: float = 0.05,
    score_type: str = "aps",
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Static Split Conformal Prediction baseline (fixed calibration quantile, no online update)."""
    cal_scores_all = compute_nonconformity_scores(cal_probs, score_type=score_type, seed=seed)
    cal_true_scores = cal_scores_all[np.arange(len(cal_y_cls)), cal_y_cls]
    n_cal = len(cal_true_scores)
    q_level = min(1.0, np.ceil((1.0 - alpha) * (n_cal + 1)) / max(1, n_cal))
    q_hat = float(np.quantile(cal_true_scores, q_level)) if n_cal > 0 else 1.0

    test_score_mat = compute_nonconformity_scores(test_probs, score_type=score_type, seed=seed + 1)
    pred_sets = test_score_mat <= q_hat
    return {
        "pred_sets": pred_sets,
        "alpha_traj": np.full(len(test_probs), alpha, dtype=np.float32),
        "q_traj": np.full(len(test_probs), q_hat, dtype=np.float32),
        "score_mat": test_score_mat,
    }
