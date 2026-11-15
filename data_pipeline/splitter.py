"""Leakage-free temporal splitting: Episode-Blocked (P1) and Global Chronological (P2)."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from data_pipeline.schema import NUM_CLASSES, SPLIT_TO_ID


def identify_episodes(
    ts_end: np.ndarray,
    y_cls: np.ndarray,
    bucket_sec: int = 60,
    min_episode_flows: int = 5000,
) -> np.ndarray:
    """Group contiguous 1-minute buckets by dominant class and merge tiny fragments."""
    n = len(ts_end)
    if n == 0:
        return np.zeros(0, dtype=np.int32)

    b_ids = (ts_end // bucket_sec).astype(np.int64)
    uniq_b, inv = np.unique(b_ids, return_inverse=True)
    num_b = len(uniq_b)

    # Dominant class per minute bucket by majority vote
    pair_idx = inv * NUM_CLASSES + y_cls
    counts = np.bincount(pair_idx, minlength=num_b * NUM_CLASSES).reshape(num_b, NUM_CLASSES)
    dom_cls = np.argmax(counts, axis=1).astype(np.int32)

    # Merge consecutive buckets with identical dominant class into contiguous runs
    runs: list[list[int]] = []
    for b_i in range(num_b):
        c = int(dom_cls[b_i])
        if not runs or runs[-1][0] != c:
            runs.append([c, b_i, b_i, int(counts[b_i].sum())])
        else:
            runs[-1][2] = b_i
            runs[-1][3] += int(counts[b_i].sum())

    # Fold small runs (< min_episode_flows) into adjacent runs so embargo never wipes out an episode
    merged: list[list[int]] = []
    for r in runs:
        if merged and r[3] < min_episode_flows:
            merged[-1][2] = r[2]
            merged[-1][3] += r[3]
        else:
            merged.append(r)
    if len(merged) > 1 and merged[0][3] < min_episode_flows:
        merged[1][1] = merged[0][1]
        merged[1][3] += merged[0][3]
        merged.pop(0)

    bucket_to_ep = np.zeros(num_b, dtype=np.int32)
    ep_dom_cls = np.zeros(len(merged), dtype=np.int32)
    for ep_id, (c_dom, b_start, b_end, _) in enumerate(merged):
        bucket_to_ep[b_start : b_end + 1] = ep_id
        ep_dom_cls[ep_id] = c_dom

    flow_ep = bucket_to_ep[inv]
    # Refine transition boundaries so tail flows of episode k-1 inside a transition bucket
    # are not placed into the head (Train split) of episode k
    for ep_id in range(1, len(merged)):
        b_trans = merged[ep_id][1]
        in_b = np.flatnonzero((inv == b_trans) | (inv == max(0, b_trans - 1)))
        if len(in_b) > 0:
            new_c = ep_dom_cls[ep_id]
            first_new = in_b[y_cls[in_b] == new_c]
            if len(first_new) > 0:
                cut_flow = int(first_new[0])
                flow_ep[in_b[0] : cut_flow] = ep_id - 1
                flow_ep[cut_flow : in_b[-1] + 1] = ep_id

    return flow_ep


def split_temporal_with_purge(
    ts_end: np.ndarray,
    y_cls: np.ndarray,
    protocol: str = "P1",
    split_ratios: Sequence[float] = (0.60, 0.10, 0.10, 0.20),
    episode_bucket_sec: int = 60,
    min_episode_flows: int = 5000,
    embargo_sec: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign each flow to train(0), val(1), cal(2), test(3), or purge(-1).

    Returns:
        raw_split: int8 array in {0, 1, 2, 3} (segment membership before boundary embargo)
        target_split: int8 array in {-1, 0, 1, 2, 3} (-1 where purged inside embargo zone)
        episode_id: int32 array of episode indices
    """
    n = len(ts_end)
    ratios = np.asarray(split_ratios, dtype=np.float64)
    ratios = ratios / ratios.sum()
    cum_ratios = np.cumsum(ratios)[:-1]  # 3 cut points -> 4 splits (Train, Val, Cal, Test)

    if protocol.upper() == "P2":
        episode_id = np.zeros(n, dtype=np.int32)
    else:
        episode_id = identify_episodes(
            ts_end,
            y_cls,
            bucket_sec=episode_bucket_sec,
            min_episode_flows=min_episode_flows,
        )

    raw_split = np.zeros(n, dtype=np.int8)
    target_split = np.zeros(n, dtype=np.int8)

    # Identify contiguous (episode_id) blocks along the sorted timeline
    change_pts = np.flatnonzero(np.diff(episode_id) != 0) + 1
    ep_starts = np.concatenate(([0], change_pts))
    ep_ends = np.concatenate((change_pts, [n]))

    for s_idx, e_idx in zip(ep_starts, ep_ends):
        m = e_idx - s_idx
        if m < 8:
            raw_split[s_idx:e_idx] = SPLIT_TO_ID["train"]
            target_split[s_idx:e_idx] = SPLIT_TO_ID["train"]
            continue

        cuts = [s_idx + int(round(r * m)) for r in cum_ratios]
        # Advance cut to the next timestamp boundary so identical timestamps never straddle two splits
        aligned_cuts: list[int] = []
        prev = s_idx + 1
        for idx_cut in cuts:
            c = max(prev, min(e_idx - 1, idx_cut))
            while c < e_idx - 1 and ts_end[c] == ts_end[c - 1]:
                c += 1
            aligned_cuts.append(c)
            prev = min(e_idx - 1, c + 1)

        seg_bounds = [s_idx] + aligned_cuts + [e_idx]
        for seg_id in range(4):
            a, b = seg_bounds[seg_id], seg_bounds[seg_id + 1]
            if b <= a:
                continue
            raw_split[a:b] = seg_id
            target_split[a:b] = seg_id
            if seg_id > 0 and embargo_sec > 0:
                t_boundary = ts_end[a]
                purge_mask = (ts_end[a:b] - t_boundary) < embargo_sec
                # Keep at least the tail of very short segments in synthetic / tiny slices
                if purge_mask.all() and (b - a) > 2:
                    purge_mask[-max(1, (b - a) // 2) :] = False
                target_split[a:b][purge_mask] = SPLIT_TO_ID["purge"]

    return raw_split, target_split, episode_id
