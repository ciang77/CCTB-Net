"""Entity-stream ordering, inter-arrival feature computation, and O(1) window indexing."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from data_pipeline.schema import NET_NUM_RAW_COLS


def build_entity_streams(
    entity_key: np.ndarray,
    ts_end: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group flows by entity_key along the globally sorted timeline.

    Assumes input arrays are already globally sorted by (ts_end, file_idx, row_idx).

    Returns:
        stream_order: int32 [N] mapping stream-contiguous position -> flow_id
        stream_pos: int32 [N] mapping flow_id -> position in stream_order
        stream_start: int32 [N] mapping flow_id -> start position of its entity stream in stream_order
        dt_prev: float32 [N] inter-arrival time (seconds) since previous flow in same entity stream
    """
    n = len(ts_end)
    flow_ids = np.arange(n, dtype=np.int32)
    # Stable sort by entity_key preserves global (ts_end, file_idx, row_idx) order within each entity
    stream_order = np.argsort(entity_key, kind="mergesort").astype(np.int32)

    sorted_keys = entity_key[stream_order]
    sorted_ts = ts_end[stream_order]

    is_new_stream = np.empty(n, dtype=np.bool_)
    is_new_stream[0] = True
    if n > 1:
        is_new_stream[1:] = sorted_keys[1:] != sorted_keys[:-1]

    # Start index in stream_order for each position
    pos_idx = np.arange(n, dtype=np.int32)
    start_markers = np.where(is_new_stream, pos_idx, 0)
    sorted_stream_start = np.maximum.accumulate(start_markers).astype(np.int32)

    # Inter-arrival time in stream_order
    sorted_dt = np.zeros(n, dtype=np.float32)
    if n > 1:
        diff = np.maximum(0.0, sorted_ts[1:] - sorted_ts[:-1])
        sorted_dt[1:] = np.where(is_new_stream[1:], 0.0, diff).astype(np.float32)

    # Scatter back to flow_id indexing
    stream_pos = np.empty(n, dtype=np.int32)
    stream_pos[stream_order] = pos_idx

    stream_start = np.empty(n, dtype=np.int32)
    stream_start[stream_order] = sorted_stream_start

    dt_prev = np.empty(n, dtype=np.float32)
    dt_prev[stream_order] = sorted_dt

    return stream_order, stream_pos, stream_start, dt_prev


def compute_full_numeric_features(num_raw: np.ndarray, dt_prev: np.ndarray) -> np.ndarray:
    """Combine 10 raw numeric columns with 4 derived columns -> [N, 14] float32."""
    n = len(num_raw)
    out = np.zeros((n, 14), dtype=np.float32)
    out[:, :10] = num_raw

    idx_sbytes = NET_NUM_RAW_COLS.index("src_bytes")
    idx_dbytes = NET_NUM_RAW_COLS.index("dst_bytes")
    idx_spkts = NET_NUM_RAW_COLS.index("src_pkts")
    idx_dpkts = NET_NUM_RAW_COLS.index("dst_pkts")

    s_bytes = num_raw[:, idx_sbytes].astype(np.float64)
    d_bytes = num_raw[:, idx_dbytes].astype(np.float64)
    s_pkts = num_raw[:, idx_spkts].astype(np.float64)
    d_pkts = num_raw[:, idx_dpkts].astype(np.float64)

    tot_bytes = s_bytes + d_bytes
    tot_pkts = s_pkts + d_pkts

    out[:, 10] = ((s_bytes + 1.0) / (d_bytes + 1.0)).astype(np.float32)  # bytes_ratio
    out[:, 11] = ((s_pkts + 1.0) / (d_pkts + 1.0)).astype(np.float32)    # pkts_ratio
    out[:, 12] = (tot_bytes / np.maximum(1.0, tot_pkts)).astype(np.float32)  # bytes_per_pkt
    out[:, 13] = np.maximum(0.0, dt_prev).astype(np.float32)             # dt_prev
    return out
