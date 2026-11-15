"""Authoritative column schema and constants for TON_IoT Network & Linux datasets."""
from __future__ import annotations

from typing import Dict, List

CLASS_NAMES: List[str] = [
    "normal",
    "scanning",
    "ddos",
    "dos",
    "xss",
    "password",
    "backdoor",
    "injection",
    "ransomware",
    "mitm",
]
CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASS_NAMES)
UNKNOWN_ATTACK_ID: int = NUM_CLASSES  # 10

SPLIT_NAMES: List[str] = ["train", "val", "cal", "test"]
SPLIT_TO_ID: Dict[str, int] = {"purge": -1, "train": 0, "val": 1, "cal": 2, "test": 3}

# Identifiers / shortcut columns that must NEVER enter model features
NET_ID_DROP_COLS: List[str] = [
    "src_ip",
    "dst_ip",
    "src_port",
    "dns_query",
    "ssl_subject",
    "ssl_issuer",
    "http_uri",
    "http_referrer",
    "http_user_agent",
    "weird_addl",
]

# 10 raw numeric columns + 4 derived numeric columns = 14 numeric features
NET_NUM_RAW_COLS: List[str] = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "missed_bytes",
    "src_pkts",
    "src_ip_bytes",
    "dst_pkts",
    "dst_ip_bytes",
    "http_request_body_len",
    "http_response_body_len",
]
NET_NUM_DERIVED_COLS: List[str] = [
    "bytes_ratio",
    "pkts_ratio",
    "bytes_per_pkt",
    "dt_prev",
]
NET_NUM_COLS: List[str] = NET_NUM_RAW_COLS + NET_NUM_DERIVED_COLS  # 14 features

# 22 raw categorical columns + 1 bucketed dst_port = 23 categorical features
NET_CAT_RAW_COLS: List[str] = [
    "proto",
    "service",
    "conn_state",
    "dns_qclass",
    "dns_qtype",
    "dns_rcode",
    "dns_AA",
    "dns_RD",
    "dns_RA",
    "dns_rejected",
    "ssl_version",
    "ssl_cipher",
    "ssl_resumed",
    "ssl_established",
    "http_trans_depth",
    "http_method",
    "http_version",
    "http_status_code",
    "http_orig_mime_types",
    "http_resp_mime_types",
    "weird_name",
    "weird_notice",
]
NET_CAT_COLS: List[str] = NET_CAT_RAW_COLS + ["dst_port_bucket"]  # 23 features

NET_REQUIRED_CSV_COLS: List[str] = (
    ["ts", "src_ip", "dst_ip", "dst_port"]
    + NET_NUM_RAW_COLS
    + NET_CAT_RAW_COLS
    + ["label", "type"]
)

# Linux host context schema (label and type are explicitly dropped on ingest)
LINUX_SOURCES: Dict[str, Dict[str, List[str]]] = {
    "process": {
        "patterns": ["Linux_process_*.csv", "linux_process_*.csv"],
        "num_cols": ["PID", "TRUN", "TSLPI", "TSLPU", "NICE", "PRI", "RTPR", "CPUNR", "EXC", "CPU"],
        "cat_cols": ["POLI", "Status", "State", "CMD"],
        "drop_cols": ["label", "type"],
    },
    "memory": {
        "patterns": ["linux_memory*.csv", "Linux_memory*.csv"],
        "num_cols": ["PID", "MINFLT", "MAJFLT", "VSTEXT", "VSIZE", "RSIZE", "VGROW", "RGROW", "MEM"],
        "cat_cols": ["CMD"],
        "drop_cols": ["label", "type"],
    },
    "disk": {
        "patterns": ["linux_disk_*.csv", "Linux_disk_*.csv"],
        "num_cols": ["PID", "RDDSK", "WRDSK", "WCANCL", "DSK"],
        "cat_cols": ["CMD"],
        "drop_cols": ["label", "type"],
    },
}

# 12 base aggregate columns per 5s bucket -> 12 raw + 12 roll60s + 12 ewm300s = 36 features
CTX_BASE_COLS: List[str] = [
    "proc_rows",
    "proc_cpu_sum",
    "proc_cpu_max",
    "proc_trun_sum",
    "proc_tslpi_sum",
    "proc_state_r_ratio",
    "mem_minflt_sum",
    "mem_majflt_sum",
    "mem_rsize_sum",
    "mem_usage_max",
    "disk_rddsk_sum",
    "disk_wrdsk_sum",
]
CTX_LOAD_KEY_COLS: List[str] = [
    "proc_cpu_sum_roll",
    "proc_trun_sum_roll",
    "mem_rsize_sum_roll",
    "disk_rddsk_sum_roll",
    "disk_wrdsk_sum_roll",
]
