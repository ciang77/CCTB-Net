# CCTB-Net 主实验与消融实验汇总表（P1 主协议）

| Variant | 说明 | Macro-F1 | Binary-F1 | AUROC(Prob) | AUROC(Anom) | FPR(Select) | FPR(Cons) | FNR(Select) | High-Load FPR | Coverage | Set Size |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0_full_cctb | CCTB-Net (Full: FiLM + TCN + Dual-Task + ACI + Jev-B) | 0.8820 | 0.9997 | 0.9996 | 0.9998 | 0.0054 | 0.0092 | 0.0004 | 0.0032 | 0.9500 | 1.356 |
| A8_wo_aci_jev | w/o ACI + Jev (Argmax Baseline) | 0.8706 | 0.9996 | 0.9996 | 0.9998 | 0.0047 | 0.0047 | 0.0006 | 0.0059 | - | - |
| A11_jev_a | CCTB-Net + ACI + Jev-A (Bayesian CPT) | 0.8651 | 0.9996 | 0.9996 | 0.9998 | 0.0083 | 0.0083 | 0.0004 | 0.0043 | 0.9500 | 1.356 |
| A1_wo_ctx_film | w/o Context & FiLM | 0.9112 | 0.9994 | 0.9997 | 0.9999 | 0.0105 | 0.0112 | 0.0008 | 0.0155 | 0.9500 | 0.984 |
| A5_wo_dynamic_center | w/o Dynamic Center Anomaly Branch | 0.8576 | 0.9994 | 0.9996 | 0.3887 | 0.0046 | 0.0062 | 0.0010 | 0.0016 | 0.9500 | 1.225 |
| A7_wo_dual_cons | w/o Dual-Task Consistency Loss | 0.8526 | 0.9962 | 0.9961 | 0.9991 | 0.0311 | 0.0417 | 0.0061 | 0.0841 | 0.9500 | 1.131 |
| C1_shuffled_ctx | Shuffled Context Control | 0.8970 | 0.9993 | 0.9998 | 0.9999 | 0.0098 | 0.0107 | 0.0009 | 0.0085 | 0.9500 | 1.015 |
