# CCTB-Net 实验工程交接文档（HANDOFF）

本文档说明 **CCTB-Net（Context-Conditioned Temporal Behavior Network）** 端到端实验代码的目录结构、常用运行命令及关键超参数位置，可直接用于后续全量训练、基线对比与消融实验。

---

## 1. 环境说明

- **Python 解释器**：`D:\desktop\journal\.venv\Scripts\python.exe`（Python 3.11.9，PyTorch 2.5.1+cu121，Numba 0.68.1）
- 所有命令均在仓库根目录 `D:\desktop\journal` 下执行。

---

## 2. 核心命令速查

### 2.1 防泄漏与保形推断单元测试
验证 Linux 脏数据清洗、标签列剥离、`ts + duration` 全局排序、P1/P2 边界 Embargo、Train-Only Scaler/词表以及 ACI 覆盖率性质：
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_no_leakage.py -v
```

### 2.2 快速冒烟测试（1% 采样切片，1 Epoch 端到端全链路）
依次执行 **数据清洗与多源对齐 → CCTB-Net 训练（AMP） → 温度缩放校准 → Numba 加速 ACI 在线保形推断 → Jev 二次裁决 → 指标输出**：
```powershell
.\.venv\Scripts\python.exe -m experiments.run_train --config configs/fast_dev.yaml
```
- 预处理产物缓存于：`data/processed/fast_dev/`
- 训练日志与权重保存于：`outputs/runs/cctb_net_fast_dev/seed_42/`（含 `best.pt`、`last.pt`、`train_log.csv`、`preds/*.npz`、`metrics.json`）

### 2.3 全量数据预处理与全量训练评估（AMP 混合精度）
1. **单独执行全量预处理**（首次运行会生成 memmap 数组并缓存到 `data/processed/full_eval/`）：
   ```powershell
   .\.venv\Scripts\python.exe -m experiments.run_preprocess --config configs/full_eval.yaml
   ```
2. **执行全量模型训练 + ACI + Jev 评估**：
   ```powershell
   .\.venv\Scripts\python.exe -m experiments.run_train --config configs/full_eval.yaml
   ```
3. **仅重跑 Phase 3（ACI + Jev 二次决策参数调优，无需重训模型）**：
   ```powershell
   .\.venv\Scripts\python.exe -m experiments.run_decision --config configs/full_eval.yaml decision.alpha=0.05 decision.eta=0.005 decision.judge_type=jev_b
   ```

### 2.4 消融实验矩阵一键运行
自动遍历 `A0_full_cctb`、`A1_wo_ctx_film`、`A2_concat_fusion`、`A3_film_gamma_only`、`A5_wo_dual_task`、`A7_wo_sad_loss`、`A11_jev_a`，并汇总至 `outputs/runs/ablation_summary.json`：
```powershell
.\.venv\Scripts\python.exe -m experiments.run_ablation --config configs/full_eval.yaml
```
也可以在命令行直接通过 `key=value` 覆盖任意配置运行单个变体，例如：
```powershell
# 切换 P2 全局时序切分（开放集未知攻击压力测试）
.\.venv\Scripts\python.exe -m experiments.run_train --config configs/full_eval.yaml project.profile=full_p2 project.name=cctb_p2 data.protocol=P2

# 切换 Jev 方案 A（贝叶斯条件概率推理裁决器）
.\.venv\Scripts\python.exe -m experiments.run_decision --config configs/full_eval.yaml decision.judge_type=jev_a
```

---

## 3. 关键模块与超参数位置

| 模块 | 文件路径 | 关键实现与可调参数 |
|---|---|---|
| 全局默认配置 | `configs/default.yaml` | 定义全部数据、模型、损失、训练与决策超参数默认值 |
| 快速验证配置 | `configs/fast_dev.yaml` | `data.sample_frac: 0.01`、`data.linux_stride: 20`、`train.epochs: 1` |
| 全量实验配置 | `configs/full_eval.yaml` | `data.sample_frac: 1.0`、`train.amp: true`、`train.epochs: 20` |
| 字段与防泄漏契约 | `data_pipeline/schema.py` | 14 维流数值特征、23 维流类别特征、36 维多尺度主机上下文特征，显式剥离 IP/URI 等捷径列与 Linux 标签列 |
| Linux 脏数据清洗 | `data_pipeline/ingest_linux.py` | 解析 `1184K`/`97%`/空格分隔数值，自动丢弃列错位脏行（如 `DSK=apache2`）并剥离 `label/type` |
| 多尺度上下文构建 | `data_pipeline/context_builder.py` | $\Delta=5\text{s}$ 时间桶聚合 + 60s 因果滚动均值 + 300s 因果 EWMA + 3 路观测掩码 `m_ctx` |
| 时序切分与 Embargo | `data_pipeline/splitter.py` | 支持 `P1`（攻击片段内 60/10/10/20 时序切分）与 `P2`（全局时序切分），切分边界强制执行 Embargo 隔离带 |
| Train-Only 归一化 | `data_pipeline/scalers.py` | `TrainOnlyRobustScaler`、`TrainOnlyCategoricalEncoder`（未见类别映射为 OOV=1）、`TrainOnlyPortBucketer` |
| CCTB-Net 主模型 | `models/cctb_net.py` | `FlowEncoder`、`ContextEncoder`、`FiLMLayer`（$\gamma = 1 + \tanh(\Delta\gamma)$ 恒等初始化）、膨胀 `TCNResidualBlock`、动态中心 `AnomalyBranch`、四项联合损失 `CCTBJointLoss` |
| 训练与温度校准 | `engine/trainer.py` | AMP 混合精度训练、余弦退火、早停、Val 集 LBFGS 温度缩放与 `val/cal/test` 预测导出 |
| 自适应保形推断 | `decision/aci.py` | 随机化 APS / LAC 非一致性分数 + Numba `@njit` Fenwick 树滑动分位数在线更新（复杂度 $O(N \log B)$） |
| Jev 二次决策器 | `decision/secondary_judge.py` | 统一 `SecondaryJudge` 接口；实现方案 B `StructuredRuleJudge`（融合上下文相对偏离 $z_{anom}$ 与规则 R1–R4）及方案 A `BayesianJudgeA` |
| 论文评估指标 | `evaluation/metrics.py` | 输出 `macro_f1`、`auroc_prob`、`auroc_anom`、`fpr_selective`、`fnr_selective`、**`high_load_normal_fpr`**（高负载时段正常流误报率）、`empirical_coverage`、`avg_set_size` |
