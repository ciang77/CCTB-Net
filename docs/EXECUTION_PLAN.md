# CCTB-Net 端到端实验执行计划（Execution Plan）

> Context-Conditioned Temporal Behavior Network —— 动态网络环境下的上下文条件化异常检测
> 本文档基于对本地 TON_IoT 数据的实测探查编写（2026-11-15），所有数字均来自 `Processed_Network_dataset/` 与 `Processed_Linux_dataset/` 的统计结果。

---

## 0. 数据实测结论（直接决定后续设计）

| 编号 | 实测事实 | 对设计的影响 |
|---|---|---|
| D1 | 流量 23 个 CSV，共 **22,339,021** 条流，时间 `2019-04-02 09:45` → `2019-04-29 14:45`（UTC）。文件之间首尾衔接，但 `_12/13/14/16/17/21/22/23` **文件内部 `ts` 不严格有序** | 必须全局**稳定排序**（`ts_end, file_idx, row_idx`） |
| D2 | **类别倒置失衡**：normal 仅 796,380（3.6%）；scanning 7.14M、ddos 6.17M、dos 3.38M、xss 2.11M、password 1.72M、backdoor 508K、injection 453K、ransomware 72.8K、**mitm 仅 1,052** | 少数类是 normal 和 mitm；FPR 的分母只有正常流，AUPRC 需以 normal 为正类另报 |
| D3 | **攻击按天排期**：04-02~04 纯正常；04-23 scanning → 04-24 scanning/dos → 04-25 ddos/dos/injection → 04-26 ddos/password → 04-27 password/xss → 04-28 backdoor/ransomware → 04-29 backdoor/mitm | **纯全局时序切分会让测试集只剩 backdoor/ransomware/mitm（训练未见类）** → 主协议必须采用“片段内时序切分” |
| D4 | Linux 6 个文件约 585 万行（atop 进程级采样），覆盖 04-02 → 05-20；每个 `ts` 仅 **1–3 行进程记录**（采样稀疏，并非整机快照） | 上下文须做多尺度滚动聚合与观测掩码，不能把单桶当作完整系统状态 |
| D5 | **按流统计，99.95% 的流量在同一 5s 时间桶内有 Linux 记录**；5s 桶重叠率 96.1%，60s 桶重叠率 96.1% | Δ=5s 时间桶对齐可行，稀疏风险可控 |
| D6 | Linux 数据较脏：带单位后缀（`1184K`、`97%`）、列错位（`DSK=apache2`）、值内含空格（`VGROW="0456 6"`）；同时含 `label/type` 列 | 需要健壮解析器；**Linux 的 label/type 必须丢弃**（防泄漏） |
| D7 | `memory1` 与 `memory2` 无完全重复行，`_1/_2` 时间范围高度重叠 | 取并集并去重，`source_file` 保留为审计字段（不作为特征） |
| D8 | 本机：Python 3.9.4（无 pandas/torch），**GTX 1650 4GB**，内存 16GB | 需要新建 3.10+ 环境；采用 parquet + memmap + 只存索引的窗口化，小模型 + AMP |

---

## 1. 项目工程目录树（Project Structure）

```
journal/                          # 仓库根目录（GitHub: ciang77/CCTB-Net）
├── configs/
│   ├── data.yaml                 # 原始路径、Δ、L、T、切分协议与比例、purge 间隔、按类采样上限
│   ├── model.yaml                # d_model、编码器/TCN 结构、FiLM 形式（逐层/单层、γ/β 开关）
│   ├── train.yaml                # 优化器、调度器、损失权重 λ、采样器、AMP、随机种子
│   ├── decision.yaml             # α、η、ACI 窗口 W、标签延迟 D、非一致性分数、Jev 方案与代价矩阵
│   └── experiments/              # 覆盖式配置：baseline_*.yaml、ablation_*.yaml、protocol_p2.yaml
├── data_pipeline/
│   ├── schema.py                 # 列角色登记：数值 / 类别 / 标识符(丢弃) / 标签；唯一权威来源
│   ├── ingest_network.py         # 逐文件读取 CSV → 类型化清洗 → 计算 ts_end → parquet 分片
│   ├── ingest_linux.py           # 健壮解析（K/M/G/% 单位、列错位、空格值）→ 丢弃 label/type → parquet
│   ├── context_builder.py        # Linux → Δ=5s 时间桶上下文矩阵 + 多尺度因果滚动 + 观测掩码/观测时距
│   ├── episodes.py               # 基于 1 分钟主导类别识别攻击片段（episode）
│   ├── splitter.py               # 协议 P1（片段内时序切分）/ P2（全局时序切分）+ purge/embargo
│   ├── scalers.py                # 仅在 Train 上拟合：log1p + RobustScaler、类别词表、OOV、裁剪分位点
│   ├── windowing.py              # 实体流索引（host-pair）、窗口结束索引（只存 int64 索引，不展开）
│   ├── dataset.py                # 基于 memmap 的 Dataset、时间均匀按类采样上限、平方根频率采样器
│   └── profiling.py              # Phase 1 探查报告（时间重叠、类别×时间热图、缺失率）
├── models/
│   ├── encoders.py               # FlowEncoder（E_net）、ContextEncoder（E_ctx）
│   ├── film.py                   # FiLM 生成器与调制（恒等初始化）
│   ├── tcn.py                    # 因果膨胀卷积残差块
│   ├── heads.py                  # AnomalyHead（上下文条件化中心 + SAD）、ThreatHead
│   ├── cctbnet.py                # 组装 + 消融开关（use_ctx / fusion=film|concat / dual_task …）
│   ├── losses.py                 # 联合损失：L_cls / L_bin / L_SAD / L_cons、不确定性加权
│   └── baselines/                # mlp.py lstm.py bigru.py tcn_only.py transformer.py
│                                 # tabular.py(XGBoost/RF) deep_svdd.py autoencoder.py iforest.py
├── decision/
│   ├── nonconformity.py          # LAC / APS / RAPS；边际与 Mondrian（按类）两种模式
│   ├── temperature.py            # 在 Val 上做温度缩放（共形前置）
│   ├── aci.py                    # 静态 CP、ACI、DtACI；滑动校准缓冲；标签延迟模拟（numba）
│   ├── router.py                 # |C|=1 直接放行；|C|≥2 / |C|=0 → 不确定样本流转
│   ├── judge_base.py             # SecondaryJudge 抽象接口：decide(evidence) -> Decision
│   ├── jev_a.py                  # 方案 A：贝叶斯网络结构化推理引擎（pgmpy）
│   ├── jev_b.py                  # 方案 B：上下文规则 + s_anom + 候选集的结构化二次裁决器
│   └── pipeline.py               # 端到端：模型输出 → 共形 → 路由 → 二次裁决 → 最终决策
├── engine/
│   ├── trainer.py                # 训练循环、AMP、梯度裁剪、早停、checkpoint
│   └── inference.py              # 批量推理，导出 logits / s_anom / h_ctx 摘要到 parquet
├── evaluation/
│   ├── metrics.py                # FPR、FNR、Macro-F1、AUROC、AUPRC、逐类指标
│   ├── conformal_metrics.py      # 经验覆盖率（整体/逐类/滚动）、集合大小、SSC、单元素率/空集率
│   ├── selective.py              # 风险-覆盖率曲线、AURC、拒判预算
│   ├── significance.py           # 多种子统计、配对 bootstrap、McNemar
│   └── plots.py                  # 论文图表统一出口
├── experiments/
│   ├── run_profile.py            # Phase 1
│   ├── run_preprocess.py         # Phase 1：ingest → context → split → scaler → windows
│   ├── run_train.py              # Phase 2：python -m experiments.run_train exp=cctbnet seed=0
│   ├── run_decision.py           # Phase 3
│   ├── run_baselines.py          # Phase 4
│   ├── run_ablation.py           # Phase 4
│   └── aggregate_results.py      # Phase 5：汇总成表格与图
├── tests/
│   ├── test_no_leakage.py        # 断言：时间边界、purge 区无窗口、scaler/词表只见 Train、上下文严格过去
│   ├── test_shapes.py            # 张量形状与掩码
│   ├── test_linux_parser.py      # 脏值解析用例
│   └── test_aci.py               # ACI 长期覆盖率性质（合成数据）
├── docs/EXECUTION_PLAN.md
├── requirements.txt
├── data/        (gitignore)      # interim/*.parquet、processed/*.npy(memmap)、artifacts/(scaler, vocab)
└── outputs/     (gitignore)      # runs/{exp}/{seed}/ ckpt、metrics.json、preds.parquet、figures/
```

---

## 2. 分阶段实施路线图

### Phase 0 — 环境准备（0.5 天）

- 用 Python 3.10/3.11 新建 `.venv`（当前系统 Python 是 3.9.4）。
- 依赖：`torch`（CUDA 12.1 构建，驱动 532 支持）、`pyarrow`、`polars` 或 `pandas`、`numpy`、`numba`、`scikit-learn`、`xgboost`、`pgmpy`、`omegaconf`、`tensorboard`、`matplotlib`、`seaborn`、`pytest`。
- Windows 注意：DataLoader 使用 `spawn` → Dataset 定义在模块顶层，入口加 `if __name__ == "__main__":`；`num_workers ≤ 2`，开启 `persistent_workers`。
- **完成标准（DoD）**：`torch.cuda.is_available()` 为 True，`pytest tests/test_shapes.py` 可运行。

### Phase 1 — 数据探查、时间重叠分析与多源对齐管线（3–4 天）

#### 1.1 探查报告（`run_profile.py`）
输出：类别×日期热图、每 5s/60s 桶的流量速率、Linux 桶覆盖率时间线、Linux 脏值统计与丢弃率、两路数据的 `ts` 交集区间。§0 的数字作为对照基准。

#### 1.2 流量清洗与特征角色（`schema.py` 为唯一权威来源）

| 角色 | 列 | 处理 |
|---|---|---|
| 排序/对齐（非特征） | `ts`、`duration` | `ts_end = ts + duration`，作为“该流记录可用时刻”（Zeek 的 `ts` 是连接**开始**时间，而字节数、时长要到连接**结束**才知道 → 以 `ts_end` 排序和对齐才因果正确） |
| 丢弃（标识符/捷径） | `src_ip`、`dst_ip`、`src_port`、`dns_query`、`ssl_subject`、`ssl_issuer`、`http_uri`、`http_referrer`、`http_user_agent`、`weird_addl` | IP 只用于构造实体流的 key，**绝不进入特征** |
| 数值（14） | `duration, src_bytes, dst_bytes, missed_bytes, src_pkts, src_ip_bytes, dst_pkts, dst_ip_bytes, http_request_body_len, http_response_body_len` + 派生 `bytes_ratio, pkts_ratio, bytes_per_pkt, log Δt_prev(流内间隔)` | `log1p` → 以 Train 的 99.9 分位裁剪 → RobustScaler（仅 Train 拟合） |
| 类别（23） | `proto, service, conn_state, dns_qclass, dns_qtype, dns_rcode, dns_AA, dns_RD, dns_RA, dns_rejected, ssl_version, ssl_cipher, ssl_resumed, ssl_established, http_trans_depth, http_method, http_version, http_status_code, http_orig_mime_types, http_resp_mime_types, weird_name, weird_notice, dst_port_bucket` | 词表仅由 Train 构建；`'-'` → 0（缺失），未见值 → 1（OOV）；`dst_port` 只按 well-known/registered/ephemeral + Train 中 Top-32 端口分桶 |
| 标签 | `type`（10 类）、`label`（0/1） | `y_cls ∈ {0..9}`，normal=0；`y_bin = label` |

产出：`data/interim/network/*.parquet`（列裁剪、类型压缩：float32 / int16），全局 `flow_id` 与稳定排序。

#### 1.3 Linux 健壮解析（`ingest_linux.py`）
- 单位后缀：`K/M/G` → ×10³/10⁶/10⁹，`%` → 去掉；值内空格 → 取第一个数值 token；数值列中出现非数字字符串（列错位）→ 整行丢弃并记录日志（预期 <0.1%，在探查报告中核对）。
- 丢弃 `label`、`type`；`CMD` **不作为身份特征**（攻击工具进程名会成为标签捷径），只用于统计粗粒度量（如活跃进程数）。
- `_1/_2` 取并集，按整行去重。

#### 1.4 上下文时间桶（`context_builder.py`）
- 桶宽 Δ=5s。每桶聚合约 30 维：
  - process：行数、唯一 PID 数、CPU 求和/最大值、TRUN/TSLPI/TSLPU 求和、State（R/S/D/Z）占比、PRI 均值、NICE≠0 占比；
  - memory：MINFLT/MAJFLT/VSIZE/RSIZE/VGROW/RGROW 求和、MEM 最大值；
  - disk：RDDSK/WRDSK/WCANCL 求和、DSK 最大值。
- 由于 atop 采样稀疏（D4），追加**因果**多尺度统计：60s 滚动均值、300s EWMA → 拼成 `C ≈ 36` 维；另加 3 位来源观测掩码与“距上次观测秒数”（log）。
- 归一化：`log1p` + StandardScaler（仅 Train 时间段拟合）。
- **严格过去对齐**：对 `ts_end = t` 的流，取桶 `k = ⌊t/Δ⌋`，上下文序列为 `b_{k−T} … b_{k−1}`（不含尚未结束的当前桶）。等价于 `merge_asof(direction="backward", allow_exact_matches=False)`。

#### 1.5 实体流与窗口化（`windowing.py`）
- 实体 key：**host-pair（无序 `{src_ip, dst_ip}`）**为主；`dst_ip`（以受害者为中心）与 `global`（全局流）作为消融。
- 每个目标流 i 的输入 = 同一实体流中按 `ts_end` 排序的前 L 条流（含自身）；不足 L 条时左侧补零并加掩码。
- 只保存 `window_end_idx` 和 `stream_offset`（int64 memmap），在 `__getitem__` 中现取 → 内存 O(N) 而不是 O(N·L·F)。

#### 1.6 防泄漏时序切分（`splitter.py`）

**协议 P1（主协议）：片段内时序切分（Episode-Blocked Chronological Split）**
1. 按 1 分钟桶计算主导类别，合并相邻同类桶 → 得到攻击片段（episode）；正常时段（04-02~04 以及攻击间隙）同样作为片段。
2. 在**每个片段内部按时间顺序**、以累计流数切成 **Train 60% | Val 10% | Cal 10% | Test 20%**。
3. 每个切分边界设置 purge 带：丢弃窗口跨越边界的样本，并额外 embargo `max(L 条流的时间跨度, T·Δ = 60s)`，保证流量窗口和上下文窗口都不越界。
4. 顺序为 Val 在 Cal 之前：**让 Cal 在时间上紧邻 Test**，减小共形校准的分布漂移；Val 专用于早停、温度缩放和拟合 Jev，从而**模型选择、共形校准、二次裁决拟合三者的数据互不重叠**。

**协议 P2（压力测试）：全局时序切分** —— 04-02 ~ 04-27 12:00 为 Train/Val/Cal，之后为 Test。backdoor/ransomware/mitm 在训练中未出现 → 用于检验开放集场景下 ACI 空集/多类集合与 Jev 的“未知攻击”处置能力。论文中如实报告两种协议。

**采样**：训练集按类设上限（如每类 40 万窗口），采用**时间均匀抽样**（不是随机抽样，以保留演化过程）；mitm 全量保留；Val/Cal/Test 保持自然分布、不做下采样。

#### 1.7 Tensor 维度定义

| 名称 | 形状 | 说明 |
|---|---|---|
| `x_net_num` | `[B, L=16, 14]` float32 | 流数值特征 |
| `x_net_cat` | `[B, L=16, 23]` int64 | 流类别索引 |
| `m_net` | `[B, L]` bool | 补齐掩码 |
| `x_ctx` | `[B, T=12, 36]` float32 | 过去 60s 的上下文桶 |
| `m_ctx` | `[B, T, 3]` bool | 来源观测掩码 |
| `y_cls` / `y_bin` | `[B]` int64 / `[B]` float32 | 目标流标签 |
| `meta` | `ts_end, flow_id, episode_id, split` | 只用于评估和 ACI 排序，不进入模型 |

**DoD**：`test_no_leakage.py` 全部通过；产出切分统计表（每个切分×类别的样本数，确认 P1 下 mitm 在 Cal/Test 中均有样本）。

---

### Phase 2 — CCTB-Net 核心网络与多任务训练（4–5 天）

#### 2.1 张量流向

```
x_net_num [B,L,14] ─Linear→32 ┐
x_net_cat [B,L,23] ─Emb(dim_i)┴concat→LN→MLP→ H_net [B,L,64]                 (E_net，逐步编码)
x_ctx [B,T,36]+m_ctx ─Linear→32→2×因果Conv1d(k=3)→掩码注意力池化→ h_ctx [B,32]  (E_ctx)
h_ctx ─MLP→[Δγ; β] [B,128] ;  γ = 1 + tanh(Δγ)                                (FiLM，末层零初始化 ⇒ 初始为恒等映射)
H̃_net = γ ⊙ H_net + β  (在 L 维上广播)                                        [B,L,64]
H̃_net ─TCN(4 个残差块, 64 通道, k=3, 膨胀 1/2/4/8, 权重归一化, dropout 0.1)→ 取最后有效步 z [B,64]
       感受野 = 1 + 2·(k−1)·(1+2+4+8) = 61 ≥ L
Anomaly：u = g(z) ∈ R^32（无偏置），c = φ(h_ctx) ∈ R^32
         s_anom = ‖u − c‖²，  p_anom = σ((s_anom − τ)/κ)，τ、κ 可学习（κ 经 softplus 保证为正）
Threat ：logits = MLP([z ; h_ctx ; log(1+s_anom).detach()]) → [B,10]，P_cls = softmax(logits / T_temp)
```
- “相对于当前上下文正常模式的偏离” 通过**上下文条件化的正常中心 c(h_ctx)** 实现：同一种流量形态，在高负载上下文下离中心近、在空闲上下文下离中心远，与论文动机一一对应。
- 可选变体：在每个 TCN 块前都加 FiLM（multi-layer FiLM），作为结构消融。
- 参数量约 0.2–0.3M，GTX 1650 下 batch=1024 + AMP 显存充足。

#### 2.2 联合损失
$$
\mathcal{L}=\lambda_{cls}\mathcal{L}_{cls}+\lambda_{bin}\mathcal{L}_{bin}+\lambda_{sad}\mathcal{L}_{SAD}+\lambda_{cons}\mathcal{L}_{cons}
$$
- $\mathcal{L}_{cls}$：logit-adjusted CE（Menon et al., 2021，$\tau=1$），备选 focal（γ=2）+ effective-number 类别权重（β=0.999）。
- $\mathcal{L}_{bin}$：$p_{anom}$ 与 $y_{bin}$ 的 BCE，正常类加权（normal 是少数类）。
- $\mathcal{L}_{SAD}$（Deep SAD 形式）：$\frac{1}{|N|}\sum_{i\in N}s_i+\eta_a\frac{1}{|A|}\sum_{j\in A}\frac{1}{s_j+\epsilon}$。防坍缩：g 不使用偏置，依靠有标签的攻击样本把分数推开，同时监控 $\mathrm{Var}(u)$。
- $\mathcal{L}_{cons}$：$p_{anom}$ 与 $1-P_{cls}(\text{normal})$ 之间的对称 BCE，约束两个分支在二元层面一致。
- 默认 $(\lambda_{cls},\lambda_{bin},\lambda_{sad},\lambda_{cons})=(1.0,0.5,0.1,0.1)$；平衡策略：①在 Val 上以 Macro-F1 + AUROC 做网格搜索（主方案）；②同方差不确定性加权（Kendall et al., 2018）作为自动化方案，写进敏感性分析。

#### 2.3 类别不平衡（倒置失衡 + 极少数 mitm）
1. 按类时间均匀抽样上限（§1.6）；
2. 每个 epoch 使用平方根频率的 `WeightedRandomSampler`（有放回，epoch 规模约 150 万）；
3. logit adjustment / focal loss；
4. mitm（训练集约 630 条）：数值特征加高斯抖动（标准化空间中 σ=0.05）+ 随机截短窗口长度；**不对序列使用 SMOTE**；
5. 评估在自然分布上进行，逐类报告；mitm 单独给出 bootstrap 置信区间。

#### 2.4 训练设置
AdamW（lr=1e-3，wd=1e-4），余弦退火，最多 30 个 epoch，基于 Val Macro-F1 早停（patience=5），梯度裁剪 1.0，AMP，5 个随机种子。训练后在 Val 上做温度缩放（APS 依赖概率校准质量）。

**DoD**：P1 Val 上 Macro-F1 与 AUROC 稳定；`inference.py` 导出 Val/Cal/Test 的 `logits, s_anom, p_anom, ctx 摘要统计` 到 parquet，供 Phase 3 离线复用（决策层实验无需重新训练）。

---

### Phase 3 — ACI 动态校准与 Jev 二次决策流水线（3 天）

#### 3.1 非一致性分数
- 主方案：**APS（随机化版本）**；对照：LAC（$1-\hat p_y$）、RAPS。
- 模式：边际（marginal）与 **Mondrian（按类条件）**。由于 normal 和 mitm 是少数类，Mondrian 能保证逐类覆盖率；mitm 的 Cal 样本约 100 条，α=0.05 时满足 $n\ge 1/\alpha-1$。

#### 3.2 ACI 在线更新（Test 按 `ts_end` 顺序流式处理）
$$
C_t=\{y:\ S(x_t,y)\le \hat Q_{1-\alpha_t}(\mathcal{S}^{(W)}_{cal})\},\qquad
\alpha_{t+1}=\alpha_t+\eta\,(\alpha-\mathrm{err}_t),\ \ \mathrm{err}_t=\mathbb{1}[y_t\notin C_t]
$$
- 目标 α=0.05（另报 0.1）；η ∈ {0.001, 0.005, 0.01, 0.05}；DtACI（多 η 专家聚合）用作鲁棒性对照。
- $\alpha_t\le 0$ → 全集；$\alpha_t\ge 1$ → 空集（与标准 ACI 定义一致）。
- 校准缓冲：用 Cal 分数初始化，按标签反馈把 Test 样本分数滑入，窗口 W=50k。
- **标签反馈假设**：ACI 需要 $y_t$ 的事后反馈（分析员确认）→ 模拟延迟 D ∈ {0, 1k, 10k} 条流；另设“无反馈”对照（静态 CP + 周期性重校准）。
- 工程效率：Test 约 450 万条，用 numba 写逐样本循环；APS 分数位于 [0,1] → 用 1 万个分箱的直方图维护滑动分位数（O(1) 更新），代替排序。

#### 3.3 路由规则（`router.py`）
| 条件 | 含义 | 流转 |
|---|---|---|
| $\lvert C_t\rvert = 1$ | 高确定性 | 直接输出该类别 |
| $\lvert C_t\rvert \ge 2$ | 多类别模糊 | 进入二次裁决，标记 `AMBIGUOUS` |
| $\lvert C_t\rvert = 0$ | 证据不足 / 疑似新型 | 进入二次裁决，标记 `EMPTY` |
| 可选：$\lvert C_t\rvert=1$ 但 $p_{anom}$ 与所给类别矛盾 | 分支冲突 | 默认关闭，作为消融开关 |

#### 3.4 二次裁决接口
`SecondaryJudge.decide(evidence) -> Decision{label ∈ K ∪ {UNKNOWN_ATTACK}, action ∈ {ACCEPT, ALERT, DEFER}, posterior}`
证据向量：`P_cls, C_t, set_type, s_anom, z_anom, top-2 间隔, 上下文负载分层, proto/service 分组`。若日后拿到官方 Jev 访问权限，实现该接口即可直接替换。

> **关于 Jev 的说明**：我无法核实 TypeSafe AI Jev 的内部实现。方案 A 是**功能等效**的复现（结构化、可解释、低延迟的 System-One 式决策），并非还原其专有内部机制。论文中建议称为 “Jev-style structured decision module” 并引用原始来源。

#### 3.5 方案 A：贝叶斯网络结构化推理引擎（`jev_a.py`，基于 pgmpy）
- 离散证据节点：`AnomLevel`（按上下文分层的 z_anom 分位）、`CtxLoad`（低/中/高）、`SetType`（模糊/空集）、`TopCand`、`MarginBin`、`ServiceGroup` → 决策节点 `Y ∈ K ∪ {UNKNOWN}`。
- 结构：专家先验 DAG + 带约束的 Hill-Climb（BDeu 评分）；CPD 采用带 Dirichlet 平滑的 MLE，**只在 Val 上拟合**。
- 推理：变量消元 → 后验 → 在代价矩阵（$c_{FN}>c_{FP}$）下取期望效用最大的决策。

#### 3.6 方案 B：结构化二次裁决器（`jev_b.py`，主方案）
- 候选集：$\lvert C\rvert\ge 2$ 时 $C'=C$；空集时 $C'=K$；另始终加入 `UNKNOWN_ATTACK`。
- 上下文相对偏离：$z_{anom}=(s_{anom}-\mu_{norm}(\text{ctx}))/\sigma_{norm}(\text{ctx})$，其中 $\mu/\sigma$ 由 Train 正常样本按上下文负载分层估计。
- 候选打分：
$$
S(c)=w_0\log\tilde P_{cls}(c)+w_1\,\psi_{anom}(c)+w_2\log P(c\mid\text{ctx-stratum})+w_3\,r(c)
$$
  其中 $\psi_{anom}(\text{normal})=-z_{anom}$，$\psi_{anom}(\text{attack})=+z_{anom}$。
- 规则 $r(c)$：
  - R1：$z_{anom}<z_{low}$ 且 normal∈C → 提升 normal（压制高负载下的误报）；
  - R2：$z_{anom}>z_{high}$ 且 $C=\varnothing$ → 输出 `UNKNOWN_ATTACK` 告警；
  - R3：候选全部是攻击类 → 二元判定为攻击，类型取最大 $S(c)$；
  - R4：以代价敏感方式取期望代价最小的决策。
- 权重 w 通过在 Val 不确定样本上拟合多项逻辑回归得到（系数可解释）；$\max$ 后验 < θ 时输出 DEFER，拒判预算 ρ ≤ 5%。

#### 3.7 决策层评估口径
`UNKNOWN_ATTACK` 计为攻击；DEFER 两种口径都报：(a) 不计入 → 选择性风险 + 覆盖率；(b) 计为告警（保守口径）。

**DoD**：P1 Test 上滚动经验覆盖率稳定在 1−α 附近（±1%）；输出 Jev-A / Jev-B 对照表。

---

### Phase 4 — 基线对比与消融实验矩阵（5–7 天）

#### 4.1 基线（同一切分、同一特征、同一采样设置，5 个种子）
| 组别 | 模型 |
|---|---|
| 表格类（逐流） | XGBoost、Random Forest、MLP |
| 序列类 | LSTM、BiGRU、TCN、Transformer Encoder |
| 上下文融合 | Concat 早期融合、Late fusion（两路分别预测后融合 logits） |
| 无监督异常 | Deep SVDD、AutoEncoder（只用正常样本训练）、Isolation Forest |
| 不确定性 | 最大 softmax 阈值、静态 Split CP、MC Dropout / Deep Ensemble 熵 |

#### 4.2 消融矩阵
| ID | 变体 | 回答的问题 |
|---|---|---|
| A0 | **CCTB-Net 完整模型**（FiLM + TCN + 双任务 + ACI + Jev-B） | — |
| A1 | w/o Context & FiLM | 上下文是否有用 |
| A2 | Context 以 concat 代替 FiLM | 调制方式是否重要 |
| A3 | FiLM 只保留 γ / 只保留 β | 缩放与偏移各自的贡献 |
| A4 | 逐层 FiLM | 调制深度 |
| A5 | w/o Dual-Task（只保留 Threat） | 异常分支的价值 |
| A6 | 只保留 Anomaly 分支 | 分类分支的价值 |
| A7 | w/o $\mathcal{L}_{SAD}$ / w/o $\mathcal{L}_{cons}$ | 损失各项的贡献 |
| A8 | w/o ACI+Jev（argmax 直接输出） | 决策层整体价值 |
| A9 | 只用 ACI（不确定 → 直接拒判） | Jev 的增量价值 |
| A10 | 静态 CP + Jev | ACI 自适应性的价值 |
| A11 | Jev-A vs Jev-B | 二次裁决方案对比 |
| A12 | 边际 vs Mondrian；LAC / APS / RAPS | 共形设计选择 |
| A13 | L ∈ {8,16,32}，Δ ∈ {1,5,10}s，T ∈ {6,12,24} | 粒度敏感性 |
| A14 | 实体 key ∈ {global, dst_ip, host-pair} | 序列构造方式 |
| A15 | η ∈ {0.001,…,0.05}，标签延迟 D | ACI 鲁棒性 |
| C1 | **打乱上下文对照**（在时间上置换 x_ctx） | 证明模型确实利用了上下文语义，而非噪声增益 |
| C2 | **仅上下文模型** | 量化上下文作为“攻击时段捷径”的程度（必须远低于完整模型） |

P1 上运行全部实验；P2 上运行 A0/A1/A8/A9/A10/A11 与不确定性基线。

---

### Phase 5 — 评估指标与可视化（3 天）

#### 5.1 指标
- **检测**：FPR、FNR（二元，基于最终决策）、Macro-F1（10 类）、二元 F1、AUROC（基于 $s_{anom}$ 与 $1-P(\text{normal})$）、AUPRC（攻击为正类 + normal 为正类各报一次）、逐类 P/R/F1。
- **共形**：经验覆盖率（整体 / 逐类 / 1 万条滚动窗口）、平均集合大小、单元素率、空集率、按集合大小分层的覆盖率（SSC）、类条件覆盖率差距。
- **决策**：不确定样本比例、不确定子集上的 Jev 准确率、Jev 前后 FPR/FNR 对比、AURC。
- **核心论点指标**：**上下文漂移子集 FPR** —— 上下文负载处于 Test 前 10% 分位时段内的正常流的 FPR（直接对应“正常负载激增导致误报”）。
- **效率**：参数量、每 1k 条流的推理延迟、Jev 额外开销。
- **统计**：5 种子 mean±std；与最强基线做配对 bootstrap（1000 次）与 McNemar 检验。

#### 5.2 图表清单
| 编号 | 内容 |
|---|---|
| F1 | 数据：类别×日期热图 + Linux 覆盖时间线（说明为何采用 P1） |
| F2 | 架构总图（张量维度标注） |
| F3 | γ/β 均值随 CPU/内存负载变化的热图（FiLM 可解释性） |
| F4 | t-SNE：H_net 与 H̃_net 对比，以上下文负载着色（调制前后正常簇的变化） |
| F5 | 不同上下文分层下 s_anom 的分布（正常 vs 攻击） |
| F6 | α_t 轨迹与滚动覆盖率随时间变化（ACI vs 静态 CP） |
| F7 | 集合大小分布 + 逐类覆盖率柱状图 |
| F8 | 风险-覆盖率曲线（ACI+Jev vs MSP vs Ensemble） |
| F9 | Jev 前后混淆矩阵 |
| F10 | 上下文漂移子集 FPR 对比柱状图 |
| F11 | 消融雷达图 / 柱状图 |
| F12 | 超参数敏感性曲线（L、Δ、η、λ） |
| T1–T6 | 数据统计、主结果（P1）、P2 开放集、消融、共形指标、效率 |

---

## 3. 关键技术风险与应对预案（Risk & Mitigation）

| 风险 | 表现 | 应对 / 兜底 |
|---|---|---|
| R1 跨源对齐稀疏 | atop 每秒只有 1–3 行进程采样，单桶信息噪声大 | 多尺度因果滚动 + EWMA + 观测掩码 + 观测时距；前向填充上限 60s，超出则掩码；Δ 消融（A13） |
| R2 Linux 脏数据 | 单位后缀、列错位、空格值 | 健壮解析器 + 单元测试 + 丢弃率记入报告；丢弃率 >1% 时改为按列容错解析 |
| R3 攻击按天排期 | 全局时序切分导致类别缺失 | 主协议 P1 片段内切分；P2 作为开放集压力测试如实报告 |
| R4 捷径学习 | IP、端口、时间、进程名、上下文与攻击时段相关 | 丢弃标识符、端口分桶、`ts` 不作为特征、上下文不含 CMD 身份；C1/C2 对照实验 |
| R5 时间因果泄漏 | Zeek `ts` 是开始时间，特征要到结束才可得 | 统一以 `ts_end` 排序与对齐；上下文严格取过去的桶；purge/embargo；`test_no_leakage.py` |
| R6 倒置失衡 / mitm 过少 | normal 3.6%，mitm 1,052 条 | 时间均匀上限 + 平方根采样 + logit adjustment + 抖动增强；mitm 报告置信区间；Mondrian 校准样本不足时与全局分位数做收缩合并 |
| R7 ACI 窗口漂移 / α_t 失控 | 片段切换时覆盖率骤降、集合突然膨胀 | DtACI 多 η 聚合；滑动校准缓冲；在 err 序列上做 ADWIN 变点检测并重置 α_t；报告标签延迟敏感性 |
| R8 SAD 表示坍缩 | u 与 c 一起坍缩到零 | g 无偏置；依赖有标签攻击样本的推开项；监控 Var(u)，必要时加 c 的方差正则 |
| R9 算力与内存（16GB / 4GB） | 2200 万行无法整体载入 | parquet 列裁剪 + float32/int16 + 只存索引的窗口化 + memmap；AMP；推理结果落盘复用 |
| R10 Jev 无公开实现 | 无法安装 | 统一 `SecondaryJudge` 接口 + 方案 A/B 双实现，可插拔替换 |
| R11 同秒并发流顺序不确定 | 部分文件内部 `ts` 乱序 | 稳定排序 `(ts_end, file_idx, row_idx)` 写入 parquet 后固定，所有实验共享 |

---

## 4. 里程碑与时间估算

| 阶段 | 工期 | 交付物 |
|---|---|---|
| Phase 0 | 0.5 天 | `.venv`、`requirements.txt` |
| Phase 1 | 3–4 天 | parquet、上下文矩阵、P1/P2 切分、窗口索引、探查报告、泄漏测试 |
| Phase 2 | 4–5 天 | CCTB-Net 训练好的 checkpoint、Val 指标、导出的推理结果 |
| Phase 3 | 3 天 | ACI + Jev-A/B 决策流水线与决策层指标 |
| Phase 4 | 5–7 天 | 基线、消融、P2 结果 |
| Phase 5 | 3 天 | 全部表格与图表 |
