# 实验结果说明：单空间 ARIAD vs Dense vs Exact top-k

本文件夹里的图和表都可以直接放进论文。每张图都有 `.png` 和 `.pdf` 两个版本（插进 LaTeX 用 pdf）。每张表都有三个版本：`.csv` 是原始数值，`.md` 用来预览，`.tex` 是 booktabs 格式，可以直接 `\input`。

文件名规则：`<编号>_<内容>_<teacher>_k<k>_e<epochs>_seed<seed>`。例如 `sparse_k32_e500_seed0` 表示 sparse teacher、k=32、500 epoch、seed 0。

---

## 1. 实验设置（所有图表共用）

### 数据：簇结构合成数据
数据由 `ug_data_generate.py` 生成。

- **路由几何 G**：先在单位球上随机取 `N/16` 个簇中心，每个簇 16 个点，每个点是簇中心加上 jitter=0.10 的扰动，再做 L2 归一化。G 的维度 d_g=64。
- **内容 V**：独立的高斯向量，维度 d_v=64。
- **模型输入**：X = standardize([G | V])，维度 128。这里 rho_g=1，所以没有额外的几何噪声。
- **Teacher logits**：先算 GGᵀ，对全体元素做全局标准化，得到 S（对角线置为 −∞）。
- **Teacher 聚合**：用 **sparse teacher**（`teacher_dense=False`）。它只在每行 S 的 top-32 上做 softmax，其余权重为 0：
  A*ᵢ = softmax(S_{i,top-32} / τ)，Y = A*V，τ = 0.5。
- **N 的取值**：{1008, 2000, 5008, 10000, 20000}。cluster_size 固定为 16，因此原计划的 1000 和 5000 分别取成最接近的整簇数 1008（63 簇）和 5008（313 簇）。

各 N 下 teacher 的统计量如下。τ 作用在全局标准化之后的 logits 上，所以对余弦相似度而言，有效逆温度是 1/(στ)，它会随 N 略有变化。

| N | 簇数 | σ | 有效逆温度 1/(στ) | eff_support | recall_oracle | stability |
|---:|---:|---:|---:|---:|---:|---:|
| 1,008 | 63 | 0.1435 | 13.94 | 9.85 | 0.936 | 0.939 |
| 2,000 | 125 | 0.1349 | 14.82 | 9.64 | 0.936 | 0.927 |
| 5,008 | 313 | 0.1289 | 15.52 | 9.39 | 0.934 | 0.905 |
| 10,000 | 625 | 0.1271 | 15.74 | 9.43 | 0.930 | 0.881 |
| 20,000 | 1,250 | 0.1261 | 15.86 | 9.58 | 0.925 | 0.854 |

### 模型：单层、单空间 attention
- Z = X W_z，其中 d_z=64。Z 同时充当 query 和 key，打分为 z_i·z_j / √d_z。
- V̂ = X W_v。
- 三种方法的参数化完全相同，初始化也相同（同一个 seed）。

### 三条基线
| 方法 | 注意力范围 | 如何找邻居 |
|---|---|---|
| **Dense** | 对全部 j≠i 做 softmax | 不需要找邻居：一次 N×N 的 GEMM，前向和反向都经过 N² 个分数 |
| **Exact top-k** | 在当前 Z 的精确 top-32 上做稀疏 softmax | 每个 step 暴力计算 N² 个分数（无梯度，按 4096 行分块），取 top-32 |
| **ARIAD** | 在 ARIAD 图 G(i) 的 32 个邻居上做稀疏 softmax | 单图局部扩张：每行候选 = O(i) 16 + O(O(i)) 80 + I(i) 32，共 **C_scored = 144**（运行时实测）。精确重打分后取 top-32，打分时排除自己 |

Exact top-k 和 ARIAD 用的是**同一个**稀疏注意力模块，唯一的区别是邻居从哪里来。所以两者在成本和质量上的差异，完全来自近邻搜索。

### 训练
- 全批量训练，每个 epoch 就是一个 step，ARIAD 图每个 step 刷新一次。
- 优化器 Adam，lr = 1e-2，weight decay = 0，共 500 epoch。
- seed 0，**目前只跑了一个 seed**。
- 所有方法都用 k = 32。

### 指标定义
- **MSE**：第 500 个 epoch 的 mse(Ŷ, Y)。它在训练 step 之后用 eval 模式的前向计算。
- **Recall@32**（日志里叫 `teacher_match`）：mean_i |T₃₂(i) ∩ N₃₂(i)| / 32。
  - T₃₂(i) 是 teacher 的邻居集合，即 S 第 i 行的 top-32，恰好就是 sparse teacher 实际聚合的那 32 个节点。
  - N₃₂(i) 是学生的邻居集合（不含自己）：
    - Exact 取当前 Z 的精确 top-32。
    - ARIAD 取它实际参与注意力的图 G(i)。
    - Dense 取学到的 z_i·z_j 的 top-32。Dense 本身对所有节点做注意力，这里的 top-32 只代表它学到的几何会选出哪些邻居。

### 计算成本口径（每个训练 step，即一次前向加一次反向；评估用的前向不计入）
- **可微注意力评分 P_attn**：Dense 为 N²；Exact 和 ARIAD 都是 N·k。
- **无梯度搜索评分 P_search**：Dense 为 0；Exact 为 N²；ARIAD 为 N·C_scored（C_scored = 144，运行时实测）。
- **估算算术 FLOPs**：F = 3·2N·d_in(d_z+d_v) + 3·2·P_attn(d_z+d_v) + 2·P_search·d_z。
  - 一次乘加记为 2 FLOPs。
  - 可微部分把反向近似为前向的 2 倍，所以乘以 3；无梯度的搜索只乘以 1。
  - **不包括** softmax、top-k、排序/去重、gather/scatter，也不包括访存。
  - 这是一个成本模型，**不是硬件实测值**，不能用它推算加速倍数。

---

## 2. 图 1：N 与估算训练 FLOPs

**文件**：`fig1_flops_vs_n_sparse_k32_e500_seed0.pdf` / `.png`

**内容**：双对数坐标。横轴是 N，纵轴是每个训练 step 的估算算术 GFLOPs。三条线分别是 Dense（蓝，圆点）、Exact top-k（橙，方块）和 ARIAD（青，三角）。每条线末端标注了 N=20,000 时的数值，图底脚注给出拟合斜率和口径说明。

**关键数值**：
| N | Dense | Exact top-k | ARIAD |
|---:|---:|---:|---:|
| 1,008 | 0.879 | 0.254 | 0.142 |
| 2,000 | 3.27 | 0.758 | 0.283 |
| 5,008 | 19.8 | 3.83 | 0.708 |
| 10,000 | 77.8 | 14.0 | 1.41 |
| 20,000 | 309 | 53.7 | 2.83 |

- **双对数斜率**：Dense 1.96，Exact 1.80，ARIAD 1.00。Exact 的斜率低于 2，是因为 N 较小时它的线性项（N·k 次稀疏注意力和投影）占比还不小；N 越大，斜率越接近 2。
- **N=20,000 时**：ARIAD 的估算 FLOPs 是 Dense 的 0.92%，是 Exact 的 5.3%。
- **FLOPs 与 teacher 无关**：估算 FLOPs 只由 N、k 和 C_scored 决定，所以 dense teacher 下画出来的图完全一样。

**可用的图注**：
> Estimated arithmetic FLOPs per training step versus $N$ (log–log, $k=32$). Dense and exact top-$k$ scale quadratically in $N$ (Dense: $N^2$ differentiable scores; Exact: $N^2$ no-grad search scores plus $Nk$ differentiable scores), whereas ARIAD scales linearly ($Nk$ differentiable plus $N\cdot C_{\rm scored}$ search scores, $C_{\rm scored}=144$). FLOPs are from a cost model and exclude softmax, top-$k$, deduplication and memory traffic.

**写作注意**：
- 这张图说明的是**评分计算量**下降了，不代表端到端一定加速。
- 实测 wall-clock 取决于实现：Dense 用的是高度优化的 GEMM，而稀疏实现受索引和访存开销的限制。

---

## 3. 表 1：各 N 下的最终 MSE 和 Recall@32（sparse teacher）

**文件**：
- `table1_mse_recall_sparse_k32_e500_seed0-1-2.tex`：LaTeX 版，需要 `\usepackage{booktabs}`。
- `.md`：Markdown 版。
- `.csv`：每格的 mean、std 和 seed 数。
- `_per_seed.csv`：每个 seed 的原始值。

**多 seed**：共 3 个训练 seed（0、1、2），每格给出 均值 ± 样本标准差（ddof=1）。
- seed 只控制训练过程：权重初始化、ARIAD 的随机初始图和候选采样。**数据集不变**，仍是同一份（生成用的 seed 为 0）。
- 当标准差小于显示精度时，写成"± <0.01"或"± <0.0001"，而不是 0。

| N | MSE (×1e-4) Dense | MSE (×1e-4) Exact top-k | MSE (×1e-4) ARIAD | Recall@32 Dense | Recall@32 Exact top-k | Recall@32 ARIAD |
|---:|---:|---:|---:|---:|---:|---:|
| 1,008 | 4.49 ± 0.24 | 4.32 ± 0.07 | 4.36 ± 0.14 | 0.8155 ± 0.0010 | 0.8079 ± 0.0027 | 0.8031 ± 0.0074 |
| 2,000 | 5.44 ± 0.09 | 4.79 ± <0.01 | 4.79 ± <0.01 | 0.9295 ± 0.0002 | 0.9303 ± <0.0001 | 0.9291 ± <0.0001 |
| 5,008 | 4.15 ± 0.02 | 2.30 ± <0.01 | 2.31 ± <0.01 | 0.9554 ± 0.0001 | 0.9583 ± <0.0001 | 0.9503 ± 0.0001 |
| 10,000 | 6.70 ± <0.01 | 1.78 ± <0.01 | 1.82 ± <0.01 | 0.9629 ± <0.0001 | 0.9686 ± <0.0001 | 0.9463 ± 0.0003 |
| 20,000 | 11.85 ± <0.01 | 0.96 ± <0.01 | 1.11 ± <0.01 | 0.9680 ± <0.0001 | 0.9767 ± <0.0001 | 0.9276 ± 0.0002 |

**为什么 N ≥ 2,000 时标准差几乎为 0**：
- 我核实过 seed 确实生效：三个 seed 在 epoch 1 和 epoch 50 的 MSE 都不同。例如 N=2,000 时，Exact 在 epoch 50 的 MSE 分别是 0.0483、0.0536、0.0476。
- 但最终都收敛到同一个解。以 N=10,000 的 Exact 为例，三个 seed 的最终 MSE 分别是 1.77941e-4、1.77939e-4、1.77964e-4。
- 原因是模型只有两个线性投影，目标 Y 固定，这个问题在收敛后基本只有一个最优解，初始化只影响中间过程。
- 所以在这个任务里，"不同 seed 结果一致"说明的是结论**不依赖初始化**，而不是 seed 数不够。

**各 seed 分别对比 ARIAD 与 Exact**（从 `_per_seed.csv` 计算）：
| N | MSE_ARIAD / MSE_Exact（seed 0 / 1 / 2） | Recall_Exact − Recall_ARIAD（seed 0 / 1 / 2） |
|---:|---:|---:|
| 1,008 | 0.967 / 1.003 / 1.060 | −0.002 / 0.006 / 0.011 |
| 2,000 | 1.000 / 1.000 / 1.000 | 0.001 / 0.001 / 0.001 |
| 5,008 | 1.004 / 1.003 / 1.003 | 0.008 / 0.008 / 0.008 |
| 10,000 | 1.020 / 1.024 / 1.022 | 0.022 / 0.023 / 0.022 |
| 20,000 | 1.154 / 1.159 / 1.155 | 0.049 / 0.049 / 0.049 |

**主要结论**（在 3 个 seed 上都成立）：
1. **Dense 的 MSE 随 N 变差，稀疏方法随 N 变好。** sparse teacher 只聚合 top-32，而 Dense 学生会把权重分给其余 N−32 个节点，N 越大误差越大。Exact 和 ARIAD 的结构与 teacher 一致。到 N=20,000 时，Exact 的 MSE 比 Dense 低约 12 倍（0.96 对 11.85，单位 ×1e-4）。
2. **N=2,000 到 10,000 时，ARIAD 的 MSE 与 Exact 相差不到 2.5%。** N=1,008 这一行尚未收敛，不参与比较（见下方"收敛情况"）。N=20,000 时差距是 15–16%（1.11 对 0.96，单位 ×1e-4），三个 seed 都一样。sparse teacher 下已经没有截断误差，所以这部分差距可以全部归因于近邻搜索质量。
3. **ARIAD 与 Exact 的 Recall 差距随 N 扩大，而且在各 seed 之间几乎不变**：N=2,000 时为 0.001，5,008 时 0.008，10,000 时 0.022，20,000 时 0.049。候选宽度固定为 C_scored=144 时，N 越大搜索质量下降越多。这是固定预算设定的局限，应当如实报告；图 3 说明加大预算可以弥补这部分差距。
4. **Dense 的 Recall 反映的是它学到的几何**，不是它实际的注意力范围，因为它对全部节点做注意力。

**收敛情况（第 400 到 500 个 epoch 的变化）**：
- N=2,000 到 20,000：Dense 和 Exact 已完全收敛，变化小于 0.1%。ARIAD 的 MSE 也已收敛（变化 ≤ 0.7%），但 N ≥ 10,000 时 Recall 每 100 个 epoch 仍会增加约 0.003。
- **N=1,008 在所有 seed 下都还没有收敛**：最后 100 个 epoch 里，MSE 仍在下降 4–10%，Recall 仍在上升约 0.01，三种方法都是如此。这也是这一行标准差明显更大的原因：各 seed 停在收敛过程的不同位置。这一行三种方法之间的差距都在一个标准差以内，**不能据此比较方法优劣**，表注里要写明。

**可用的表注**：
> Final-epoch MSE ($\times 10^{-4}$) and Recall@32 against the teacher top-32 neighbor set, with a sparse top-32 teacher ($k=32$, 500 epochs). Mean $\pm$ sample std over 3 training seeds on the same dataset; "$<$" marks a std below display precision (the solution is essentially initialisation-independent at convergence). For Dense, Recall is computed on the top-32 of its learned scores. $N=1{,}008$ has not fully converged at 500 epochs; differences between methods in that row are within one std.

---

## 4. 图 2：训练动态——图的变化与 MSE 收敛（N = 10,000）

**文件**：`fig2_training_dynamics_N10000_sparse_k32_e500_seed0.pdf` / `.png`

**目的**：
- 说明 ARIAD 维护的是一张**随训练不断变化的图**，而不是最后碰巧得到了一张好图。
- 同时对比三种方法的 MSE 收敛速度。

**实验设置**：N=10,000，sparse top-32 teacher，k=32，500 个 epoch，seed 0，全批量训练（1 个 step 即 1 个 epoch），Adam，lr=1e-2，三种方法初始化相同。

**内容**：上下两栏，共用对数刻度的 epoch 横轴。
- **(a) 每个 epoch 的邻居替换率（%）。** 三条线分别对应三种方法，每条都取自该方法**自己那次训练**中的 32 邻居图：
  - 蓝线 Dense：用 Dense 学到的打分 z_i·z_j 取 top-32。Dense 本身对所有节点做注意力，并不使用这张图；这条线表示它学到的几何所隐含的近邻图变化有多快。
  - 橙线 Exact top-k：Exact 实际用来做注意力的精确 top-32 图。
  - 青线 ARIAD：ARIAD 维护的图，C_scored=144。
- **(b) MSE 收敛曲线**，纵轴为对数坐标。
  - 三条线分别是 Dense、Exact top-k、ARIAD，颜色和标记与图 1 一致。
  - 取点为 epoch 1 以及之后每隔 10 个 epoch 一个点。加入 epoch 1 是为了让曲线在对数横轴上从起点画起。
  - 线尾标注各方法第 500 个 epoch 的 MSE。

**数据来源**：两栏都来自同一组 seed-0 训练。
- (b) 的数据来自 `compare_dense_exact_ariad_N10000_sparseT32_k32_e500_seed0_log.csv`，即表 1 中 N=10,000 那一行对应的训练。
- (a) 的数据来自 `graph_tracking_N10000_sparseT32_k32_e500_seed0.csv`。这份数据是把 Dense、Exact 和 ARIAD 各重新训练一遍并加入审计得到的，训练过程与 (b) 中对应的曲线完全相同，最终 MSE 逐位一致（Dense 6.696e-4，Exact 1.779e-4，ARIAD 1.815e-4）。因此 (a) 中每条线描述的，就是 (b) 中同色曲线背后的那张图。

**精确审计**（用于 (a) 和下方的补充数值）：
- 每个 epoch 对全部 N 行做一次（不抽样）：用该步更新后的学生 Z_t 算出精确 top-32 图 E_t；在 ARIAD 训练中还同时记下 ARIAD 本步刷新后的图 G_t。
- 审计在计时区间之外，**不计入任何训练成本**。
- 由此计算：
  - **替换率**：1 − mean_i |S_t(i) ∩ S_{t−1}(i)| / 32，其中 S_t 是该方法自己的图：Dense 和 Exact 取 E_t，ARIAD 取 G_t。
  - **ARIAD 目标的替换率**（补充数值，未画进图）：在 ARIAD 训练中，对 ARIAD 自己学生的 E_t 计算同样的替换率。它衡量 ARIAD 追踪的那个目标每个 epoch 移动了多少。
  - **追踪 recall**：mean_i |G_t(i) ∩ E_t(i)| / 32，也就是 match_exhaustive，衡量 ARIAD 对当前学生精确图的覆盖。它比 teacher recall 更直接：teacher recall 还混入了学生的几何学得好不好，而追踪 recall 只反映搜索本身。
  - **精确图与其最终状态的重合**：mean_i |E_t(i) ∩ E_500(i)| / 32，每 10 个 epoch 取一次快照。
  - **track_recall_pre**：去掉一步滞后的追踪 recall，和上一步的 E_{t−1} 比较；G_t 正是用上一步的 Z 打分得到的。

**关键数值：(a) 中三条线的替换率，以及 (b) 中三条线的 MSE**
| epoch | 替换率 Dense | 替换率 Exact | 替换率 ARIAD | MSE Dense | MSE Exact | MSE ARIAD |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | — | — | — | 1.24e-1 | 2.27e-1 | 2.08e-1 |
| 2 | 13.8% | 16.1% | 57.9% | | | |
| 10 | 7.4% | 7.2% | 14.6% | 1.10e-1 | 1.27e-1 | 1.23e-1 |
| 20 | 5.9% | 4.3% | 5.5% | 1.08e-1 | 1.14e-1 | 1.12e-1 |
| 50 | 4.3% | 6.2% | 7.3% | 1.03e-1 | 1.02e-1 | 8.65e-2 |
| 100 | 2.6% | 2.3% | 1.9% | 3.43e-3 | 5.01e-3 | 2.82e-3 |
| 150 | 0.25% | 0.26% | 0.19% | 6.86e-4 | 2.26e-4 | 2.00e-4 |
| 200 | 0.02% | 0.02% | 0.02% | 6.70e-4 | 1.78e-4 | 1.83e-4 |
| 500 | 0.01% | 0.00% | 0.01% | 6.70e-4 | 1.78e-4 | 1.81e-4 |
| **累计**（逐 epoch 替换率之和） | 5.8 | 5.4 | 7.2 | | | |

**补充数值：ARIAD 对自身目标的追踪**（未画进图；来自 ARIAD 训练的审计，C_scored=144，括号内为 C_scored=256）
| epoch | 目标（ARIAD 学生的精确 top-32）替换率 | ARIAD 图替换率 | 目标与最终状态的重合 | 追踪 recall |
|---:|---:|---:|---:|---:|
| 1 | — | — | 0.128 | 0.013 (0.024) |
| 10 | 7.6% | 14.6% | 0.086 | 0.501 (0.741) |
| 20 | 4.0% | 5.5% | 0.074 | 0.754 (0.858) |
| 50 | 5.8% | 7.3% | 0.501 | 0.779 (0.827) |
| 100 | 1.9% | 1.9% | 0.905 | 0.891 (0.918) |
| 150 | 0.18% | 0.19% | 0.992 | 0.942 (0.966) |
| 200 | 0.01% | 0.02% | 0.999 | 0.949 (0.978) |
| 500 | 0.00% | 0.01% | 1 | **0.963 (0.995)** |

**MSE 收敛速度**（按 10 epoch 的采样统计）：
| 方法 | 首次 MSE ≤ 1e-2 | 首次 MSE ≤ 1e-3 | 首次进入自身最终值的 10% 以内 | 第 500 个 epoch 的 MSE |
|---|---:|---:|---:|---:|
| Dense | 90 | 120 | 140 | 6.70e-4 |
| Exact top-k | 100 | 130 | 160 | 1.78e-4 |
| ARIAD | 90 | 120 | 160 | 1.81e-4 |

**主要结论**：
1. **三种方法的近邻图在训练中都在大幅变化。**
   - 前 100 个 epoch，Dense 和 Exact 的 32 邻居图每个 epoch 替换 2–16% 的邻居。ARIAD 在最初几个 epoch 更高，见下一条；从 epoch 20 起，三者都在 2–8% 之间。三张图都要到 epoch 150–200 之后才基本不再变化。
   - 把逐 epoch 替换率累加：Dense 为 5.8，Exact 为 5.4，ARIAD 为 7.2。也就是说，每张图在训练中总共变化了 5–7 张完整图的量。同一个邻居可能被多次换入换出，所以这是总的变化量，不是不同邻居的个数。
   - 早期的图和最终的图差别很大：epoch 20 时，Exact 的图与它的最终状态只有 4% 重合，Dense 的隐含图为 18%，ARIAD 的目标为 7%。
   - 这说明"近邻图"并不是一个固定的对象：学生的表示在变，图也随之在变。
2. **ARIAD 的图与 Exact 的图变化速度相当。**
   - 从 epoch 20 左右起，ARIAD 与 Exact 的替换率曲线基本同步。两者都在训练中段出现一个小峰，峰值相近：ARIAD 在 epoch 47 达到 7.6%，Exact 在 epoch 55 达到 7.6%。Dense 的隐含图在 epoch 73 达到 6.0%。峰的位置不同，是因为几次训练的轨迹不完全一样；之后都降到接近 0。
   - 在此之前 ARIAD 的替换率明显更高（epoch 2 为 58%，Exact 为 16%），这是它离开随机初始图、追赶目标的阶段，也是 ARIAD 累计替换量（7.2）高于 Exact（5.4）的主要原因。
   - 在 ARIAD 自己的训练中，从 epoch 25 起，它的图替换率与其目标（ARIAD 学生的精确 top-32）的替换率几乎重合（见补充表）。这说明 ARIAD 每个 epoch 更新的邻居数量正好跟上目标移动的速度。
3. **ARIAD 跟踪的是当前的目标，而不是提前拿到了最终图**（补充数值，未画进图）。
   - epoch 20 时追踪 recall 已达 0.75（C=256 时为 0.86），而此时精确图与最终图只有 7% 重合，说明 ARIAD 覆盖的是**当时**的近邻。
   - 从 epoch 20 起，追踪 recall 始终不低于 0.74；最终为 0.963（C=256 时 0.995），即图 3 中的 match_exhaustive。
   - epoch 45 附近目标移动加快，精确图替换率出现一个 6.6% 的小峰，追踪 recall 随之从 0.816（epoch 29）降到 0.746（epoch 45），到 epoch 55 恢复原有水平。
   - 一步滞后的影响最多为 0.04（track_recall_pre 与 track_recall 之差），训练后期小于 1e-4。
4. **动态的图没有拖慢优化。** 虽然 ARIAD 的图从随机初始化开始，但它的 MSE 收敛不比 Exact 慢：
   - 达到 1e-2 和 1e-3 都比 Exact 早 10 个 epoch。
   - 在下降段（epoch 40–160），ARIAD 的 MSE 始终低于 Exact。
   - epoch 170 之后两者交叉，最终只差 2%（1.81e-4 对 1.78e-4）。
   - 三种方法都在 epoch 50–150 之间快速下降，epoch 200 之后进入平台，这与 (a) 中各图停止变化的时间吻合。
5. **Dense 的平台更高。** 它的最终 MSE 是 6.7e-4，约为稀疏方法的 3.7 倍。原因是 sparse teacher 只聚合 top-32，而 Dense 会把权重分给其余节点，存在结构性误差。这与表 1 的结论一致。

**可用的图注**：
> Training dynamics at $N=10{,}000$ (sparse top-32 teacher, $k=32$; log-scaled epochs; seed 0). (a) Per-epoch fraction of neighbors replaced in each method's own 32-neighbor graph during its own training run: the top-32 of Dense's learned scores (Dense itself attends to all nodes), the exact top-32 graph used by exact top-$k$, and the graph maintained by ARIAD ($C_{\rm scored}=144$); computed by an exact all-row audit that is not part of training cost. (b) MSE of the same three runs (epoch 1 and every 10 epochs). All neighbor graphs change substantially during the first ~150 epochs, and ARIAD's graph changes at the same rate as the exact one once it has left its random initialisation; nevertheless ARIAD's MSE converges as fast as exact top-$k$ and reaches the same plateau (within 2%), while Dense plateaus higher because the teacher aggregates only over its top-32 neighbors.

正文中可以用一句话补充追踪 recall，这些数值不在图里：
> Throughout training from epoch 20 on, ARIAD's graph covers at least 74% of the *current* exact top-32 neighbors (final 0.963 at $C_{\rm scored}=144$, 0.995 at $256$), although at epoch 20 the exact graph shares only 7% of its neighbors with its final state.

**写作注意**：
- 这是单个 seed 的结果。"ARIAD 在下降段略快于 Exact"和"epoch 45 附近追踪先降后恢复"这类细节，与这次训练的具体轨迹有关，换一个 seed 未必出现在同一位置。可以作为现象描述，但不要写成普遍规律。表 1 的多 seed 结果表明最终值与 seed 无关。
- 图中的"epoch"就是优化步数（全批量训练），不反映每一步的实际耗时。
- 审计是精确的（覆盖全部行、每个 epoch 都做），N=10,000 时代价很小。如果更大的 N 需要做审计，可以改成抽样部分 query，得到的是无偏估计。

---

## 5. 图 3：ARIAD 候选预算扫描（N = 10,000）

**文件**：`fig3_budget_sweep_N10000_sparse_k32_e500_seed0.pdf` / `.png`

**实验设置**：
- N 固定为 10,000，使用 sparse top-32 teacher，k=32，训练 500 个 epoch，seed 0。数据集与表 1、图 2 相同。
- 只改变 ARIAD 的候选预算，所有扩张来源按同一倍数 s 缩放：
  - c_o2 = 80s，c_i1 = 32s。
  - 反向缓冲宽度 k_rev = 32s，与 c_i1 同步缩放，否则 I(i) 的实际宽度 min(c_i1, k_rev) 会被截住。
  - O(i)，即当前 k=32 个邻居，**不参与缩放**。它是图本身，每轮都会无条件保留，不算作扩张预算。
  - c_oi 和 c_rand 在基准配置中本来就是 0，缩放后仍为 0。
- 由此 C_scored = 32 + 112s，运行时实测结果与公式一致。
- s 取 {0.125, 0.25, 0.5, 1, 2, 4}，其中 s=1 就是前面所有实验用的默认配置。
- s=1 这一档复现出的结果与表 1 中 N=10,000 的 ARIAD 完全一致（MSE 1.815e-4，Recall 0.9464）。
- Dense 和 Exact 不受这个预算影响，参考值取自同一数据集、同一 seed 的那次训练。

**内容**：左右两个子图。左图是最终 MSE，右图是最终 Recall@32；横轴都是 C_scored（对数坐标），刻度下方标注对应的倍数 s。
- 两图中的橙色虚线都是 Exact top-k。
- 右图的蓝色点线是 Dense。
- Dense 的最终 MSE 是 6.70e-4，远超出左图的坐标范围，因此只写在图底脚注里。

**数值**：
| s | c_o2 | c_i1 = k_rev | C_scored | 最终 MSE | 最终 Recall@32 | match_exhaustive | 估算 GFLOPs/step | ms/step | 搜索 ms/step | 峰值显存 MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.125 | 10 | 4 | 46 | 1.871e-4 | 0.9183 | 0.9286 | 1.288 | 5.27 | 2.56 | 354 |
| 0.25 | 20 | 8 | 60 | 1.866e-4 | 0.9209 | 0.9315 | 1.306 | 5.56 | 2.84 | 354 |
| 0.5 | 40 | 16 | 88 | 1.856e-4 | 0.9247 | 0.9360 | 1.341 | 6.05 | 3.34 | 354 |
| **1** | 80 | 32 | 144 | 1.815e-4 | 0.9464 | 0.9626 | 1.413 | 7.40 | 4.69 | 354 |
| 2 | 160 | 64 | 256 | 1.782e-4 | 0.9668 | 0.9954 | 1.556 | 9.23 | 6.50 | 587 |
| 4 | 320 | 128 | 480 | 1.780e-4 | 0.9685 | 0.9996 | 1.843 | 13.23 | 10.47 | 1,058 |
| *Exact top-k（参考）* | | | 10,000 | 1.779e-4 | 0.9686 | 1 | 14.03 | 7.18 | 4.48 | 354 |
| *Dense（参考）* | | | — | 6.70e-4 | 0.9628 | — | 77.78 | 15.46 | — | 1,659 |

**主要结论**：
1. **预算越大，ARIAD 越接近 Exact。** 当 s=4（C_scored=480）时，Recall 为 0.9685，Exact 是 0.9686；MSE 为 1.780e-4，Exact 是 1.779e-4，两者基本相同。此时 match_exhaustive 为 0.9996，说明 ARIAD 的图几乎就是精确 top-32。当 s=2（C_scored=256）时，Recall 与 Exact 只差 0.002，MSE 只差 0.2%。
2. **提升主要发生在 s=0.5 到 s=2 之间。** 在这一段 Recall 从 0.925 升到 0.967；s ≤ 0.5 时 Recall 基本停在 0.92 左右，s ≥ 2 后已经饱和。
3. **MSE 对预算不敏感。** 从最小预算到最大预算，MSE 总共只变化 5%（1.87e-4 到 1.78e-4）。左图纵轴范围很窄，看起来落差很大，实际并不大，写论文时要说明。即使预算最小（C_scored=46），ARIAD 的 MSE 也只比 Exact 高 5%，仍然比 Dense 低 3.6 倍。这说明漏掉的主要是注意力权重较小的边界邻居，对输出影响有限。
4. **成本随预算线性增加，但估算 FLOPs 一直远低于 Exact。** 在 s=2 时，ARIAD 的质量已与 Exact 持平，而估算 FLOPs 只有 Exact 的 11%（1.56 对 14.0 GFLOPs）。
5. **Wall-clock 与 FLOPs 的结论不同，需要如实写明。** 在 N=10,000 时，当前实现下 ARIAD 只有在 s ≤ 0.5 时才比 Exact 快。s=2 时是 9.2 ms，比 Exact 的 7.2 ms 还慢。原因是 Exact 的暴力搜索只是一个 GEMM 加一次 top-k，而 ARIAD 的开销主要在 gather 和去重上。另外，预算越大，候选 gather 产生的中间张量越大（形状为 [4096, C, 64]），所以 s ≥ 2 时峰值显存会上升。

**可用的图注**：
> ARIAD candidate-budget sweep at $N=10{,}000$ (sparse top-32 teacher, $k=32$, 500 epochs). Every expansion source is scaled by the same factor $s$ ($c_{o2}=80s$, $c_{i1}=k_{\rm rev}=32s$; the current $k$ neighbors are always kept), giving $C_{\rm scored}=32+112s$. Left: final MSE; right: final Recall@32. Dashed: exact top-$k$; dotted: Dense. With $C_{\rm scored}\ge 256$ ARIAD matches exact top-$k$ in both MSE and recall, while scoring about 2.6% as many candidates per row.

**写作注意**：
- 图注里的"2.6%"是按每行的搜索评分数算的，即 256/10,000。
- 这是单个 seed 的结果。s ≤ 0.5 那几档的 MSE 差距只有 0.1–0.3%，很可能在不同 seed 之间的波动范围内。
- 表 1 中观察到大 N 下 Recall 下降，这组实验说明，适当加大预算就能弥补这部分损失。

---

## 6. 已知局限和待补的实验
- **多 seed 只覆盖表 1**：表 1 用了 3 个 seed，图 1–3 仍然只用 seed 0。图 1 是确定的成本模型，与 seed 无关；图 2、图 3 是单次训练的结果。鉴于表 1 显示各 seed 收敛到几乎相同的解，图 2、图 3 的最终值应当有代表性，但训练过程中的细节（如图 2 对应审计中 epoch 45 附近的追踪下降）仍可能随 seed 不同而变化。
- **计时和显存还没做成图表**：已有数据（每个 N 的 `step_ms_median`、`search_ms_median`、`peak_mem_mb`），但单次运行的计时波动有 5–30%，建议用多 seed 的中位数。另外，这里的 Dense 是朴素实现，会把整个 N×N 矩阵存下来；如果要比较 wall-clock，应当补一个 SDPA/FlashAttention 版本的 Dense 基线。
- **ARIAD 的候选预算是固定的**（C_scored=144）。大 N 下 Recall 会下降。图 3 已经说明，在 N=10,000 时，把预算加到 C_scored=256 就能追平 Exact。还缺的是在 N=20,000 上做同样的扫描，看需要的预算是否随 N 增大。
- **dense teacher 的结果**：同样的实验在 dense teacher 下也跑过（数据文件带 `_k32_e500_seed0`，不带 `sparseT32`）。把 `--teacher dense` 传给下面的脚本就能生成对应的图表。

---

## 7. 复现方法

在 `ariad_single_space/` 目录下运行：

```bash
# 1) 训练并记录（每个 N 一次；数据集不存在时会自动生成）
for c in 63 125 313 625 1250; do
  python compare_dense_exact_ariad.py --n-clusters $c --epochs 500 --teacher-dense 0 --seed 0
done
# 1b) ARIAD 候选预算扫描（N=10000，图 3 用）
python budget_sweep.py --n-clusters 625 --epochs 500 --teacher-dense 0 --seed 0
# 1c) 训练中的精确图审计（N=10000，C_scored=144 与 256，图 2(a) 及追踪 recall 数值用）
python graph_tracking.py --n-clusters 625 --epochs 500 --teacher-dense 0 --seed 0
# 2) 画图和生成表格，输出到 result/
python plot_results.py --teacher sparse              # 图 1–3（默认 --fig all；图 2/3 默认 --n 10000）
python make_tables.py  --teacher sparse --seeds 0 1 2   # 表 1（多 seed；需先对每个 seed 跑步骤 1，加 --seed 1 / --seed 2）
```

**原始数据**（位于上一级目录 `ariad_single_space/`）：
- `compare_dense_exact_ariad_N<N>_sparseT32_k32_e500_seed0_log.csv`：逐 epoch 记录 mse、teacher_match、match_exhaustive、各类评分数、est_flops、step/search 耗时和峰值显存。
- `compare_dense_exact_ariad_N<N>_sparseT32_k32_e500_seed0_summary.csv`：每种方法的最终指标和成本汇总。
- `budget_sweep_N10000_sparseT32_k32_e500_seed0_{log,summary}.csv`：图 3 所用的预算扫描数据。log 记录每个 epoch，summary 是汇总，并包含 scale、c_o2、c_i1、k_rev 等列。
- `graph_tracking_N10000_sparseT32_k32_e500_seed0.csv`：图 2(a) 和第 4 节追踪 recall 数值所用的逐 epoch 审计数据，包含 4 次训练：ARIAD（scale=1、2）、Dense、Exact。列包括 method、scale、c_scored、epoch、mse、exact_churn（该次训练学生的精确 top-32 图的替换率）、teacher_recall、track_recall、track_recall_pre、ariad_churn、exact_vs_final。Dense 和 Exact 那两次训练中，只属于 ARIAD 的列为空。
- `sweep_logs_sparseT32_e500/`：每个 N 的完整运行输出，以及预算扫描和图审计的输出（`budget_sweep_N10000.log`、`graph_tracking_N10000.log`）。

**相关代码**：
- `ariad_single.py`：ARIAD 单图 seeker。
- `ariad_train_single.py`：模型定义和 `ARIAD_CONFIG`，其中 k=32、c_o2=80、c_i1=32、c_oi=0、k_rev=32。
- `ug_data_generate.py`：数据生成，`teacher_dense` 旋钮在这里。
- `compare_dense_exact_ariad.py`：三方对比和成本统计。
