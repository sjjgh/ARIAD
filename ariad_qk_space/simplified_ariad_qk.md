# 双空间 Ariad（QK 版）：三张图 + 局部扩张

本文档描述 `ariad_qk_space` 目录下**当前实际运行**的双空间 ARIAD 算法。代码在：
- `ariad_qk.py`：`AriadSeekerQK`，负责 QK 图；
- `ariad_single.py`：`AriadSeekerSingle`，负责 QQ 图和 KK 图，是单空间简化版的一份拷贝；
- `ariad_train_qk.py`：模型 `OneLayerQKAttention` 和默认配置；
- `scaling_sweep.py`：论文双空间实验实际使用的配置覆盖。

单空间版本见 `../ariad_single_space/simplified_ariad.md`。本文沿用那里的记号：
- `O(i)`：节点 i 在某张图里的出邻居（它当前的邻居列表）；
- `I(i)`：入邻居（谁的邻居列表里有 i），通过反向缓冲取得。

## 1. 从单空间到双空间

单空间里 query 和 key 是同一个向量 `z_i`，只需维护一张 `z` 空间的近邻图，注意力直接用它。

双空间里 `q_i = x_i W_q`、`k_j = x_j W_k` 是两个**独立**的投影，注意力要找的是
"与 query `q_i` 打分最高的 32 个 key"，也就是 **QK 近邻**。这张图不能从单个空间的近邻关系直接得到：
`q_i` 的近邻 query、`k_j` 的近邻 key，与"`q_i` 的高分 key"是三种不同的关系。

因此算法同时维护三张图，每张都用"保留当前邻居 + 局部扩张 + 精确重打分取 top-k"的方式逐步刷新：

| 图 | 形状 | 含义 | 维护者 |
|---|---|---|---|
| QQ | [N, 64] | query 空间的近邻：`q_i · q_j` 高 | `AriadSeekerSingle`（输入 Q） |
| KK | [N, 64] | key 空间的近邻：`k_i · k_j` 高 | `AriadSeekerSingle`（输入 K） |
| QK | [N, 32] | query i 当前关注的 key 集合 `A(i)`；注意力**实际使用**的就是这张图 | `AriadSeekerQK` |

QQ 和 KK 本身不参与注意力计算，它们只是为 QK 图提供候选的"辅助结构"。

这里是在同一组 N 个节点上做自注意力：节点 i 同时是一个 query 和一个 key。所以"自身"指 key id 等于
query id 的情况，三张图都要排除（QQ/KK 的现状见第 10 节）。

## 2. 算法状态

- `QQ`、`KK`、`QK` 三张 `LongTensor` 图，都从**均匀随机**的 id 初始化，不使用 teacher 或精确近邻。
- QQ 和 KK 的内部状态与单空间版本完全相同，没有副本、reset 或 elite memory carry。
- 不维护任何额外状态。反向缓冲每步从当前图重新构建。

## 3. 每个训练 step 的刷新顺序

`OneLayerQKAttention.forward`（训练模式）中，每步按下面的顺序执行。Q 和 K 都已经 detach，刷新过程不传梯度。

```text
q = X W_q,  k = X W_k,  v = X W_v

1. QQ.refresh(q)                 # 单空间算法，输入 q
2. KK.refresh(k)                 # 单空间算法，输入 k
3. QK.refresh(q, k, KK, QQ)      # 使用第 1、2 步刚刷新好的 QQ/KK，以及上一步的 QK
4. 注意力在 QK 上做稀疏 softmax
```

- 第 1、2 步各自独立，就是单空间 ARIAD 的 `refresh`，只是一个吃 Q、一个吃 K，见单空间文档 4 节。
- 第 3 步中，qF 和 qR 两路候选读的是**上一步**的 QK 图。整个 refresh 期间 `self.QK` 保持不变，全部行算完后才一次性替换，所以分块处理的顺序不会影响结果。
- 第一次调用时，三张图先随机初始化，再立即各刷新一次，不会把未经打分的随机图交给注意力使用。
- eval 模式下三张图都**不刷新**，直接复用训练前向刷新后的图。评估时存在一步滞后：图是用更新前的 q、k 打分的，而评估用的是更新后的 q、k。

## 4. QK 图的候选来源

对每个 query i，候选池由下面几类拼接而成。所有候选都是 **key id**，打分统一为 `q_i · k_t`。

| 来源 | 记号 | 路径 | 采样方式 | 预算 |
|---|---|---|---|---:|
| 当前 key 集合 | `A(i) = O_QK(i)` | — | 全部保留 | `k_final` |
| key 侧前向 | `kF(i) = O_KK(A(i))` | 我当前的 key a → a 在 KK 图里的邻居 | 从 `A(i)` 随机取锚点 a，再从 `O_KK(a)` 随机取一个 | `b_kF` |
| query 侧前向 | `qF(i) = O_QK(O_QQ(i))` | 与我相似的 query j → j 当前关注的 key | 从 `O_QQ(i)` 随机取 j，再从 `QK_prev(j)` 随机取一个 | `b_qF` |
| key 侧反向 | `kR(i) = I_KK(A(i))` | 我当前的 key a → KK 图里"把 a 当邻居"的 key | 从 `A(i)` 随机取 a，再从 a 的 KK 反向缓冲随机取一个槽位 | `b_kR` |
| query 侧反向 | `qR(i) = O_QK(I_QQ(i))` | QQ 图里"把我当邻居"的 query j → j 的 key | 从 i 的 QQ 反向缓冲随机取 j，再从 `QK_prev(j)` 随机取一个 | `b_qR`（**默认 0，关闭**） |

四类扩张候选都由单空间的同一个原语 `AriadSeekerSingle._sample_forward_paths(anchors, graph, c)` 生成，语义是"随机锚点、随机一跳"，与单空间里 `O(O(i))` 的构造完全一致。

**直觉**：
- **kF**：如果 a 是 q_i 的高分 key，那么和 a 在 key 空间里相近的 key，大概率也是 q_i 的高分 key。
- **qF**：如果 j 和 i 在 query 空间里相近，那么 j 找到的高分 key，大概率也适合 i。这相当于在相似的 query 之间共享搜索成果。
- **kR / qR**：分别是 kF 和 qF 的反向版本。

**类型约束：为什么不用 `I_QK`。** `I_QK(k)` 表示"当前关注 key k 的那些 query"，返回的是 **query id**，不能直接作为 key 候选。类型上正确的三跳路径 `O_QK(I_QK(A(i)))` 理论上存在，但本版本没有实现。

**qR 默认关闭。** 它被保留为诊断开关，`qk_candidate_path_ablation.py` 是对应的只读诊断（与单空间附录 A 同一思路），但目前还没有成文的消融结论。

## 5. 反向缓冲

kR 和 qR 需要 KK 图和 QQ 图的反向索引，由 `build_reverse_with_mask`（`ariad_qk.py`）每步从当前图重新构建：
- 构建方式与单空间 `_build_reverse` 相同：真实入边优先填入，入度不足 `k_rev` 的行用随机 id 补齐（并避开自身）；
- 额外返回一个布尔掩码，标出每个槽位是真实入边还是随机填充。

这个掩码用于诊断（写入 `last_refresh_stats["kk_rev_real_frac"]`）。它能区分"通过真实反向边找到的候选"和"通过随机填充找到的候选"，否则 kR 的效果里可能混入了隐藏的随机探索。**刷新本身不使用这个掩码**，填充槽位照样参与采样，所以 kR 实际包含一部分随机探索。

反向缓冲只在 `b_kR > 0`（KK）或 `b_qR > 0`（QQ）时才构建。QQ/KK 图各自内部的 `I(i)` 候选，使用它们自己在单空间算法中构建的反向缓冲，与这里的是两份独立的缓冲。

## 6. QK 打分与选择

对拼接后的候选池（每行宽度 `C_QK = k_final + b_kF + b_qF + b_kR + b_qR`）：

1. 精确打分：`score(i, t) = q_i · k_t`（不除以 √d，不影响排序）；
2. 行内去重：同一 key id 只保留一份，其余置 −∞；
3. 排除自身：`t == i` 置 −∞；
4. 取 top `k_final` 作为新的 `QK(i)`。如果有效候选不足 `k_final` 个，用候选池前几列补齐（默认预算下不会发生）。

与单空间一样，**当前 key 集合 `A(i)` 无条件留在候选池中**。所以，一个在当前打分下仍属于候选池 top-k 的 key，不会被这一步挤掉（单空间文档 4.2 节的单调性论证在这里同样成立）。

## 7. 注意力

```text
s_ij   = q_i · k_j / sqrt(d)     for j in QK(i),   s_ii = -inf
a_i    = softmax_j(s_ij)
y_hat_i = Σ_{j ∈ QK(i)} a_ij v_j
```

梯度只经过这 N × k_final 个分数和对应的 value；三张图的刷新都不传梯度。

## 8. 伪代码

```text
init（第一次 forward）:
    QQ <- random(N, k_QQ);  KK <- random(N, k_KK);  QK <- random(N, k_final)

forward(X), 训练模式:
    q, k, v = X W_q, X W_k, X W_v
    Zq, Zk  = detach(q), detach(k)

    QQ <- single_space_refresh(QQ, Zq)          # O + O(O) + I，精确重打分 top-k_QQ
    KK <- single_space_refresh(KK, Zk)

    KKrev <- build_reverse(KK, k_rev)           # 仅当 b_kR > 0
    QQrev <- build_reverse(QQ, k_rev)           # 仅当 b_qR > 0
    QKprev <- QK
    for each query i:
        A    = QKprev[i]
        kF   = sample_forward(A,          KK,     b_kF)
        qF   = sample_forward(QQ[i],      QKprev, b_qF)
        kR   = sample_forward(A,          KKrev,  b_kR)
        qR   = sample_forward(QQrev[i],   QKprev, b_qR)
        cand = concat(A, kF, qF, kR, qR)
        score = Zq[i] . Zk[cand];  mask duplicates and cand == i
        QK[i] = cand[topk(score, k_final)]

    attention over QK (7 节)

forward(X), eval 模式: 不刷新，直接用现有 QK 做注意力
```

## 9. 当前配置与每行打分候选数

默认值见 `ariad_train_qk.py`；论文双空间实验通过 `scaling_sweep.py` 的 `configure()` 设置 `k_final=32`，其余图参数保持默认：

| 图 | 参数 | 每行打分候选数 |
|---|---|---:|
| QQ | `k=64, k_rev=32, c_o2=80, c_i1=32, c_oi=0, c_rand=0` | 64 + 80 + 32 = **176** |
| KK | 同上 | **176** |
| QK | `k_final=32, k_rev=32, b_kF=32, b_qF=32, b_kR=32, b_qR=0` | 32 + 32 + 32 + 32 = **128** |
| 合计 | | **C = 480** |

- **按配置数出来，而不是运行时测得。** 这个 C 包含去重和自身屏蔽之前的重复项，因为这些候选确实被打过分。双空间代码目前**没有** `n_scored` 计数（见第 10 节），所以不像单空间那样有实测值。
- **与单空间的比较。** 单空间的 C_scored = 144。双空间每行要维护三张图，打分量约为单空间的 3.3 倍，但仍然是 O(N·C)，与 N 呈线性关系。
- **QQ/KK 图比 QK 图宽**（64 对 32）。这个宽度是直接沿用单空间当时的配置（`k=64`），没有针对双空间单独调过。
- **成本模型**（`result/README.md`）：F = 6·N·d_in·(2·d_z + d_v) + 6·P_attn·(d_z + d_v) + 2·P_search·d_z。ARIAD 的 P_search = N·C。

论文双空间实验中与算法无关的设置（数据、teacher 缩放、τ=0.01、`value_mode="fixed_v"`、恒定学习率等），以及 Dense 和 Exact top-k 基线，见 `result/README.md`。

## 10. 已知的实现细节与待同步项

1. **QQ/KK 图没有排除自身。** `ariad_qk_space/ariad_single.py` 是单空间代码的一份拷贝，拷贝时间早于 2026-09-26 单空间版加入的"打分时排除自环"修改（见单空间文档 9 节）。因此 QQ/KK 图的某一行可能包含节点自己，而 `q_i·q_i`、`k_i·k_i` 通常很高，这种情况很可能出现。
   - **QK 图不受影响**：`AriadSeekerQK` 自己会排除 `t == i`，注意力也会屏蔽自身，所以结果正确。
   - **影响在于浪费预算**：如果 `O_QQ(i)` 包含 i，qF 会借用 i 自己上一步的 key，这些都与 `A(i)` 重复；如果 `O_KK(a)` 包含 a，kF 会把 a 重新提出来一次。两种情况都浪费候选槽位，并让 QQ/KK 图的有效宽度少 1。
   - **建议**：把单空间的修改同步过来。这会改变双空间的训练结果，需要重跑 `scaling_sweep.py`。
2. **没有 `n_scored` 计数。** 同样是因为拷贝较早，第 9 节的 C=480 只能按配置推算，无法在运行时实测。
3. **三个空间共用一个打分分块参数**（`score_chunk=4096`），这只影响显存，不影响结果。
4. **图宽度与评估宽度的关系。** 注意力和评估都用 QK 图的 `k_final=32`，与 teacher 的 top-32 一致。QQ/KK 的宽度（64）只影响候选质量，不直接出现在任何指标中。

## 11. 与原始 Ariad v0（`phase1/ariad_v0.md`）的区别

v0 是最早的完整设计，包括：KK/QK 两侧各有多份副本（`kk_M`、`qk_M`）；周期性异步重置（`kk_R`、`qk_R`）和 age ensemble；把 KK 副本合并成 KK main；以及 final_expand 等步骤。

当前双空间版本在单空间简化的基础上做了以下改变：
- **去掉副本、重置、age ensemble 和最终合并。** 每个空间只保留一张图，理由与单空间文档第 1 节相同：没有 reset，也就不需要额外的安全网。
- **增加一张独立的 QQ 图。** 它为 QK 提供 query 侧的候选（qF/qR），v0 里没有这一路。
- **QK 图直接就是注意力使用的邻居集合**，不再有"局部刷新图"和"最终图"两层之分。
- **Q 和 K 默认使用独立的投影**（`bond_QK=False`）。原始 `phase1/ariad_train.py` 共享 `W_kq`，因此 q == k，那时 QQ 图和 KK 图实际上是同一张图。

单空间版中"去掉 reset 不影响收敛"是经验观察，不是证明（单空间文档第 7 节）；双空间版同样如此，适用范围仅限于这组合成数据。
