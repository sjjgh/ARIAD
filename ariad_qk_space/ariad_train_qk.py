from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ug_data_generate as datagen
from ariad_single import AriadSeekerSingle, AriadSingleConfig
from ariad_qk import AriadQKConfig, AriadSeekerQK

# =========================
# User Knobs (edit here)
# =========================
DATA_CONFIG = {
    "dataset_root": Path(__file__).resolve().parent / "synthetic_tau_datasets",
    # 数据目录名由 GEN_CONFIG 自动拼出（dataset_name()），改任何生成旋钮都会落到新目录，
    # 不会误读旧数据。
}

# 自动生成数据的旋钮：dataset_root/dataset_name(GEN_CONFIG) 下没有数据时，自动用
# ug_data_generate.py 生成。生成器是 O(N^2) 内存/计算（N=20000 约需 10GB+，40000 >30GB）。
#
# 这一组默认值是把玩具脚本 qk_independent_dense_probe.py 里发现的问题修进正式管线后的
# 配置（学生仍然从 X=[G_noise|V] 学 Wq/Wk/Wv，teacher 仍然是 G 的相似度）：
#   1. teacher_scale="fixed"：teacher/学生同一个 1/(tau*sqrt(d)) 缩放，不做全局 μ/σ 标准化。
#   2. cluster_size(10) <= 聚合宽度 k(32)：否则 top-k 质量有结构性上限（簇内伙伴不够 k 个）。
#   3. tau=0.01 + cluster_jitter=0.05：固定缩放下 G 是单位向量，点积很小（std~0.14），需要
#      小 tau 才能让 teacher 尖锐（eff_support≈10，top-k 质量≈1），跑之前看数据生成时的输出。
#   4. teacher 是对称的（G·Gᵀ），所以 tied(bond_QK) 和 untied 面对的都是双方可表示的 teacher。
# 要回到论文单空间的设定：teacher_scale="global_standardize", tau=0.5, cluster_size=16,
# k_eval=16, teacher_dense=False, teacher_topk=32（见 ariad_single_space）。
GEN_CONFIG = {
    "n_clusters": 200,
    "cluster_size": 10,   # N = 2000
    "d_g": 64,
    "d_v": 64,
    "cluster_jitter": 0.05,
    "k_eval": 32,         # 需与 QK_CONFIG["k_final"] 对齐（recall/match 在这个宽度上比）
    "k_obs": 16,
    "rho_g": 1.0,
    "seed": 0,
    "block_size": 512,
    "attention_diag_topk": 32,
    # True: teacher 是全局 dense attention，A* = softmax_{j!=i}(S/tau)（Y = A* @ V 对全部 N 个 key 聚合）。
    # False: teacher 只在 S 的 top-teacher_topk 上做稀疏聚合，其余 A*_ij = 0，
    #        即 teacher 自己也是 top-k 稀疏聚合。
    "teacher_dense": True,
    "teacher_topk": 32,   # 仅 teacher_dense=False 时生效；跟 QK_CONFIG["k_final"] / k_eval 对齐
    "tau": 0.01,
    "teacher_scale": "fixed",   # "fixed" | "global_standardize"，见 ug_data_generate.GEN_CONFIG
}


def dataset_name(cfg: Dict = None) -> str:
    """Directory name encoding every generation knob, so a config change can never silently
    reuse a stale dataset."""
    c = GEN_CONFIG if cfg is None else cfg
    n = c["n_clusters"] * c["cluster_size"]
    name = (f"{c['teacher_scale']}_N{n}_cs{c['cluster_size']}_tau{c['tau']:g}_j{c['cluster_jitter']:g}"
            f"_ke{c['k_eval']}_seed{c['seed']}")
    return name if c["teacher_dense"] else f"{name}_sparseT{c['teacher_topk']}"


TRAIN_CONFIG = {
    "seed": 0,
    "epochs": 2000,
    "lr": 1e-2,
    "weight_decay": 0.0,
    "print_every": 200,
    # 余弦衰减到 lr*floor（跟玩具脚本同一个调度，同一 lr/epochs 才能对表）。
    "cosine_decay": True,
    "cosine_floor": 0.01,
}

EVAL_CONFIG = {
    "exhaustive_sample_size": 1000,
}

MODEL_CONFIG = {
    "use_oracle_init": False,
    # True: 强制 W_k 跟 W_q 是同一份参数（真正的权重绑定，不是只在初始化时让它们
    # 相等——两个分开的 Parameter 就算初始值一样，各自的梯度也会让它们分开漂移）。
    # 绑定后数值上 q==k，等价于退回单空间那种"Q=K共享"的设定（跟原版
    # phase1/ariad_train.py 的 w_kq 一致），用来验证之前的不收敛是不是分开W_q/W_k
    # 本身导致的可辨识性问题。数据生成那边本来就没有 Q/K 概念，不受这个旋钮影响。
    "bond_QK": False,
    # True: W_k 直接设成 oracle 值（识别出 x 里 G_noise 那部分的 identity 投影，
    # 跟 maybe_oracle_init 给 w_q/w_k 用的是同一个构造）并冻结，不参与训练，只训
    # W_q。去掉 W_q/W_k 的双线性联合优化，只保留"K已知正确、只学Q"这个更简单的
    # 子问题，用来在这个数据管线（有正确的全局标准化）下重新检验分开是否收敛。
    # 要求 bond_QK=False（绑定时只有一个 w_q，没有独立的 w_k 可冻结）。
    "freeze_wk_to_oracle": False,
    # "learned": 学生自己学 W_v（v = x @ W_v，论文单空间的设定）。
    # "fixed_v": W_v 固定成 x 里 V 那块的恒等映射（values = 原始 V，跟玩具脚本一致）。
    # 诊断发现（N=2000）：W_v 可学时 untied 的 Wq/Wk 学不出路由——注意力散在 ~200 个 key 上，
    # 靠大量 key 的加权组合去回归每个 query 的目标（簇内质量 0.004=随机），mse 却能降到 0.019；
    # W_v 固定成 V 后 untied 立刻学会（簇内质量 0.98）。tied 不受影响（对称性自带簇内偏置）。
    "value_mode": "learned",
}

# Exact QK 对照旋钮：开启后完全不走 Ariad（qq_seeker/kk_seeker/qk_seeker 都会被
# 构造但永远不会被调用），每个 epoch 直接用当前 q,k 做一次精确（brute-force）
# top-k 查找，找到的这 k 个 key 走和 Ariad 路径完全一样的聚合。跟
# ../ariad_single_space_cluster_teacher 的 EXACT_TOPK_CONFIG 同一个思路，用来把
# "QK 图搜索质量" 和 "Q/K 表征学习质量" 解耦：这个基线下 match_exhaustive 应该
# 恒为 ~1，recall/match_true 纯粹反映梯度训练能不能学出正确的 Q/K。
EXACT_QK_CONFIG = {
    "enabled": False,
    "k": None,            # None = 复用 QK_CONFIG["k_final"]，保证跟 Ariad 路径聚合宽度一致
    "score_chunk": 4096,   # 分块大小：精确搜索计算量仍是 O(n_q * n_k)，内存只到 O(chunk * n_k)
}

# Dense attention 对照旋钮：开启后彻底不做任何 top-k 选择——每个节点对全部其他
# 节点做标准稠密 softmax attention（O(n^2) 显存/计算，n 大的话会很吃资源，这里默
# 认 n=2000 没问题）。跟 Ariad/EXACT_QK 不一样，这个模式下聚合本身用的是全量 n 个
# key 的连续权重，不存在"选中的 k 个"。但每个 epoch 仍然会从这个稠密分数矩阵里
# 取当前 top-k_final，跟 true_topk_idx 算 match_true，报告"如果把这套 dense
# attention 硬截断成 top-k，现在跟 ground truth 重合多少"。这是最上限的对照组：
# 图搜索完全不存在，纯看 Q/K 表征学习能不能把注意力集中到正确的近邻上。
DENSE_ATTENTION_CONFIG = {
    "enabled": False,
}

# QQ / KK：各自用简化版单空间 Ariad（O+OO+I，无副本/reset/elite-carry，见
# ../ariad_single_space/simplified_ariad.md），QQ 喂 Q，KK 喂 K，两份互相独立。
# 预算这次直接对齐 ariad_single_space/ariad_train_single.py 的 ARIAD_CONFIG
# （k=64, c_o2=80, c_i1=32），跟已验证收敛的单空间基线同宽度，做严格对照。
QQ_CONFIG = {
    "k": 64, "k_rev": 32, "c_o2": 80, "c_i1": 32, "c_oi": 0, "c_rand": 0,
    "score_chunk": 4096, "use_tiebreak": False, "compute_churn_stats": False,
}
KK_CONFIG = {
    "k": 64, "k_rev": 32, "c_o2": 80, "c_i1": 32, "c_oi": 0, "c_rand": 0,
    "score_chunk": 4096, "use_tiebreak": False, "compute_churn_stats": False,
}

# QK（真正决定 attention 用哪些 key 的图）：A(i)=当前 keys，+kF（key侧前向）+qF
# （query侧前向）+kR（key侧反向）+qR（query侧反向，默认关闭，先当诊断开关）。
# 预算跟着 QQ/KK 一起放宽到 32（原来 16），保持跟单空间基线同一量级对照。
QK_CONFIG = {
    "k_final": 32,   # 需与数据集 k_eval 对齐
    "k_rev": 32,
    "b_kF": 32,
    "b_qF": 32,
    "b_kR": 32,
    "b_qR": 0,        # 诊断开关：True 场景下改成 >0（见 qk_candidate_path_ablation.py）
    "score_chunk": 4096,
    "use_tiebreak": False,
}


def ensure_dataset(dataset_dir: Path) -> None:
    if (dataset_dir / "metadata.json").exists():
        return
    cfg = GEN_CONFIG
    dataset = datagen.generate_tau_first_synthetic_dataset(
        n_clusters=cfg["n_clusters"], cluster_size=cfg["cluster_size"],
        d_g=cfg["d_g"], d_v=cfg["d_v"], cluster_jitter=cfg["cluster_jitter"],
        k_eval=cfg["k_eval"], k_obs=cfg["k_obs"], tau=cfg["tau"], rho_g=cfg["rho_g"],
        seed=cfg["seed"], block_size=cfg["block_size"],
        attention_diag_topk=cfg["attention_diag_topk"],
        teacher_dense=cfg["teacher_dense"], teacher_topk=cfg["teacher_topk"],
        teacher_scale=cfg["teacher_scale"],
    )
    a = dataset["A_star"]
    top = np.sort(a, axis=1)[:, -cfg["k_eval"]:].sum(axis=1).mean()
    print(f"teacher check: eff_support={dataset['gate']['eff_support']:.2f}/{a.shape[0]}  "
          f"top{cfg['k_eval']}_attention_mass={top:.4f}")
    datagen.save_single_dataset(dataset, dataset_dir)
    gate = dataset.get("gate", {})
    print(f"generated dataset at {dataset_dir}: X={dataset['X'].shape}, gate={gate}")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_dir: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    x = torch.from_numpy(np.load(dataset_dir / "X.npy")).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(np.load(dataset_dir / "Y.npy")).to(device=device, dtype=torch.float32)
    true_topk_idx = torch.from_numpy(np.load(dataset_dir / "true_topk_idx.npy")).to(device=device, dtype=torch.long)

    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {"X": x, "Y": y, "true_topk_idx": true_topk_idx, "metadata": metadata}


def _rowwise_overlap_counts(gt_idx: torch.Tensor, pred_idx: torch.Tensor) -> torch.Tensor:
    a = gt_idx.unsqueeze(2)
    b = pred_idx.unsqueeze(1)
    return (a == b).any(dim=2).sum(dim=1).float()


def recall_at_k_directed_torch(gt_idx: torch.Tensor, pred_idx: torch.Tensor) -> float:
    if gt_idx.shape != pred_idx.shape:
        raise ValueError(f"shape mismatch: gt={gt_idx.shape}, pred={pred_idx.shape}")
    n, k = gt_idx.shape
    hits = _rowwise_overlap_counts(gt_idx, pred_idx).sum()
    return float(hits.item()) / float(n * k)


def topk_from_scores(scores: torch.Tensor, k: int) -> torch.Tensor:
    return torch.topk(scores, k=k, dim=1).indices


def overlap_coverage_torch(gt_idx: torch.Tensor, pred_idx: torch.Tensor) -> float:
    if gt_idx.shape[0] != pred_idx.shape[0]:
        raise ValueError(f"row mismatch: gt={gt_idx.shape}, pred={pred_idx.shape}")
    n, k_gt = gt_idx.shape
    hits = _rowwise_overlap_counts(gt_idx, pred_idx).sum()
    return float(hits.item()) / float(n * k_gt)


@torch.no_grad()
def exact_qk_neighbors(q: torch.Tensor, k: torch.Tensor, k_final: int, chunk: int = 4096) -> torch.Tensor:
    """Brute-force exact top-k (self excluded) over ALL n_k keys, scored
    against the CURRENT q,k -- the EXACT_QK_CONFIG comparison baseline in
    place of AriadSeekerQK's approximate graph search. Compute is
    O(n_q * n_k) (same as any exhaustive search), memory bounded to
    O(chunk * n_k): scored one query-row-block at a time, never
    materializing the full n_q x n_k matrix. Assumes self-attention over the
    same node set (key_id == query_id excluded, like AriadSeekerQK)."""
    n_q = q.size(0)
    device = q.device
    scale = torch.sqrt(torch.tensor(float(q.size(1)), device=device))
    out = torch.empty(n_q, k_final, device=device, dtype=torch.long)
    row_ar = torch.arange(n_q, device=device)
    for s in range(0, n_q, chunk):
        e = min(s + chunk, n_q)
        scores = (q[s:e] @ k.T) / scale
        local_rows = torch.arange(e - s, device=device)
        scores[local_rows, row_ar[s:e]] = float("-inf")
        out[s:e] = torch.topk(scores, k=k_final, dim=1).indices
    return out


class OneLayerQKAttention(nn.Module):
    """QK dual-space attention. By default SEPARATE W_q/W_k projections
    (Q != K), so the QQ graph and KK graph are genuinely different
    structures (unlike ../ariad_single_space, and unlike the ORIGINAL
    phase1/ariad_train.py which shares W_kq). Self-attention over the same n
    nodes: node i is simultaneously a query and a key.

    `bond_qk=True` ties K's projection to LITERALLY the same Parameter
    object as Q's (not just equal-at-init) -- real weight tying, matching
    the original phase1/ariad_train.py convention, so Q == K numerically
    for the whole run. Implemented by simply never creating a second
    `w_k` Parameter and reusing `self.w_q` in forward() instead -- NOT by
    assigning `self.w_k = self.w_q`, which would register the same
    Parameter object under two names and cause optimizers to apply two
    updates to it per step()."""

    def __init__(self, d_in: int, d_head: int, d_v_out: int,
                 qq_cfg: AriadSingleConfig, kk_cfg: AriadSingleConfig, qk_cfg: AriadQKConfig,
                 bond_qk: bool = False, exact_qk_k: int = None, exact_qk_chunk: int = 4096,
                 dense_attention: bool = False):
        super().__init__()
        self.bond_qk = bool(bond_qk)
        self.dense_attention = bool(dense_attention)
        self.k_final = int(qk_cfg.k_final)   # diagnostic top-k width, used in dense_attention mode too

        self.w_q = nn.Parameter(torch.empty(d_in, d_head))
        nn.init.xavier_uniform_(self.w_q)
        if self.bond_qk:
            self.w_k = None   # no separate parameter; forward() reuses self.w_q for k
        else:
            self.w_k = nn.Parameter(torch.empty(d_in, d_head))
            nn.init.xavier_uniform_(self.w_k)

        self.w_v = nn.Parameter(torch.empty(d_in, d_v_out))
        nn.init.xavier_uniform_(self.w_v)

        self.qq_seeker = AriadSeekerSingle(qq_cfg)
        self.kk_seeker = AriadSeekerSingle(kk_cfg)
        self.qk_seeker = AriadSeekerQK(qk_cfg)

        # EXACT_QK_CONFIG comparison baseline: when set, qq/kk/qk seekers are
        # constructed but NEVER called -- neighbors come from a fresh
        # brute-force exact top-k every forward instead.
        self.exact_qk_k = exact_qk_k
        self.exact_qk_chunk = exact_qk_chunk

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = x @ self.w_q
        k = x @ (self.w_q if self.bond_qk else self.w_k)
        v = x @ self.w_v

        d_head = q.size(1)
        scale = torch.sqrt(torch.tensor(float(d_head), device=x.device))
        n = q.size(0)

        if self.dense_attention:
            # Standard full dense attention: every node attends to ALL other
            # nodes (O(n^2)), no top-k selection anywhere in the aggregation.
            scores_dense = (q @ k.T) / scale                       # [n, n]
            scores_dense = scores_dense.clone()
            scores_dense.fill_diagonal_(float("-inf"))              # self excluded (same node set)
            attn = F.softmax(scores_dense, dim=1)
            y_hat = attn @ v
            # Diagnostic only: current top-k of this dense distribution --
            # not used for aggregation, just for match_true/match_exhaustive
            # reporting (same self-attention row, so this IS exhaustive by
            # construction: no separate approximate search to compare against).
            with torch.no_grad():
                neighbors = torch.topk(scores_dense, k=self.k_final, dim=1).indices
            return y_hat, neighbors, q, k

        if self.exact_qk_k is not None:
            neighbors = exact_qk_neighbors(q.detach(), k.detach(), self.exact_qk_k, chunk=self.exact_qk_chunk)
        else:
            qq_graph = self.qq_seeker(q)   # AriadSeekerSingle.forward: init/refresh internally, returns its own G
            kk_graph = self.kk_seeker(k)
            neighbors = self.qk_seeker(q, k, kk_graph, qq_graph)   # [n, k_final] key ids per query

        k_neighbors = k[neighbors]
        v_neighbors = v[neighbors]

        scores_sparse = (q.unsqueeze(1) * k_neighbors).sum(dim=2) / scale

        row_idx = torch.arange(n, device=x.device).unsqueeze(1)
        scores_sparse = scores_sparse.masked_fill(neighbors == row_idx, float("-inf"))

        attn = F.softmax(scores_sparse, dim=1)
        y_hat = (attn.unsqueeze(-1) * v_neighbors).sum(dim=1)
        return y_hat, neighbors, q, k


def maybe_oracle_init(model: OneLayerQKAttention, metadata: Dict, d_in: int, use_oracle_init: bool) -> None:
    if not use_oracle_init:
        return
    d_g = int(metadata["d_g"])
    d_v = int(metadata["d_v"])
    d_head = model.w_q.shape[1]

    with torch.no_grad():
        ws = (model.w_q,) if model.bond_qk else (model.w_q, model.w_k)
        for w in ws:
            w.zero_()
            z_dim = min(d_head, d_g)
            w[:z_dim, :z_dim] = torch.eye(z_dim, device=w.device)

        model.w_v.zero_()
        x_v_start = d_g
        v_dim = min(d_v, model.w_v.shape[1])
        if x_v_start + v_dim <= d_in:
            model.w_v[x_v_start : x_v_start + v_dim, :v_dim] = torch.eye(v_dim, device=model.w_v.device)


def freeze_wk_to_oracle_(model: OneLayerQKAttention, metadata: Dict) -> None:
    """Set w_k to the SAME identity-block oracle value maybe_oracle_init uses
    (recovers the G_noise sub-block of x exactly) and freeze it -- removes
    the Wq/Wk bilinear joint optimization, leaving only Wq to train against a
    K that is already exactly correct."""
    assert not model.bond_qk, "freeze_wk_to_oracle requires bond_QK=False (need a separate w_k to freeze)"
    d_g = int(metadata["d_g"])
    d_head = model.w_k.shape[1]
    with torch.no_grad():
        model.w_k.zero_()
        z_dim = min(d_head, d_g)
        model.w_k[:z_dim, :z_dim] = torch.eye(z_dim, device=model.w_k.device)
    model.w_k.requires_grad_(False)


def fix_wv_to_identity_v_(model: OneLayerQKAttention, metadata: Dict) -> None:
    """Set w_v to the identity on the V block of x (columns d_g : d_g+d_v) and freeze it,
    so the values are exactly the raw V the teacher aggregates -- the same setting as the
    toy probe (qk_independent_dense_probe.py), where V is given, not learned."""
    d_g = int(metadata["d_g"])
    d_v = int(metadata["d_v"])
    assert model.w_v.shape[1] == d_v, "value_mode='fixed_v' needs d_v_out == d_v"
    with torch.no_grad():
        model.w_v.zero_()
        model.w_v[d_g:d_g + d_v, :] = torch.eye(d_v, device=model.w_v.device)
    model.w_v.requires_grad_(False)


def train() -> Dict:
    """Trains one configuration and returns {"history": [per-eval records], "n": N, "k_eval": k}.
    Each record: epoch, mse (post-update, eval mode, training nodes), recall (teacher match@k),
    match_exhaustive (current-graph recall on a sampled subset of rows), replace_pct
    (100*[1 - mean_i |H_t(i) & H_{t-1}(i)|/k] vs the previous epoch's graph; only when evaluated
    on consecutive epochs, i.e. print_every=1)."""
    dataset_dir = DATA_CONFIG["dataset_root"] / dataset_name()
    ensure_dataset(dataset_dir)

    seed = TRAIN_CONFIG["seed"]
    epochs = TRAIN_CONFIG["epochs"]
    lr = TRAIN_CONFIG["lr"]
    weight_decay = TRAIN_CONFIG["weight_decay"]
    print_every = TRAIN_CONFIG["print_every"]
    use_oracle_init = MODEL_CONFIG["use_oracle_init"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    data = load_dataset(dataset_dir, device)
    x = data["X"]
    y = data["Y"]
    true_topk_idx = data["true_topk_idx"]
    metadata = data["metadata"]
    got_dense = bool(metadata.get("teacher_dense", True))   # datasets predating the knob were all dense
    if got_dense != bool(GEN_CONFIG["teacher_dense"]):
        raise ValueError(f"{dataset_dir} has teacher_dense={got_dense}, but GEN_CONFIG['teacher_dense']="
                         f"{GEN_CONFIG['teacher_dense']}; delete the dir or fix the knob")

    n, d_in = x.shape
    d_v_out = y.shape[1]
    d_head = int(metadata["d_head"])
    k_eval = true_topk_idx.shape[1]

    qq_cfg = AriadSingleConfig(**QQ_CONFIG)
    kk_cfg = AriadSingleConfig(**KK_CONFIG)
    qk_cfg = AriadQKConfig(**QK_CONFIG)
    bond_qk = bool(MODEL_CONFIG["bond_QK"])
    dense_attention = bool(DENSE_ATTENTION_CONFIG["enabled"])
    exact_qk_k = None
    if EXACT_QK_CONFIG["enabled"]:
        exact_qk_k = int(EXACT_QK_CONFIG["k"]) if EXACT_QK_CONFIG["k"] is not None else int(qk_cfg.k_final)
    model = OneLayerQKAttention(
        d_in=d_in, d_head=d_head, d_v_out=d_v_out, qq_cfg=qq_cfg, kk_cfg=kk_cfg, qk_cfg=qk_cfg, bond_qk=bond_qk,
        exact_qk_k=exact_qk_k, exact_qk_chunk=int(EXACT_QK_CONFIG["score_chunk"]),
        dense_attention=dense_attention,
    ).to(device)
    maybe_oracle_init(model, metadata, d_in=d_in, use_oracle_init=use_oracle_init)

    freeze_wk_to_oracle = bool(MODEL_CONFIG["freeze_wk_to_oracle"])
    if freeze_wk_to_oracle:
        freeze_wk_to_oracle_(model, metadata)
    if MODEL_CONFIG["value_mode"] == "fixed_v":
        fix_wv_to_identity_v_(model, metadata)
    elif MODEL_CONFIG["value_mode"] != "learned":
        raise ValueError(f"value_mode must be 'learned' or 'fixed_v', got {MODEL_CONFIG['value_mode']!r}")

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=lr, weight_decay=weight_decay)
    scheduler = None
    if TRAIN_CONFIG["cosine_decay"]:
        import math
        floor = float(TRAIN_CONFIG["cosine_floor"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda ep: floor + 0.5 * (1 - floor) * (1 + math.cos(math.pi * min(1.0, ep / max(1, epochs)))),
        )

    if dense_attention:
        mode_str = "DENSE_ATTENTION"
    elif exact_qk_k is not None:
        mode_str = f"EXACT_QK(k={exact_qk_k})"
    else:
        mode_str = "Ariad"
    print(f"device={device}, n={n}, d_in={d_in}, d_head={d_head}, d_v={d_v_out}, k_eval={k_eval}, bond_QK={bond_qk}, "
          f"freeze_wk_to_oracle={freeze_wk_to_oracle}, "
          f"mode={mode_str}, k_final={qk_cfg.k_final}, b_kF={qk_cfg.b_kF}, b_qF={qk_cfg.b_qF}, b_kR={qk_cfg.b_kR}, b_qR={qk_cfg.b_qR}")
    print(f"dataset={dataset_dir}  teacher={'dense' if got_dense else 'sparse top-' + str(metadata.get('teacher_topk'))}")

    history = []
    prev_neighbors, prev_epoch = None, None
    # Dedicated RNG for the eval-time row sample, so evaluating more or less often never changes
    # the training random stream (Ariad's graph refresh draws from the global RNG).
    eval_gen = torch.Generator(device=device).manual_seed(12345)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        y_hat, _, _, _ = model(x)
        mse = F.mse_loss(y_hat, y)
        mse.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if epoch % print_every == 0 or epoch == epochs:
            with torch.no_grad():
                model.eval()
                y_hat_eval, neighbors_eval, q_eval, k_eval_tensor = model(x)
                mse_eval = F.mse_loss(y_hat_eval, y).item()

                scale = torch.sqrt(torch.tensor(float(d_head), device=device))
                k_pred = neighbors_eval.shape[1]
                sample_size = EVAL_CONFIG["exhaustive_sample_size"]

                if sample_size is not None and sample_size < n:
                    sample_idx = torch.randperm(n, device=device, generator=eval_gen)[:sample_size]
                    dense_scores_sample = (q_eval[sample_idx] @ k_eval_tensor.T) / scale
                    row_ar = torch.arange(sample_size, device=device)
                    dense_scores_sample[row_ar, sample_idx] = float("-inf")
                    exhaustive_topk_idx = topk_from_scores(dense_scores_sample, k=k_pred)
                    match_exhaustive = recall_at_k_directed_torch(exhaustive_topk_idx, neighbors_eval[sample_idx])
                else:
                    dense_scores_eval = (q_eval @ k_eval_tensor.T) / scale
                    dense_scores_eval = dense_scores_eval.clone()
                    dense_scores_eval.fill_diagonal_(float("-inf"))
                    exhaustive_topk_idx = topk_from_scores(dense_scores_eval, k=k_pred)
                    match_exhaustive = recall_at_k_directed_torch(exhaustive_topk_idx, neighbors_eval)

                if k_pred == k_eval:
                    recall_k = recall_at_k_directed_torch(true_topk_idx, neighbors_eval)
                else:
                    recall_k = float("nan")
                match_true = overlap_coverage_torch(true_topk_idx, neighbors_eval)

                replace_pct = None
                if prev_neighbors is not None and prev_epoch == epoch - 1:
                    kept = recall_at_k_directed_torch(prev_neighbors, neighbors_eval)
                    replace_pct = 100.0 * (1.0 - kept)
                prev_neighbors, prev_epoch = neighbors_eval.clone(), epoch

            history.append({"epoch": epoch, "mse": mse_eval, "recall": recall_k,
                            "match_exhaustive": match_exhaustive, "match_true": match_true,
                            "replace_pct": replace_pct})
            recall_str = f"{recall_k:.6f}" if np.isfinite(recall_k) else "N/A"
            print(
                f"epoch={epoch:04d} mse={mse_eval:.8f} recall@{k_eval}={recall_str} "
                f"match_exhaustive={match_exhaustive:.6f} match_true={match_true:.6f}"
            )

    return {"history": history, "n": int(n), "k_eval": int(k_eval)}


if __name__ == "__main__":
    train()
