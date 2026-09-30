from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ariad_single import AriadSingleConfig, AriadSeekerSingle


# =========================
# User Knobs (edit here)
# =========================
DATA_CONFIG = {
    "dataset_root": Path(__file__).resolve().parent / "synthetic_tau_datasets",
    "tau_tag": "tau_0.5",
}

TRAIN_CONFIG = {
    "seed": 0,
    "epochs": 150,
    "lr": 1e-2,
    "weight_decay": 0.0,
    "print_every": 1,
}

# 诊断开销旋钮：match_exhaustive 需要用当前 Z 做一次 N x N 暴力打分，是 O(N^2)，
# 跟 Ariad 训练本身的线性开销完全不是一回事。N 变大后这里会成为每 epoch 最慢的部分。
EVAL_CONFIG = {
    # None = 用全部 N 个节点算 exhaustive top-k（精确，但 O(N^2)）。
    # 填一个整数（比如 2000）则每次只随机抽这么多行做暴力对照，
    # match_exhaustive 变成无偏估计，开销降到 O(sample_size * N)。
    "exhaustive_sample_size": 1000,
}

MODEL_CONFIG = {
    "use_oracle_init": False,
}

# 简化版单空间 Ariad：只维护一张图，没有副本、没有 reset、没有 elite memory carry
# （见 simplified_ariad.md）。score_ij = z_i . z_j，Z 同时充当 query 和 key。
#
# c_oi（OI 路径）已去掉，预算转给 c_o2（OO 路径）：candidate_path_ablation.py 的
# 只读诊断 + real_ablation_oi_to_oo.py 的真实训练对比都显示 OI 跟"多撒同等预算的
# OO"在入选率/得分增益/最终任务指标上没有可辨识差异（见 simplified_ariad.md 附录A）。
ARIAD_CONFIG = {
    "k": 32,               # 近邻数（唯一一张图的宽度）；teacher match@k 也用同一个 k（teacher top-k 由 S.npy 现算）
    "k_rev": 16 * 2,        # 反向缓冲大小：每个节点记录多少"谁指向我"
    "c_o2": 64 + 16,        # 局部扩张候选预算：二跳 outward 候选数量（原 64，加上从 OI 转来的 16）
    "c_i1": 16 * 2,         # 局部扩张候选预算：一跳 inbound 候选数量（不变）
    "c_oi": 0,              # 局部扩张候选预算：OI 混合候选（outbound 的 inbound）数量 -- 已去掉
    "c_rand": 0,            # 局部扩张候选预算：随机候选数量
    "score_chunk": 4096,    # 打分分块大小：越大通常越快但显存/内存更高
    "use_tiebreak": False,
    "compute_churn_stats": False,
}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def teacher_topk_from_s(dataset_dir: Path, k: int, device: torch.device, chunk: int = 4096) -> torch.Tensor:
    """Teacher neighbor set T_k(i) = top-k of the teacher logits S[i, j], j != i.

    S.npy is the globally standardised G G^T with the diagonal already -inf, so
    this equals the top-k of the teacher attention row A*_i = softmax(S_i / tau)
    (softmax is monotone). Independent of the generator's k_eval, which is tied
    to its sanity gate (cluster_size) and cannot simply be raised."""
    s = np.load(dataset_dir / "S.npy", mmap_mode="r")
    out = []
    for r0 in range(0, s.shape[0], chunk):
        rows = torch.from_numpy(np.array(s[r0 : r0 + chunk])).to(device)
        out.append(rows.topk(k, dim=1).indices)
    return torch.cat(out, dim=0)


def load_dataset(dataset_dir: Path, device: torch.device, teacher_k: int | None = None) -> Dict[str, torch.Tensor]:
    """teacher_k=None: use the stored true_topk_idx (k = generator k_eval).
    Otherwise recompute the teacher top-teacher_k from S.npy."""
    x = torch.from_numpy(np.load(dataset_dir / "X.npy")).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(np.load(dataset_dir / "Y.npy")).to(device=device, dtype=torch.float32)
    true_topk_idx = torch.from_numpy(np.load(dataset_dir / "true_topk_idx.npy")).to(device=device, dtype=torch.long)
    if teacher_k is not None and teacher_k != true_topk_idx.shape[1]:
        true_topk_idx = teacher_topk_from_s(dataset_dir, teacher_k, device)

    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    import json

    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "X": x,
        "Y": y,
        "true_topk_idx": true_topk_idx,
        "metadata": metadata,
    }


def _rowwise_overlap_counts(gt_idx: torch.Tensor, pred_idx: torch.Tensor) -> torch.Tensor:
    """Vectorized per-row |gt ∩ pred| via broadcast comparison (no Python loop).

    gt_idx: [n, k_gt], pred_idx: [n, k_pred] (widths may differ). Returns [n]
    float tensor of intersection sizes. Assumes no duplicate ids within a row
    (true for top-k neighbor lists here), same as the original set-based version.
    """
    a = gt_idx.unsqueeze(2)      # [n, k_gt, 1]
    b = pred_idx.unsqueeze(1)    # [n, 1, k_pred]
    return (a == b).any(dim=2).sum(dim=1).float()  # [n]


def recall_at_k_directed_torch(gt_idx: torch.Tensor, pred_idx: torch.Tensor) -> float:
    if gt_idx.shape != pred_idx.shape:
        raise ValueError(f"shape mismatch: gt={gt_idx.shape}, pred={pred_idx.shape}")

    n, k = gt_idx.shape
    hits = _rowwise_overlap_counts(gt_idx, pred_idx).sum()
    return float(hits.item()) / float(n * k)


def topk_from_scores(scores: torch.Tensor, k: int) -> torch.Tensor:
    topk = torch.topk(scores, k=k, dim=1)
    return topk.indices


def overlap_coverage_torch(gt_idx: torch.Tensor, pred_idx: torch.Tensor) -> float:
    if gt_idx.shape[0] != pred_idx.shape[0]:
        raise ValueError(f"row mismatch: gt={gt_idx.shape}, pred={pred_idx.shape}")

    n, k_gt = gt_idx.shape
    hits = _rowwise_overlap_counts(gt_idx, pred_idx).sum()
    return float(hits.item()) / float(n * k_gt)


class OneLayerSingleSpaceAttention(nn.Module):
    """Single-space attention: one projection W_z produces Z, which is used
    as BOTH query and key (score_ij = z_i . z_j / sqrt(d_z)). This is the
    minimal-complexity counterpart to ariad_train.py's OneLayerFullAttention,
    which keeps Q and K as separate (numerically identical, since they share
    W_kq) tensors flowing through a two-phase KK/QK Ariad seeker."""

    def __init__(self, d_in: int, d_z: int, d_v_out: int, ariad_cfg: AriadSingleConfig):
        super().__init__()
        self.w_z = nn.Parameter(torch.empty(d_in, d_z))
        self.w_v = nn.Parameter(torch.empty(d_in, d_v_out))
        nn.init.xavier_uniform_(self.w_z)
        nn.init.xavier_uniform_(self.w_v)
        self.ariad_seeker = AriadSeekerSingle(ariad_cfg)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = x @ self.w_z
        v = x @ self.w_v

        neighbors = self.ariad_seeker(z)

        d_z = z.size(1)
        scale = torch.sqrt(torch.tensor(float(d_z), device=x.device))

        z_neighbors = z[neighbors]
        v_neighbors = v[neighbors]

        scores_sparse = (z.unsqueeze(1) * z_neighbors).sum(dim=2) / scale

        n = z.size(0)
        row_idx = torch.arange(n, device=x.device).unsqueeze(1)
        scores_sparse = scores_sparse.masked_fill(neighbors == row_idx, float("-inf"))

        attn = F.softmax(scores_sparse, dim=1)
        y_hat = (attn.unsqueeze(-1) * v_neighbors).sum(dim=1)
        return y_hat, neighbors, z


def maybe_oracle_init(model: OneLayerSingleSpaceAttention, metadata: Dict, d_in: int, use_oracle_init: bool) -> None:
    if not use_oracle_init:
        return

    d_g = int(metadata["d_g"])
    d_v = int(metadata["d_v"])
    d_z = model.w_z.shape[1]

    with torch.no_grad():
        model.w_z.zero_()
        model.w_v.zero_()

        z_dim = min(d_z, d_g)
        model.w_z[:z_dim, :z_dim] = torch.eye(z_dim, device=model.w_z.device)

        x_v_start = d_g
        v_dim = min(d_v, model.w_v.shape[1])
        if x_v_start + v_dim <= d_in:
            model.w_v[x_v_start : x_v_start + v_dim, :v_dim] = torch.eye(v_dim, device=model.w_v.device)


def train() -> None:
    dataset_dir = DATA_CONFIG["dataset_root"] / DATA_CONFIG["tau_tag"]

    seed = TRAIN_CONFIG["seed"]
    epochs = TRAIN_CONFIG["epochs"]
    lr = TRAIN_CONFIG["lr"]
    weight_decay = TRAIN_CONFIG["weight_decay"]
    print_every = TRAIN_CONFIG["print_every"]
    use_oracle_init = MODEL_CONFIG["use_oracle_init"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    data = load_dataset(dataset_dir, device, teacher_k=ARIAD_CONFIG["k"])
    x = data["X"]
    y = data["Y"]
    true_topk_idx = data["true_topk_idx"]
    metadata = data["metadata"]

    n, d_in = x.shape
    d_v_out = y.shape[1]
    d_z = int(metadata["d_head"])
    k_eval = true_topk_idx.shape[1]

    ariad_cfg = AriadSingleConfig(**ARIAD_CONFIG)
    model = OneLayerSingleSpaceAttention(d_in=d_in, d_z=d_z, d_v_out=d_v_out, ariad_cfg=ariad_cfg).to(device)
    maybe_oracle_init(model, metadata, d_in=d_in, use_oracle_init=use_oracle_init)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print(f"device={device}, n={n}, d_in={d_in}, d_z={d_z}, d_v={d_v_out}, k_eval={k_eval}")
    print(f"dataset={dataset_dir}")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        y_hat, _, _ = model(x)
        mse = F.mse_loss(y_hat, y)
        mse.backward()
        optimizer.step()

        if epoch % print_every == 0:
            with torch.no_grad():
                model.eval()
                y_hat_eval, neighbors_eval, z_eval = model(x)
                mse_eval = F.mse_loss(y_hat_eval, y).item()

                scale = torch.sqrt(torch.tensor(float(d_z), device=device))
                k_pred = neighbors_eval.shape[1]
                sample_size = EVAL_CONFIG["exhaustive_sample_size"]

                if sample_size is not None and sample_size < n:
                    # Sampled exhaustive comparison (unbiased estimate of
                    # match_exhaustive): O(sample_size * n) instead of the
                    # full O(n^2) dense z_eval @ z_eval.T.
                    sample_idx = torch.randperm(n, device=device)[:sample_size]
                    dense_scores_sample = (z_eval[sample_idx] @ z_eval.T) / scale
                    row_ar = torch.arange(sample_size, device=device)
                    dense_scores_sample[row_ar, sample_idx] = float("-inf")
                    exhaustive_topk_idx = topk_from_scores(dense_scores_sample, k=k_pred)
                    match_exhaustive = recall_at_k_directed_torch(
                        exhaustive_topk_idx, neighbors_eval[sample_idx]
                    )
                else:
                    dense_scores_eval = (z_eval @ z_eval.T) / scale
                    dense_scores_eval = dense_scores_eval.clone()
                    dense_scores_eval.fill_diagonal_(float("-inf"))
                    exhaustive_topk_idx = topk_from_scores(dense_scores_eval, k=k_pred)
                    match_exhaustive = recall_at_k_directed_torch(exhaustive_topk_idx, neighbors_eval)

                if k_pred == k_eval:
                    recall_k = recall_at_k_directed_torch(true_topk_idx, neighbors_eval)
                else:
                    recall_k = float("nan")
                match_true = overlap_coverage_torch(true_topk_idx, neighbors_eval)

            recall_str = f"{recall_k:.6f}" if np.isfinite(recall_k) else "N/A"
            print(
                f"epoch={epoch:04d} mse={mse_eval:.8f} recall@{k_eval}={recall_str} "
                f"match_exhaustive={match_exhaustive:.6f} match_true={match_true:.6f}"
            )


if __name__ == "__main__":
    train()
