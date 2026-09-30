from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

"""
Synthetic generator for tau-first experiments.

Core setup:
- Routing geometry G and content V are generated independently.
- Teacher is exact full attention over all j!=i:
    S = G G^T / sqrt(d_g), A* = softmax(S / tau), Y = A* V
- Observed graph is built from noisy routing geometry G_noise via top-k.
- Model input is X = [G_noise | V], label is vector Y.
"""


# =========================
# User Knobs (edit here)
# =========================
# 说明：日常做实验只需要改这部分，__main__ 会直接使用这些配置。
TAUS_TO_GENERATE = [0.02]
OUTPUT_ROOT_DIRNAME = "synthetic_tau_datasets"

GEN_CONFIG = {
    "n_clusters": 125,
    "cluster_size": 16,
    "d_g": 64,
    "d_v": 64,
    "cluster_jitter": 0.10,
    "k_eval": 16,
    "k_obs": 16,
    "rho_g": 1.0,
    "seed": 0,
    "block_size": 512,
    "attention_diag_topk": 32,
    # True: teacher 是全局 dense attention，A* = softmax_{j!=i}(S/tau)。
    # False: teacher 只在 S 的 top-teacher_topk 上做稀疏聚合，其余 A*_ij = 0。
    "teacher_dense": True,
    "teacher_topk": 16,       # 仅 teacher_dense=False 时生效；建议与学生聚合宽度 k_final 一致
    "tau": 0.5,
    # "global_standardize": S = (g.g - mu)/sigma（论文单空间的 teacher 定义，离线算一次）。
    # "fixed": S = g.g / sqrt(d_g)，即 mu=0, sigma=sqrt(d_g)，跟学生的 q.k/sqrt(d) 同一个标准
    #          缩放，teacher 的 logits = g.g/(tau*sqrt(d_g))，没有任何数据相关的重新标准化。
    "teacher_scale": "global_standardize",
}

RANDOM_RECALL_CHECK = {
    "enabled": True,
    "embed_dim": 32,
    "seed": 123,
}


def _rng(seed: Optional[int]) -> np.random.RandomState:
    return np.random.RandomState(seed)


def row_l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def feature_standardize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32)


def _mask_self_neg_inf(scores: np.ndarray) -> np.ndarray:
    out = scores.copy()
    np.fill_diagonal(out, -np.inf)
    return out


def _global_standardize_logits(logits: np.ndarray) -> Tuple[np.ndarray, float, float]:
    finite_mask = np.isfinite(logits)
    off = logits[finite_mask]
    mu = float(off.mean())
    sigma = float(off.std())
    if sigma <= 1e-12:
        raise ValueError(f"global sigma too small: {sigma}")
    s = (logits - mu) / sigma
    s[~finite_mask] = -np.inf
    return s, mu, sigma


def _topk_indices_scores_rowwise(scores: np.ndarray, k: int, row_block: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    bsz, n = scores.shape
    if bsz > row_block and k < n:
        # Row-blocked: argpartition on the full matrix allocates an int64 N x N array (3 GB at N=20000).
        idx = np.empty((bsz, k), dtype=np.int64)
        val = np.empty((bsz, k), dtype=scores.dtype)
        for r0 in range(0, bsz, row_block):
            r1 = min(r0 + row_block, bsz)
            idx[r0:r1], val[r0:r1] = _topk_indices_scores_rowwise(scores[r0:r1], k, row_block)
        return idx, val
    if k >= n:
        idx = np.argsort(-scores, axis=1)
        val = np.take_along_axis(scores, idx, axis=1)
        return idx, val

    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    idx = np.take_along_axis(part, order, axis=1)
    val = np.take_along_axis(part_scores, order, axis=1)
    return idx, val


def topk_scaled_dot_blockwise(
    x: np.ndarray,
    k: int,
    block_size: int = 512,
    exclude_self: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    n, d = x.shape
    if not (0 < k < n):
        raise ValueError(f"require 0 < k < n, got k={k}, n={n}")

    scale = 1.0 / np.sqrt(float(d))
    nn_idx = np.empty((n, k), dtype=np.int64)
    nn_scores = np.empty((n, k), dtype=np.float32)

    for i0 in range(0, n, block_size):
        i1 = min(i0 + block_size, n)
        xb = x[i0:i1]
        scores = (xb @ x.T) * scale
        if exclude_self:
            rows = np.arange(i1 - i0)
            cols = i0 + rows
            scores[rows, cols] = -np.inf
        idx, val = _topk_indices_scores_rowwise(scores, k)
        nn_idx[i0:i1] = idx
        nn_scores[i0:i1] = val.astype(np.float32)
    return nn_idx, nn_scores


def softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    z = x - x_max
    expz = np.exp(z)
    expz[~np.isfinite(x)] = 0.0
    return expz / (np.sum(expz, axis=axis, keepdims=True) + 1e-12)


def scaled_dot_scores_full(x: np.ndarray, mask_self: bool = True) -> np.ndarray:
    _, d = x.shape
    s = (x @ x.T) / np.sqrt(float(d))
    if mask_self:
        np.fill_diagonal(s, -np.inf)
    return s


def build_observed_graph_topk(
    g_noise: np.ndarray,
    k_obs: int,
    block_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    _ = block_size
    s_obs = _mask_self_neg_inf(g_noise @ g_noise.T)
    obs_topk_idx, _ = _topk_indices_scores_rowwise(s_obs, k_obs)
    n = g_noise.shape[0]
    src = np.repeat(np.arange(n, dtype=np.int64), k_obs)
    dst = obs_topk_idx.reshape(-1).astype(np.int64)
    edge_index_obs = np.stack([src, dst], axis=0)
    return edge_index_obs, obs_topk_idx


def _oracle_cluster_recall(true_topk: np.ndarray, lab: np.ndarray, k_eval: int) -> float:
    """Fraction of the (at most cluster_size-1) same-cluster mates that the teacher's
    top-k_eval recovers. When k_eval > cluster_size-1 the top-k necessarily contains
    outsiders, so we only ask the first min(k_eval, mates) slots to be mates (normalizing
    by k_eval would cap this at (cluster_size-1)/k_eval and fail the gate for any k > cluster)."""
    n = true_topk.shape[0]
    hits = 0
    denom = 0
    for i in range(n):
        same_cluster = np.where(lab == lab[i])[0]
        same_cluster = same_cluster[same_cluster != i]
        m = min(k_eval, len(same_cluster))
        hits += len(set(true_topk[i, :m].tolist()).intersection(set(same_cluster.tolist())))
        denom += m
    return float(hits) / float(denom)


def _topk_overlap(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    n, k = a.shape
    hits = 0
    for i in range(n):
        hits += len(set(a[i].tolist()).intersection(set(b[i].tolist())))
    return float(hits) / float(n * k)


def sanity_gate(dataset: Dict[str, np.ndarray]) -> None:
    a_star = dataset["A_star"]
    s = dataset["S"]
    g = dataset["G"]
    lab = dataset["lab"]
    meta = dataset["meta"]

    tau = float(meta["tau"])
    cluster_size = int(meta["cluster_size"])
    k_eval = int(meta["k_eval"])

    eff_support = float(np.mean(1.0 / (np.sum(a_star * a_star, axis=1) + 1e-12)))

    true_topk, _ = _topk_indices_scores_rowwise(s, k_eval)
    recall_oracle = _oracle_cluster_recall(true_topk, lab, k_eval)

    g_perturb = row_l2_normalize(
        g + 0.05 * _rng(int(meta["seed"]) + 999).randn(*g.shape).astype(np.float32)
    ).astype(np.float32)
    l_perturb = _mask_self_neg_inf(g_perturb @ g_perturb.T)
    s_perturb, _, _ = _global_standardize_logits(l_perturb)
    pert_topk, _ = _topk_indices_scores_rowwise(s_perturb, k_eval)
    # Only the first min(k_eval, cluster_size-1) slots are cluster mates with a
    # well-defined order; beyond that the top-k is filled with outsiders whose
    # relative order is inherently unstable under any perturbation.
    k_stab = min(k_eval, cluster_size - 1)
    stability = _topk_overlap(true_topk[:, :k_stab], pert_topk[:, :k_stab])

    if 0.2 <= tau <= 0.5:
        lower = 0.2 * cluster_size
        upper = 3.0 * cluster_size
        if not (lower <= eff_support <= upper):
            raise ValueError(
                f"sanity_gate fail: eff_support={eff_support:.4f} out of [{lower:.2f}, {upper:.2f}] for tau={tau}. "
                "try tuning cluster_jitter/tau/cluster_size"
            )

    if recall_oracle < 0.90:
        raise ValueError(
            f"sanity_gate fail: recall_oracle={recall_oracle:.4f} < 0.90. "
            "try tuning cluster_jitter/tau/cluster_size"
        )

    if stability < 0.80:
        raise ValueError(
            f"sanity_gate fail: stability={stability:.4f} < 0.80. "
            "try tuning cluster_jitter/tau/cluster_size"
        )

    dataset["gate"] = {
        "eff_support": eff_support,
        "recall_oracle": recall_oracle,
        "stability": stability,
    }


def recall_at_k_directed(gt_nn_idx: np.ndarray, pred_nn_idx: np.ndarray) -> float:
    if gt_nn_idx.shape != pred_nn_idx.shape:
        raise ValueError("gt_nn_idx and pred_nn_idx must have same shape")
    n, k = gt_nn_idx.shape
    hits = 0
    for i in range(n):
        hits += len(set(gt_nn_idx[i].tolist()).intersection(set(pred_nn_idx[i].tolist())))
    return float(hits) / float(n * k)


def compute_model_topk_neighbors(
    z: np.ndarray,
    k: int,
    block_size: int = 512,
    exclude_self: bool = True,
) -> np.ndarray:
    pred_idx, _ = topk_scaled_dot_blockwise(z, k=k, block_size=block_size, exclude_self=exclude_self)
    return pred_idx


def generate_tau_first_synthetic_dataset(
    *,
    n_clusters: int = 312,
    cluster_size: int = 16,
    d_g: int = 64,
    d_v: int = 64,
    cluster_jitter: float = 0.10,
    k_eval: int = 16,
    k_obs: int = 16,
    tau: float = 0.5,
    rho_g: float = 1.0,
    seed: int = 0,
    block_size: int = 512,
    attention_diag_topk: int = 64,
    return_dense_attention: bool = False,
    teacher_dense: bool = True,
    teacher_topk: int = 16,
    teacher_scale: str = "global_standardize",
) -> Dict[str, np.ndarray]:
    if teacher_scale not in ("global_standardize", "fixed"):
        raise ValueError(f"teacher_scale must be 'global_standardize' or 'fixed', got {teacher_scale!r}")
    n = n_clusters * cluster_size
    if n_clusters <= 0 or cluster_size <= 1:
        raise ValueError("n_clusters must be >0 and cluster_size must be >1")
    if not (0 < k_eval < n):
        raise ValueError(f"require 0 < k_eval < n, got k_eval={k_eval}, n={n}")
    if not (0 < k_obs < n):
        raise ValueError(f"require 0 < k_obs < n, got k_obs={k_obs}, n={n}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if not (0.0 <= rho_g <= 1.0):
        raise ValueError(f"rho_g must be in [0,1], got {rho_g}")
    if d_g <= 0 or d_v <= 0:
        raise ValueError("d_g and d_v must be positive")
    if cluster_jitter < 0.0:
        raise ValueError("cluster_jitter must be non-negative")
    if not teacher_dense and not (0 < teacher_topk < n):
        raise ValueError(f"require 0 < teacher_topk < n, got teacher_topk={teacher_topk}, n={n}")

    rng = _rng(seed)

    centers = row_l2_normalize(rng.randn(n_clusters, d_g).astype(np.float32)).astype(np.float32)
    lab = np.repeat(np.arange(n_clusters, dtype=np.int64), cluster_size)
    g = centers[lab] + cluster_jitter * rng.randn(n, d_g).astype(np.float32)
    g = row_l2_normalize(g).astype(np.float32)

    v = rng.randn(n, d_v).astype(np.float32)

    l = _mask_self_neg_inf(g @ g.T)
    if teacher_scale == "fixed":
        mu, sigma = 0.0, float(np.sqrt(d_g))
        s = l / np.float32(sigma)          # -inf on the diagonal stays -inf
    else:
        s, mu, sigma = _global_standardize_logits(l)
    del l   # free one N x N buffer before the teacher softmax
    if teacher_dense:
        teacher_logits = s / float(tau)
    else:
        # sparse teacher: keep only each row's top-teacher_topk logits, the rest get zero weight
        t_idx, t_val = _topk_indices_scores_rowwise(s, teacher_topk)
        teacher_logits = np.full_like(s, -np.inf)
        np.put_along_axis(teacher_logits, t_idx, t_val / float(tau), axis=1)
    a_star = softmax_stable(teacher_logits, axis=1).astype(np.float32)
    del teacher_logits
    y = (a_star @ v).astype(np.float32)

    true_topk_idx, true_topk_scores = _topk_indices_scores_rowwise(s, k_eval)

    k_for_gap = min(k_eval + 1, n - 1)
    s_top_gap_idx, s_top_gap_scores = _topk_indices_scores_rowwise(s, k_for_gap)
    if s_top_gap_scores.shape[1] >= k_eval + 1:
        true_topk_gap = (s_top_gap_scores[:, k_eval - 1] - s_top_gap_scores[:, k_eval]).astype(np.float32)
    else:
        true_topk_gap = np.zeros((n,), dtype=np.float32)

    row_entropy = (-a_star * np.log(np.maximum(a_star, 1e-12))).sum(axis=1).astype(np.float32)

    rand_geom = row_l2_normalize(rng.randn(n, d_g).astype(np.float32))
    g_noise = row_l2_normalize(
        rho_g * g + np.sqrt(max(1.0 - rho_g * rho_g, 0.0)) * rand_geom
    ).astype(np.float32)

    edge_index_obs, obs_topk_idx = build_observed_graph_topk(
        g_noise=g_noise,
        k_obs=k_obs,
        block_size=block_size,
    )

    x = np.concatenate([g_noise, v], axis=1).astype(np.float32)
    x = feature_standardize(x, eps=1e-6)

    a_diag_k = min(max(1, int(attention_diag_topk)), n - 1)
    a_star_topk_idx, a_star_topk_val = _topk_indices_scores_rowwise(a_star, a_diag_k)

    out: Dict[str, np.ndarray] = {
        "X": x,
        "Y": y,
        "edge_index_obs": edge_index_obs,
        "true_topk": true_topk_idx.astype(np.int64),
        "true_topk_idx": true_topk_idx.astype(np.int64),
        "true_topk_scores": true_topk_scores.astype(np.float32),
        "true_topk_gap": true_topk_gap,
        "A_star_topk_idx": a_star_topk_idx.astype(np.int64),
        "A_star_topk_val": a_star_topk_val.astype(np.float32),
        "A_star_row_entropy": row_entropy,
        "A_star": a_star,
        "G": g,
        "V": v,
        "G_noise": g_noise,
        "lab": lab,
        "S": s.astype(np.float32),
        "obs_topk_idx": obs_topk_idx.astype(np.int64),
    }
    if return_dense_attention:
        out["A_star_dense"] = a_star

    out["meta"] = {
        "tau": float(tau),
        "rho_g": float(rho_g),
        "sigma": float(sigma),
        "mu": float(mu),
        "r": int(d_g),
        "d_head": int(d_g),
        "d_g": int(d_g),
        "d_v": int(d_v),
        "n": int(n),
        "n_clusters": int(n_clusters),
        "cluster_size": int(cluster_size),
        "cluster_jitter": float(cluster_jitter),
        "k_eval": int(k_eval),
        "k_obs": int(k_obs),
        "seed": int(seed),
        "teacher_dense": bool(teacher_dense),
        "teacher_topk": None if teacher_dense else int(teacher_topk),
        "teacher_scale": teacher_scale,
    }
    out["metadata"] = out["meta"]
    sanity_gate(out)
    return out


def generate_tau_sweep_datasets(
    taus: Iterable[float],
    *,
    n_clusters: int = 312,
    cluster_size: int = 16,
    d_g: int = 64,
    d_v: int = 64,
    cluster_jitter: float = 0.10,
    k_eval: int = 16,
    k_obs: int = 16,
    rho_g: float = 1.0,
    seed: int = 0,
    block_size: int = 512,
    attention_diag_topk: int = 64,
    return_dense_attention: bool = False,
    teacher_dense: bool = True,
    teacher_topk: int = 16,
    teacher_scale: str = "global_standardize",
) -> Dict[float, Dict[str, np.ndarray]]:
    datasets: Dict[float, Dict[str, np.ndarray]] = {}
    for tau in taus:
        tau_key = float(tau)
        datasets[tau_key] = generate_tau_first_synthetic_dataset(
            n_clusters=n_clusters,
            cluster_size=cluster_size,
            d_g=d_g,
            d_v=d_v,
            cluster_jitter=cluster_jitter,
            k_eval=k_eval,
            k_obs=k_obs,
            tau=tau_key,
            rho_g=rho_g,
            seed=seed,
            block_size=block_size,
            attention_diag_topk=attention_diag_topk,
            return_dense_attention=return_dense_attention,
            teacher_dense=teacher_dense,
            teacher_topk=teacher_topk,
            teacher_scale=teacher_scale,
        )
    return datasets


def dataset_tag(tau_tag: str, teacher_dense: bool, teacher_topk: int) -> str:
    """Directory name for a dataset; sparse-teacher datasets get their own suffix."""
    return tau_tag if teacher_dense else f"{tau_tag}_sparseT{teacher_topk}"


def save_single_dataset(dataset: Dict[str, np.ndarray], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "X.npy", dataset["X"])
    np.save(save_dir / "Y.npy", dataset["Y"])
    np.save(save_dir / "edge_index_obs.npy", dataset["edge_index_obs"])
    np.save(save_dir / "true_topk_idx.npy", dataset["true_topk_idx"])
    np.save(save_dir / "true_topk_scores.npy", dataset["true_topk_scores"])
    np.save(save_dir / "true_topk_gap.npy", dataset["true_topk_gap"])
    np.save(save_dir / "A_star_topk_idx.npy", dataset["A_star_topk_idx"])
    np.save(save_dir / "A_star_topk_val.npy", dataset["A_star_topk_val"])
    np.save(save_dir / "A_star_row_entropy.npy", dataset["A_star_row_entropy"])
    np.save(save_dir / "true_topk.npy", dataset["true_topk"])
    np.save(save_dir / "G.npy", dataset["G"])
    np.save(save_dir / "V.npy", dataset["V"])
    np.save(save_dir / "G_noise.npy", dataset["G_noise"])
    np.save(save_dir / "lab.npy", dataset["lab"])
    np.save(save_dir / "S.npy", dataset["S"])
    np.save(save_dir / "obs_topk_idx.npy", dataset["obs_topk_idx"])

    if "A_star_dense" in dataset:
        np.save(save_dir / "A_star_dense.npy", dataset["A_star_dense"])

    metadata = dataset.get("meta", dataset.get("metadata", {}))
    with (save_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_tau_sweep_datasets(
    datasets: Dict[float, Dict[str, np.ndarray]],
    root_dir: Path,
) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    for tau, dataset in datasets.items():
        tau_dir = root_dir / f"tau_{tau:g}"
        save_single_dataset(dataset, tau_dir)


def generate_toy_dataset_joint_graph_feature(
    *,
    n: int = 5000,
    d_H: int = 64,
    d_g: int = 64,
    k: int = 16,
    n_clusters: int = 1,
    cluster_std: float = 1.0,
    outlier_ratio: float = 0.0,
    rho_g: float = 1.0,
    sigma_g_obs: float = 0.0,
    random_state: int = 0,
    block_size: int = 512,
) -> Dict[str, np.ndarray]:
    n_clusters_eff = int(n_clusters) if int(n_clusters) > 1 else max(1, int(n // 16))
    cluster_size_eff = max(2, int(max(2, n) // n_clusters_eff))
    _ = cluster_std
    _ = outlier_ratio
    _ = sigma_g_obs
    return generate_tau_first_synthetic_dataset(
        n_clusters=n_clusters_eff,
        cluster_size=cluster_size_eff,
        d_g=d_g,
        d_v=d_H,
        cluster_jitter=0.10,
        k_eval=k,
        k_obs=k,
        tau=1.0,
        rho_g=rho_g,
        seed=random_state,
        block_size=block_size,
    )


if __name__ == "__main__":
    taus = TAUS_TO_GENERATE
    output_root = Path(__file__).resolve().parent / (
        OUTPUT_ROOT_DIRNAME if GEN_CONFIG["teacher_dense"] else f"{OUTPUT_ROOT_DIRNAME}_sparseT{GEN_CONFIG['teacher_topk']}"
    )

    all_data = generate_tau_sweep_datasets(
        taus,
        n_clusters=GEN_CONFIG["n_clusters"],
        cluster_size=GEN_CONFIG["cluster_size"],
        d_g=GEN_CONFIG["d_g"],
        d_v=GEN_CONFIG["d_v"],
        cluster_jitter=GEN_CONFIG["cluster_jitter"],
        k_eval=GEN_CONFIG["k_eval"],
        k_obs=GEN_CONFIG["k_obs"],
        rho_g=GEN_CONFIG["rho_g"],
        seed=GEN_CONFIG["seed"],
        block_size=GEN_CONFIG["block_size"],
        attention_diag_topk=GEN_CONFIG["attention_diag_topk"],
        teacher_dense=GEN_CONFIG["teacher_dense"],
        teacher_topk=GEN_CONFIG["teacher_topk"],
        teacher_scale=GEN_CONFIG["teacher_scale"],
    )

    save_tau_sweep_datasets(all_data, output_root)
    print(f"Saved datasets to: {output_root}")

    for tau, data in all_data.items():
        ent = data["A_star_row_entropy"]
        gate = data.get("gate", {})
        print(
            f"tau={tau:>4}: X={data['X'].shape}, Y={data['Y'].shape}, "
            f"edges={data['edge_index_obs'].shape[1]}, entropy_mean={float(ent.mean()):.4f}, "
            f"eff_support={gate.get('eff_support', float('nan')):.3f}, "
            f"recall_oracle={gate.get('recall_oracle', float('nan')):.3f}"
        )

    if RANDOM_RECALL_CHECK["enabled"]:
        eval_tau = float(taus[0])
        z_rand = _rng(RANDOM_RECALL_CHECK["seed"]).randn(
            all_data[eval_tau]["X"].shape[0],
            RANDOM_RECALL_CHECK["embed_dim"],
        ).astype(np.float32)
        pred_idx = compute_model_topk_neighbors(
            z_rand,
            k=GEN_CONFIG["k_eval"],
            block_size=GEN_CONFIG["block_size"],
            exclude_self=True,
        )
        rec = recall_at_k_directed(all_data[eval_tau]["true_topk_idx"], pred_idx)
        print(f"Random recall@{GEN_CONFIG['k_eval']} (tau={eval_tau:g}):", rec)
