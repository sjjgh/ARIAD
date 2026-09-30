"""compare_dense_exact_ariad.py

Three-way comparison on the same dataset / seed / weight init:

  dense : full softmax attention over all j != i (one N x N GEMM for QK^T)
  exact : sparse attention over the EXACT top-k of the current Z
          (brute-force N x N search under no_grad, every step)
  ariad : sparse attention over the Ariad single-graph neighbors
          (ariad_single.py, ARIAD_CONFIG from ariad_train_single.py)

exact and ariad share the SAME sparse attention module
(OneLayerSingleSpaceAttention); they differ only in the neighbor finder, so
any gap in cost/quality is attributable to neighbor search alone.

A single k = ARIAD_CONFIG["k"] is used for every method AND for evaluation.

Quality metrics (evaluated after each training step, on the post-step model):
  teacher match@k = mean_i |T_k(i) ∩ N_k(i)| / k
      T_k(i) : teacher neighbor set = top-k of the teacher logits S[i, j],
               j != i (== top-k of A*_i, softmax is monotone); recomputed from
               S.npy by ariad_train_single.teacher_topk_from_s
      N_k(i) : student neighbor set, self excluded
               dense : top-k of the learned z_i . z_j (dense attends to all j;
                       this is the set its learned geometry would select)
               exact : exact top-k of the current Z (what it attends over)
               ariad : the Ariad graph G(i) it attends over (last refreshed
                       in the training step, i.e. from the pre-step Z)
  match_exhaustive = mean_i |TopK_Z(i) ∩ N_k(i)| / k  (search quality against
      the exact top-k of the current student Z; 1 by construction for
      dense/exact)

Cost accounting (per TRAINING step, i.e. one forward + backward; the
per-epoch evaluation pass is not counted). Two kinds of similarity scores
are kept SEPARATE, because they cost differently:

  P_attn   : differentiable attention scores (enter softmax, get gradients)
               dense : N^2            (full GEMM, diagonal masked afterwards)
               exact : N * k
               ariad : N * k
  P_search : no-grad search scores, used only to choose neighbors
               dense : 0
               exact : N^2            (brute-force, counted at runtime)
               ariad : counted at runtime = total columns of the candidate
                       pool actually scored in AriadSeekerSingle.refresh()
                       (duplicate/self candidates are scored then masked, so
                       they ARE counted; disabled families are not
                       concatenated, so they are NOT counted)
  C_scored = P_search / N   (measured per-row search width; this is the
                             quantity to report, not a per-family formula)
  total_scores = P_attn + P_search  (all similarity scores computed)

  Estimated arithmetic FLOPs -- a MODEL, not a hardware measurement.
  Conventions: 1 multiply-add = 2 FLOPs; backward approximated as 2x forward
  for differentiable ops (so they count 3x); the no-grad search counts 1x:
      F = 3 * 2*N*d_in*(d_z + d_v)          # projections W_z, W_v
        + 3 * 2*P_attn*(d_z + d_v)          # attention scores + weighted V
        + 2*P_search*d_z                    # neighbor-search scoring
  Excluded: softmax, top-k selection, sort/dedup, gather/scatter, reverse-
  buffer construction and memory traffic -- precisely the parts that dominate
  the sparse paths' wall clock. F must therefore NOT be converted into a
  speed-up factor: dense runs one highly optimised GEMM, while the sparse
  paths are bound by indexing and irregular memory access.

Usage: python compare_dense_exact_ariad.py [--n-clusters 125] [--seed 0] [--epochs 150]
(N = n_clusters * GEN_CONFIG['cluster_size']; dataset generated if missing.)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ariad_single import AriadSeekerSingle, AriadSingleConfig
from ariad_train_single import (
    ARIAD_CONFIG,
    TRAIN_CONFIG,
    OneLayerSingleSpaceAttention,
    load_dataset,
    overlap_coverage_torch,
    recall_at_k_directed_torch,
    set_seed,
    topk_from_scores,
)
from ug_data_generate import GEN_CONFIG, dataset_root_name, generate_tau_sweep_datasets, save_tau_sweep_datasets

HERE = Path(__file__).resolve().parent
TAU = 0.5
K = ARIAD_CONFIG["k"]       # single k for all methods and for teacher match@k
METHODS = ["dense", "exact", "ariad"]
WARMUP_EPOCHS = 10          # excluded from timing statistics


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---------- neighbor finders ----------

class ExactTopkSeeker(nn.Module):
    """Brute-force exact top-k of Z Z^T (self excluded), row-chunked, no grad."""

    def __init__(self, k: int, score_chunk: int = 4096):
        super().__init__()
        self.k = k
        self.score_chunk = score_chunk
        self.last_refresh_stats = None

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        Z = Z.detach()
        n = Z.size(0)
        out = torch.empty(n, self.k, device=Z.device, dtype=torch.long)
        n_scored = 0
        with torch.no_grad():
            for s in range(0, n, self.score_chunk):
                e = min(s + self.score_chunk, n)
                scores = Z[s:e] @ Z.T
                n_scored += scores.numel()
                rows = torch.arange(e - s, device=Z.device)
                scores[rows, rows + s] = float("-inf")
                out[s:e] = scores.topk(self.k, dim=1).indices
        self.last_refresh_stats = {"n_scored": n_scored}
        return out


EXACT_EVAL = ExactTopkSeeker(K)   # evaluation-only exact top-k (not timed, not counted)


class TimedSeeker(nn.Module):
    """Wraps a neighbor finder; records wall time + scored count of training calls."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.last_time_s = 0.0
        self.last_n_scored = 0

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return self.inner(Z)
        sync(Z.device)
        t0 = time.perf_counter()
        out = self.inner(Z)
        sync(Z.device)
        self.last_time_s = time.perf_counter() - t0
        self.last_n_scored = int(self.inner.last_refresh_stats["n_scored"])
        return out


# ---------- models ----------

class DenseSingleSpaceAttention(nn.Module):
    """Same parameterisation as OneLayerSingleSpaceAttention, full attention."""

    def __init__(self, d_in: int, d_z: int, d_v_out: int):
        super().__init__()
        self.w_z = nn.Parameter(torch.empty(d_in, d_z))
        self.w_v = nn.Parameter(torch.empty(d_in, d_v_out))
        nn.init.xavier_uniform_(self.w_z)
        nn.init.xavier_uniform_(self.w_v)

    def forward(self, x: torch.Tensor):
        z = x @ self.w_z
        v = x @ self.w_v
        scores = (z @ z.T) / float(z.size(1)) ** 0.5
        scores = scores.masked_fill(torch.eye(z.size(0), dtype=torch.bool, device=x.device), float("-inf"))
        y_hat = F.softmax(scores, dim=1) @ v
        return y_hat, None, z


def build_model(method: str, d_in: int, d_z: int, d_v_out: int, k: int,
                ariad_overrides: Dict | None = None) -> nn.Module:
    if method == "dense":
        return DenseSingleSpaceAttention(d_in, d_z, d_v_out)
    ariad_cfg = AriadSingleConfig(**{**ARIAD_CONFIG, **(ariad_overrides or {}), "k": k})
    model = OneLayerSingleSpaceAttention(d_in, d_z, d_v_out, ariad_cfg)
    inner = AriadSeekerSingle(ariad_cfg) if method == "ariad" else ExactTopkSeeker(k, ariad_cfg.score_chunk)
    model.ariad_seeker = TimedSeeker(inner)   # seeker construction consumes no RNG -> identical W init
    return model


# ---------- cost model ----------

def est_flops(n: int, d_in: int, d_z: int, d_v: int, p_attn: int, p_search: int) -> float:
    return (3 * 2 * n * d_in * (d_z + d_v)
            + 3 * 2 * p_attn * (d_z + d_v)
            + 2 * p_search * d_z)


# ---------- training ----------

def train_one(method: str, data: Dict, device: torch.device, epochs: int, seed: int,
              ariad_overrides: Dict | None = None) -> List[Dict]:
    x, y, true_topk_idx = data["X"], data["Y"], data["true_topk_idx"]
    n, d_in = x.shape
    d_z = int(data["metadata"]["d_head"])
    d_v = y.shape[1]
    k = K
    if true_topk_idx.shape[1] != k:
        raise ValueError(f"teacher top-k width {true_topk_idx.shape[1]} != k={k}")

    set_seed(seed)
    model = build_model(method, d_in, d_z, d_v, k, ariad_overrides).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["lr"], weight_decay=TRAIN_CONFIG["weight_decay"])

    print(f"\n=== {method} (n={n}, k={k}) ===")
    records: List[Dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        t0 = time.perf_counter()
        opt.zero_grad()
        y_hat, _, _ = model(x)
        loss = F.mse_loss(y_hat, y)
        loss.backward()
        opt.step()
        sync(device)
        step_time = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")

        if method == "dense":
            p_attn, p_search, search_time = n * n, 0, 0.0
        else:
            p_attn, p_search, search_time = n * k, model.ariad_seeker.last_n_scored, model.ariad_seeker.last_time_s

        with torch.no_grad():
            model.eval()
            y_hat_eval, neighbors, z = model(x)
            mse = F.mse_loss(y_hat_eval, y).item()
            exh_idx = EXACT_EVAL(z)   # exact top-k of the post-step Z, all rows (chunked)
            if neighbors is None:   # dense: its neighbor set = exact top-k of the learned Z
                match_exhaustive = 1.0
                teacher_match = overlap_coverage_torch(true_topk_idx, exh_idx)
            else:
                match_exhaustive = recall_at_k_directed_torch(exh_idx, neighbors)
                teacher_match = overlap_coverage_torch(true_topk_idx, neighbors)
            del y_hat_eval, neighbors, z, exh_idx   # keep eval tensors out of the next step's peak memory

        records.append({
            "method": method, "n": n, "k": k, "seed": seed, "epoch": epoch, "mse": mse,
            "teacher_match": teacher_match, "match_exhaustive": match_exhaustive,
            "attn_scores": p_attn, "search_scores": p_search, "total_scores": p_attn + p_search,
            "c_scored": p_search / n,
            "est_flops": est_flops(n, d_in, d_z, d_v, p_attn, p_search),
            "step_time_s": step_time, "search_time_s": search_time, "peak_mem_mb": peak_mb,
        })
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:04d} mse={mse:.8f} teacher_match@{k}={teacher_match:.6f} "
                f"match_exhaustive={match_exhaustive:.6f} step={step_time * 1e3:.2f}ms "
                f"search={search_time * 1e3:.2f}ms"
            )
    return records


def summarize(records: List[Dict]) -> Dict:
    timed = [r for r in records if r["epoch"] > WARMUP_EPOCHS] or records
    last = records[-1]
    return {
        "method": last["method"], "n": last["n"], "k": last["k"], "seed": last["seed"], "epochs": last["epoch"],
        "final_mse": last["mse"], "final_teacher_match": last["teacher_match"],
        "final_match_exhaustive": last["match_exhaustive"],
        "attn_scores": last["attn_scores"], "search_scores": last["search_scores"],
        "total_scores": last["total_scores"], "c_scored": last["c_scored"],
        "est_arith_gflops": last["est_flops"] / 1e9,
        "step_ms_median": 1e3 * float(np.median([r["step_time_s"] for r in timed])),
        "search_ms_median": 1e3 * float(np.median([r["search_time_s"] for r in timed])),
        "peak_mem_mb": float(np.max([r["peak_mem_mb"] for r in timed])),
    }


def ensure_dataset(n_clusters: int, teacher_dense: bool) -> Path:
    n = n_clusters * GEN_CONFIG["cluster_size"]
    t_k = GEN_CONFIG["teacher_topk"]
    root = HERE / dataset_root_name(n, teacher_dense, t_k)
    ddir = root / f"tau_{TAU:g}"
    if not (ddir / "metadata.json").exists():
        cfg = {**GEN_CONFIG, "n_clusters": n_clusters, "teacher_dense": teacher_dense}
        datasets = generate_tau_sweep_datasets([TAU], **cfg)
        save_tau_sweep_datasets(datasets, root)
        print(f"generated {ddir}: gate={datasets[TAU]['gate']}")
    meta = json.loads((ddir / "metadata.json").read_text(encoding="utf-8"))
    got = bool(meta.get("teacher_dense", True))   # datasets predating the knob were all dense
    if got != teacher_dense or (not got and meta.get("teacher_topk") != t_k):
        raise ValueError(f"{ddir} has teacher_dense={got}, teacher_topk={meta.get('teacher_topk')}; "
                         f"requested teacher_dense={teacher_dense}, teacher_topk={t_k}")
    return ddir


def run() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clusters", type=int, default=125)
    ap.add_argument("--seed", type=int, default=TRAIN_CONFIG["seed"])
    ap.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    ap.add_argument("--teacher-dense", type=int, choices=[0, 1], default=int(GEN_CONFIG["teacher_dense"]),
                    help="1: dense global teacher; 0: sparse top-teacher_topk teacher (default from GEN_CONFIG)")
    args = ap.parse_args()
    teacher_dense = bool(args.teacher_dense)

    ddir = ensure_dataset(args.n_clusters, teacher_dense)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(ddir, device, teacher_k=K)
    n = data["X"].shape[0]
    t_tag = "" if teacher_dense else f"_sparseT{GEN_CONFIG['teacher_topk']}"
    tag = f"N{n}{t_tag}_k{K}_e{args.epochs}_seed{args.seed}"
    print(f"device={device} dataset={ddir} epochs={args.epochs} lr={TRAIN_CONFIG['lr']} seed={args.seed} "
          f"teacher={'dense' if teacher_dense else 'sparse top-' + str(GEN_CONFIG['teacher_topk'])}")

    all_records: List[Dict] = []
    summaries: List[Dict] = []
    for m in METHODS:
        try:
            recs = train_one(m, data, device, args.epochs, args.seed)
        except torch.cuda.OutOfMemoryError as e:
            print(f"!!! OOM: method={m} N={n}: {str(e).splitlines()[0]}")
            summaries.append({"method": m, "n": n, "k": K, "seed": args.seed, "status": "OOM"})
            torch.cuda.empty_cache()
            continue
        all_records.extend(recs)
        summaries.append({**summarize(recs), "status": "ok"})

    # dense reference from the cost model (valid even if dense itself OOMs)
    d_in, d_z, d_v = data["X"].shape[1], int(data["metadata"]["d_head"]), data["Y"].shape[1]
    dense_scores = n * n
    dense_gflops = est_flops(n, d_in, d_z, d_v, n * n, 0) / 1e9
    for s in summaries:
        if s["status"] == "ok":
            s["scores_vs_dense"] = s["total_scores"] / dense_scores
            s["flops_vs_dense"] = s["est_arith_gflops"] / dense_gflops

    log_csv = HERE / f"compare_dense_exact_ariad_{tag}_log.csv"
    sum_csv = HERE / f"compare_dense_exact_ariad_{tag}_summary.csv"
    for path, rows in [(log_csv, all_records), (sum_csv, summaries)]:
        if not rows:
            continue
        fields = list(dict.fromkeys(key for r in rows for key in r))
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(rows)
    print(f"\nper-epoch log: {log_csv}\nsummary:       {sum_csv}")

    ok = [s for s in summaries if s["status"] == "ok"]
    print(f"\n===== Cost per training step (N={n}; timing = median over epochs > {WARMUP_EPOCHS}) =====")
    hdr = (f"{'method':<7}{'attn(grad)':>14}{'search(no-grad)':>16}{'C_scored':>9}{'total':>14}{'vs dense':>9}"
           f"{'est.arith GFLOPs':>17}{'vs dense':>9}{'ms/step':>9}{'search ms':>10}{'peak MB':>9}")
    print(hdr)
    print("-" * len(hdr))
    for s in ok:
        print(
            f"{s['method']:<7}{s['attn_scores']:>14,}{s['search_scores']:>16,}{s['c_scored']:>9.1f}"
            f"{s['total_scores']:>14,}{s['scores_vs_dense']:>9.4f}"
            f"{s['est_arith_gflops']:>17.4f}{s['flops_vs_dense']:>9.4f}"
            f"{s['step_ms_median']:>9.2f}{s['search_ms_median']:>10.2f}{s['peak_mem_mb']:>9.1f}"
        )
    print("\n===== Quality (final epoch) =====")
    for s in summaries:
        if s["status"] != "ok":
            print(f"{s['method']:<7} {s['status']}")
            continue
        print(f"{s['method']:<7} mse={s['final_mse']:.8f} teacher_match@{K}={s['final_teacher_match']:.6f} "
              f"match_exhaustive={s['final_match_exhaustive']:.6f}")


if __name__ == "__main__":
    run()
