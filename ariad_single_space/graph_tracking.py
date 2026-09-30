"""graph_tracking.py

Does ARIAD work on a MOVING target? Audit every epoch of a training run
against the exact top-k graph of that run's CURRENT student Z:

  E_t      exact top-k of the post-step Z_t (all N rows, self excluded)
  G_t      ARIAD graph after the step-t refresh (the one attended over;
           ARIAD runs only)

  exact_churn_t  = 1 - mean_i |E_t(i) ∩ E_{t-1}(i)| / k   (target moves)
  ariad_churn_t  = 1 - mean_i |G_t(i) ∩ G_{t-1}(i)| / k   (graph updates)
  track_recall_t = mean_i |G_t(i) ∩ E_t(i)| / k           (= match_exhaustive)
  track_recall_pre_t = mean_i |G_t(i) ∩ E_{t-1}(i)| / k   (vs the pre-step Z
                   that the refresh actually scored with; removes the 1-step lag)
  exact_vs_final_t   = mean_i |E_t(i) ∩ E_final(i)| / k   (how far the target
                   still is from where it ends up; every SNAP_EVERY epochs)

Runs audited: ARIAD at budget scales SCALES, plus the Dense and Exact top-k
runs. For Exact, E_t is the graph it attends over; for Dense, E_t is the
top-k of its learned scores (its implicit neighbor graph -- Dense itself
attends to all nodes). Graph-only columns are NaN for Dense/Exact.

The audit is exact (all rows, not sampled), runs outside the timed training
step and is not counted in any training cost. Training itself is identical to
compare_dense_exact_ariad.train_one (same seed, init, optimizer, eval pass),
so every run reproduces that script's numbers.

Usage: python graph_tracking.py [--n-clusters 625] [--epochs 500] [--seed 0] [--teacher-dense 0]
Output: graph_tracking_N<N><t>_k32_e<E>_seed<S>.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from ariad_train_single import TRAIN_CONFIG, load_dataset, overlap_coverage_torch, set_seed
from budget_sweep import budget_overrides
from compare_dense_exact_ariad import EXACT_EVAL, GEN_CONFIG, K, build_model, ensure_dataset

HERE = Path(__file__).resolve().parent
SCALES = [1.0, 2.0]
SNAP_EVERY = 10
NAN = float("nan")


def overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    return overlap_coverage_torch(a, b)   # mean_i |a(i) ∩ b(i)| / k


def run(method: str, s: float | None, data, device, epochs: int, seed: int):
    x, y, teacher_idx = data["X"], data["Y"], data["true_topk_idx"]
    n, d_in = x.shape
    d_z = int(data["metadata"]["d_head"])
    ov = budget_overrides(s) if method == "ariad" else None

    set_seed(seed)
    model = build_model(method, d_in, d_z, y.shape[1], K, ov).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG["lr"], weight_decay=TRAIN_CONFIG["weight_decay"])

    rows = []
    snaps = {}   # epoch -> exact graph E_t (CPU, int32)
    prev_e = prev_g = None
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        y_hat, _, _ = model(x)
        F.mse_loss(y_hat, y).backward()
        opt.step()

        with torch.no_grad():   # audit: not part of training cost
            model.eval()
            y_eval, g, z = model(x)
            mse = F.mse_loss(y_eval, y).item()
            e = EXACT_EVAL(z)
            rec = {"method": method, "scale": s if method == "ariad" else NAN,
                   "c_scored": model.ariad_seeker.last_n_scored / n if method != "dense" else NAN,
                   "epoch": epoch, "mse": mse,
                   "exact_churn": 1.0 - overlap(prev_e, e) if prev_e is not None else NAN}
            if method == "ariad":
                g = g.clone()
                rec.update(
                    teacher_recall=overlap(teacher_idx, g),
                    track_recall=overlap(e, g),
                    track_recall_pre=overlap(prev_e, g) if prev_e is not None else NAN,
                    ariad_churn=1.0 - overlap(prev_g, g) if prev_g is not None else NAN,
                )
                prev_g = g
            else:   # the run's own graph is E_t itself
                rec.update(teacher_recall=overlap(teacher_idx, e), track_recall=NAN,
                           track_recall_pre=NAN, ariad_churn=NAN)
            prev_e = e
            if epoch == 1 or epoch % SNAP_EVERY == 0:
                snaps[epoch] = e.to("cpu", torch.int32)
        rows.append(rec)
        if epoch in (1, 10, 50, 100, 200, 300, 500):
            print(f"{method} s={s} epoch={epoch:4d} mse={mse:.3e} exact_churn={rec['exact_churn']:.4f} "
                  f"ariad_churn={rec['ariad_churn']:.4f} track={rec['track_recall']:.4f}")

    final_e = prev_e.to("cpu", torch.int32)
    for r in rows:
        r["exact_vs_final"] = overlap(snaps[r["epoch"]], final_e) if r["epoch"] in snaps else NAN
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clusters", type=int, default=625)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher-dense", type=int, choices=[0, 1], default=0)
    args = ap.parse_args()
    teacher_dense = bool(args.teacher_dense)

    ddir = ensure_dataset(args.n_clusters, teacher_dense)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(ddir, device, teacher_k=K)
    n = data["X"].shape[0]
    t_tag = "" if teacher_dense else f"_sparseT{GEN_CONFIG['teacher_topk']}"

    rows = []
    for s in SCALES:
        rows += run("ariad", s, data, device, args.epochs, args.seed)
    for method in ("dense", "exact"):
        rows += run(method, None, data, device, args.epochs, args.seed)

    path = HERE / f"graph_tracking_N{n}{t_tag}_k{K}_e{args.epochs}_seed{args.seed}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
