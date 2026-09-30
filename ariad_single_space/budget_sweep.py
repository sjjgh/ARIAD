"""budget_sweep.py

ARIAD candidate-budget sweep at fixed N: every expansion source is scaled by
the same factor s, relative to ARIAD_CONFIG:

    c_o2  = round(80 * s)     O(O(i)) 2-hop forward
    c_i1  = round(32 * s)     I(i) reverse 1-hop
    k_rev = round(32 * s)     reverse-buffer width, scaled with c_i1 so the
                              I(i) width min(c_i1, k_rev) really scales
    c_oi, c_rand stay 0       (0 in the base config)
    O(i) (the current k=32 neighbors) is NOT scaled -- it is the graph
    itself, unconditionally kept, not an expansion budget.

  => C_scored = k + c_o2 + min(c_i1, k_rev), measured at runtime (n_scored/N).

Dense and Exact do not depend on this budget; their reference values come
from the existing compare_dense_exact_ariad run on the same dataset / seed.

Usage: python budget_sweep.py [--n-clusters 625] [--epochs 500] [--seed 0] [--teacher-dense 0]
Outputs: budget_sweep_N<N><t>_k32_e<E>_seed<S>_{log,summary}.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch

from ariad_train_single import ARIAD_CONFIG, load_dataset
from compare_dense_exact_ariad import K, GEN_CONFIG, ensure_dataset, summarize, train_one

HERE = Path(__file__).resolve().parent
SCALES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]


def budget_overrides(s: float) -> Dict:
    return {
        "c_o2": max(1, round(ARIAD_CONFIG["c_o2"] * s)),
        "c_i1": max(1, round(ARIAD_CONFIG["c_i1"] * s)),
        "k_rev": max(1, round(ARIAD_CONFIG["k_rev"] * s)),
        "c_oi": 0,
        "c_rand": 0,
    }


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
    tag = f"N{n}{t_tag}_k{K}_e{args.epochs}_seed{args.seed}"
    print(f"device={device} dataset={ddir} scales={SCALES}")

    all_records: List[Dict] = []
    summaries: List[Dict] = []
    for s in SCALES:
        ov = budget_overrides(s)
        print(f"\n##### scale={s}: {ov}")
        recs = train_one("ariad", data, device, args.epochs, args.seed, ariad_overrides=ov)
        for r in recs:
            r.update(scale=s, **ov)
        all_records.extend(recs)
        summaries.append({"scale": s, **ov, **summarize(recs)})

    for path, rows in [(HERE / f"budget_sweep_{tag}_log.csv", all_records),
                       (HERE / f"budget_sweep_{tag}_summary.csv", summaries)]:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"saved {path}")

    print(f"\n{'scale':>6}{'c_o2':>6}{'c_i1':>6}{'C_scored':>10}{'mse':>12}{'recall':>9}{'match_exh':>10}{'GFLOPs':>9}{'ms/step':>9}")
    for s in summaries:
        print(f"{s['scale']:>6}{s['c_o2']:>6}{s['c_i1']:>6}{s['c_scored']:>10.0f}{s['final_mse']:>12.3e}"
              f"{s['final_teacher_match']:>9.4f}{s['final_match_exhaustive']:>10.4f}"
              f"{s['est_arith_gflops']:>9.3f}{s['step_ms_median']:>9.2f}")


if __name__ == "__main__":
    main()
