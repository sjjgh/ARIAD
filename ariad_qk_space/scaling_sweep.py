"""Sweep N for the dual-space (QK) experiment: Dense vs Exact top-k vs ARIAD.

Reproduces the single-space paper experiment (Table 1, Fig. 1, Fig. 2) on the QK pipeline
(ariad_train_qk.py + ug_data_generate.py). Setup, deviations and how to read the outputs are in
result/README.md.

Usage:
    python scaling_sweep.py --all                # datasets + all runs (skips finished ones)
    python scaling_sweep.py --gen N              # only generate the dataset for N
    python scaling_sweep.py --run N METHOD SEED  # one run, writes result/raw/N{N}_{METHOD}_s{SEED}.json
Then: python make_results.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RESULT_DIR = HERE / "result"
RAW_DIR = RESULT_DIR / "raw"
LOG_DIR = RAW_DIR / "logs"

NS = [1008, 2000, 5008, 10000, 20000]
METHODS = ["dense", "exact_topk", "ariad"]
SEEDS = [0, 1, 2]
EPOCHS = 500
CLUSTER_SIZE = 16          # N/16 clusters of 16 points, like the paper
DYNAMICS_N, DYNAMICS_SEED = 10000, 0   # Fig. 2 is logged every epoch on this run
DATA_SEED = 0              # one fixed dataset per N; the training seed only changes init/graph noise


def configure(m, n: int, method: str, seed: int, epochs: int, print_every: int) -> None:
    assert n % CLUSTER_SIZE == 0, f"N={n} must be a multiple of {CLUSTER_SIZE}"
    m.GEN_CONFIG.update(
        n_clusters=n // CLUSTER_SIZE, cluster_size=CLUSTER_SIZE, d_g=64, d_v=64,
        cluster_jitter=0.05, k_eval=32, k_obs=16, rho_g=1.0, seed=DATA_SEED,
        teacher_dense=False, teacher_topk=32, tau=0.01, teacher_scale="fixed",
    )
    m.QK_CONFIG["k_final"] = 32
    m.MODEL_CONFIG.update(bond_QK=False, freeze_wk_to_oracle=False, value_mode="fixed_v", use_oracle_init=False)
    m.DENSE_ATTENTION_CONFIG["enabled"] = method == "dense"
    m.EXACT_QK_CONFIG["enabled"] = method == "exact_topk"
    m.TRAIN_CONFIG.update(seed=seed, epochs=epochs, lr=1e-2, weight_decay=0.0,
                          print_every=print_every, cosine_decay=False)   # paper: plain full-batch Adam


def out_path(n: int, method: str, seed: int) -> Path:
    return RAW_DIR / f"N{n}_{method}_s{seed}.json"


def do_gen(n: int) -> None:
    import ariad_train_qk as m
    configure(m, n, "ariad", 0, EPOCHS, EPOCHS)
    m.ensure_dataset(m.DATA_CONFIG["dataset_root"] / m.dataset_name())


def do_run(n: int, method: str, seed: int) -> None:
    import torch
    import ariad_train_qk as m
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    track = (n == DYNAMICS_N and seed == DYNAMICS_SEED)
    configure(m, n, method, seed, EPOCHS, 1 if track else EPOCHS)
    rec = {"n": n, "method": method, "seed": seed, "epochs": EPOCHS, "tracked": track}
    t0 = time.time()
    try:
        out = m.train()
        hist = out["history"]
        last = hist[-1]
        rec.update(status="ok", final=last, k_eval=out["k_eval"], seconds=time.time() - t0)
        if track:
            rec["history"] = hist
    except torch.cuda.OutOfMemoryError as e:
        rec.update(status="OOM", error=str(e).splitlines()[0], seconds=time.time() - t0)
    out_path(n, method, seed).write_text(json.dumps(rec), encoding="utf-8")
    print(f"wrote {out_path(n, method, seed)}  status={rec['status']}")


def launch(n: int, method: str, seed: int):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_DIR / f"N{n}_{method}_s{seed}.log", "w", encoding="utf-8")
    return subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "--run", str(n), method, str(seed)],
                            stdout=log, stderr=subprocess.STDOUT)


def run_wave(jobs, parallel: int) -> None:
    running = []
    jobs = list(jobs)
    while jobs or running:
        while jobs and len(running) < parallel:
            n, method, seed = jobs.pop(0)
            print(f"[{time.strftime('%H:%M:%S')}] start N={n} {method} seed={seed}", flush=True)
            running.append((launch(n, method, seed), (n, method, seed)))
        time.sleep(2)
        still = []
        for p, key in running:
            if p.poll() is None:
                still.append((p, key))
            else:
                print(f"[{time.strftime('%H:%M:%S')}] done  N={key[0]} {key[1]} seed={key[2]} rc={p.returncode}", flush=True)
        running = still


def do_all() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for n in NS:
        pending = [(n, mth, s) for mth in METHODS for s in SEEDS if not out_path(n, mth, s).exists()]
        if not pending:
            continue
        print(f"=== N={n}: generating dataset", flush=True)
        subprocess.run([sys.executable, str(Path(__file__)), "--gen", str(n)], check=True)
        # Dense at large N materializes several N x N tensors: keep it alone on the GPU.
        big = n >= 10000
        sparse = [j for j in pending if j[1] != "dense"]
        dense = [j for j in pending if j[1] == "dense"]
        run_wave(sparse, parallel=3 if big else 9)
        run_wave(dense, parallel=1 if big else 9)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gen", type=int)
    ap.add_argument("--run", nargs=3, metavar=("N", "METHOD", "SEED"))
    a = ap.parse_args()
    if a.gen:
        do_gen(a.gen)
    elif a.run:
        do_run(int(a.run[0]), a.run[1], int(a.run[2]))
    elif a.all:
        do_all()
    else:
        ap.print_help()
