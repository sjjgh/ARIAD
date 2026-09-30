"""Standalone highlight figure: watch ARIAD's maintained k-NN graph converge to the
teacher's k-NN graph, at a scale small enough (N~100) to draw every point and edge.

Reuses the REAL, already-validated single-space pipeline as-is (no reimplementation,
no modification of any existing file):
  - ug_data_generate.generate_tau_first_synthetic_dataset  (the real data generator)
  - ariad_train_single.OneLayerSingleSpaceAttention          (the real model: W_z, W_v,
    AriadSeekerSingle -- the same class trained in ariad_train_single.py / used by
    compare_dense_exact_ariad.py)
  - ariad_single.AriadSeekerSingle / AriadSingleConfig        (the real search module)
Training here IS the real ARIAD mechanism end to end (not a Dense stand-in): the model
learns Z = X @ Wz and V = X @ Wv, AriadSeekerSingle maintains Z's neighbor graph from a
random start, and backprop flows through the attention weights over ARIAD's CURRENT
graph every step -- identical mechanics to ariad_train_single.train(). We only add: the
small-scale knobs below, a snapshot hook around the same model.eval() forward pass
ariad_train_single.py already uses for its epoch diagnostics, and the plotting code.

Tuning notes (see session notes for the full search):
  - d_g=2 for the teacher's own geometry could not satisfy the generator's sanity gate
    at any jitter/tau/cluster-count (2D limits how well many clusters separate, and the
    gate's fixed stability-perturbation scale assumes the generator's usual ~0.10
    jitter regime) -- so the teacher lives in d_g=10, plotted via PCA for the teacher
    panel only (real positions, not schematic).
  - cluster_size must equal K+1 exactly: otherwise the teacher's own true top-K is only
    a SUBSET of a larger same-cluster pool, and two independently-computed subsets of
    that pool overlap by chance alone (e.g. ~55% for two random 5-of-9 subsets) --
    this silently caps "recall" well below 100% for ANY model, including a
    hand-verified-perfect one. cluster_size=K+1 removes that ambiguity entirely.
  - the student embedding dim (D_Z) matters a lot for how fast/cleanly ARIAD's sparse,
    N=100-scale training converges: d_z=2 plateaus in the 15-30% range even after 1500
    epochs (confirmed NOT a capacity limit -- a hand-crafted, well-separated 2D layout
    scores 99%+ -- it is an optimization/local-minimum issue). d_z=32 (matching the
    real single-space experiments' own k=32) converges far better. We plot with the raw
    D_Z-dim Z PCA-projected to 2D -- exactly the same technique used to visualize
    learned embeddings in ML papers generally.
  - K (search/aggregation width) also matters: raising K from 5 to 16 (with
    cluster_size raised to 17 to keep the no-ambiguity property) roughly doubled the
    achieved recall for the same epoch budget -- more same-cluster candidates per row
    means a denser gradient signal per step, which this session's other experiments
    also found (ARIAD's sparse per-row signal is comparatively weak at N~100 vs the
    real experiments' N=10,000+).

Orientation note (student panels): the loss only supervises dot-product (routing)
structure, invariant to a global rotation/reflection of Z, so PCA's chosen axes (and a
panel's absolute orientation) are arbitrary and can differ panel to panel -- what
converges is the GRAPH, shown via point colour (true cluster id, consistent across
panels) and edge colour (green = also a teacher top-K edge; red = not), plus recall@K
annotated per panel.

Usage: python highlight_kNN_convergence.py
Output: kNN_convergence.png / .pdf in this folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SINGLE_SPACE_DIR = Path(__file__).resolve().parent.parent   # -> ariad_single_space
sys.path.insert(0, str(SINGLE_SPACE_DIR))
import ug_data_generate as datagen                                                    # noqa: E402
from ariad_single import AriadSingleConfig                                           # noqa: E402
from ariad_train_single import OneLayerSingleSpaceAttention, recall_at_k_directed_torch, set_seed  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

# ---- toy-task knobs -------------------------------------------------------------
# TRAIN at a real scale (N~2000): this session separately found that recall is capped
# well below what the real experiments reach when N is tiny (~100), REGARDLESS of
# embedding dimension -- W_z/W_v are shared across all N rows, and at N~100 each
# gradient step only averages ~100 rows' worth of signal, noisy enough to strand the
# shared parameters in a mediocre solution. At N~2000 with the SAME dims, recall
# reaches ~98% by epoch 200 (vs ~85-94%, seed-dependent, at N=102). So: train on the
# full N_CLUSTERS*CLUSTER_SIZE, but only DRAW a handful of whole clusters (DISPLAY_
# CLUSTERS) for legibility -- picking whole clusters (not a random sample of individual
# points) matters because true neighbors are almost all same-cluster here
# (cluster_size = K + 1 exactly), so a random points-only sample would sever nearly
# every edge; a whole-cluster sample keeps them intact.
N_CLUSTERS, CLUSTER_SIZE = 118, 17   # N = 2006; cluster_size = K + 1 exactly (see module
                                      # docstring -- removes the same-cluster subset ambiguity)
DISPLAY_CLUSTERS = 6                 # how many whole clusters (~102 points) to actually draw
D_G, D_V = 10, 8                     # teacher routing-geometry / value dims
D_Z = 32                             # student embedding dim -- matches the real single-space
                                      # experiments' own k=32 (plotted via PCA-to-2D, not literally 2D)
K = 16
JITTER = 0.10                        # the real generator's own default
TAU = 0.3
LR = 0.02
EPOCHS = 160
SNAP_EPOCHS = [0, 40, 80, 120, 160]   # 160 is already ~converged (recall 0.96, vs 0.98 at 200)
SEED = 0

ARIAD_CFG = AriadSingleConfig(k=K, k_rev=K, c_o2=2 * K, c_i1=K, c_oi=0, c_rand=0,
                              score_chunk=4096, use_tiebreak=False, compute_churn_stats=False)


def build_dataset():
    ds = datagen.generate_tau_first_synthetic_dataset(
        n_clusters=N_CLUSTERS, cluster_size=CLUSTER_SIZE, d_g=D_G, d_v=D_V,
        cluster_jitter=JITTER, k_eval=K, k_obs=K, tau=TAU, rho_g=1.0, seed=SEED,
        teacher_dense=False, teacher_topk=K,   # sparse teacher: Y aggregates EXACTLY its own
                                                # reported top-K, so true_topk_idx and Y are
                                                # defined over the identical set (no dense-vs-
                                                # reported-topk mismatch). Matches the convention
                                                # this session settled on in ariad_qk_space.
    )
    print(f"teacher gate: {ds['gate']}")
    return ds


def train_and_snapshot(ds):
    device = torch.device("cpu")
    set_seed(SEED)
    x = torch.from_numpy(ds["X"]).to(device)
    y = torch.from_numpy(ds["Y"]).to(device)
    true_topk_idx = torch.from_numpy(ds["true_topk_idx"]).to(device)
    n, d_in = x.shape

    model = OneLayerSingleSpaceAttention(d_in=d_in, d_z=D_Z, d_v_out=y.shape[1], ariad_cfg=ARIAD_CFG).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    snapshots = {}

    def snap(step):
        model.eval()
        with torch.no_grad():
            y_hat, neighbors, z = model(x)
            mse = F.mse_loss(y_hat, y).item()
            recall = recall_at_k_directed_torch(true_topk_idx, neighbors)
        snapshots[step] = {"z": z.numpy().copy(), "graph": neighbors.numpy().copy(),
                           "mse": mse, "recall": recall}
        model.train()

    snap(0)
    for step in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        y_hat, _, _ = model(x)
        F.mse_loss(y_hat, y).backward()
        opt.step()
        if step in SNAP_EPOCHS:
            snap(step)

    return ds["lab"], ds["true_topk_idx"], snapshots


def pca_2d(mat: np.ndarray) -> np.ndarray:
    """Top-2 principal components, for visualization only -- never used for anything
    that affects training or the recall metric."""
    centered = mat - mat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


# ---------------------------------------------------------------- plotting
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
EDGE_OK, EDGE_BAD = "#1baf7a", "#e34948"      # matches teacher top-k (green) / doesn't (red)
CMAP = plt.get_cmap("tab10")                  # 6 clusters: a qualitative colormap for visual grouping only,
                                              # not a data-encoding chart -- CVD validation doesn't apply here.


def pick_display_clusters(lab: np.ndarray) -> np.ndarray:
    """Global row indices of the first DISPLAY_CLUSTERS whole clusters -- picking WHOLE
    clusters (not a random sample of individual points) matters: true neighbors here are
    almost all same-cluster (cluster_size = K + 1 exactly), so a points-only sample would
    sever nearly every edge before it could even be drawn."""
    keep_clusters = np.arange(DISPLAY_CLUSTERS)
    return np.where(np.isin(lab, keep_clusters))[0]


def draw_panel(ax, pos, lab, display_idx, edges_full, true_topk_full, title, recall, mse):
    """pos/lab: DISPLAY_CLUSTERS-subset only (already indexed). display_idx: global row id
    for each local row of pos/lab. edges_full/true_topk_full: FULL [N, K] arrays (global
    ids) -- an edge is drawn only when BOTH its endpoints are in display_idx; edges leaving
    the drawn subset are simply not rendered (recall/mse annotated are still the FULL-N
    values, computed in train_and_snapshot, not recomputed on this subset)."""
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")

    local_of = {int(g): i for i, g in enumerate(display_idx)}
    true_set = {int(g): set(true_topk_full[g].tolist()) for g in display_idx}
    for i, gi in enumerate(display_idx):
        for gj in edges_full[gi]:
            gj = int(gj)
            if gj == gi or gj not in local_of:
                continue
            j = local_of[gj]
            ok = gj in true_set[int(gi)]
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color=EDGE_OK if ok else EDGE_BAD, linewidth=0.45,
                    alpha=0.6 if ok else 0.28, zorder=1)

    ax.scatter(pos[:, 0], pos[:, 1], c=[CMAP(int(c) % 10) for c in lab], s=34,
              edgecolors=SURFACE, linewidths=0.7, zorder=2)

    ax.set_title(title, fontsize=11, color=INK, loc="center", pad=15)
    if recall is not None:
        ax.text(0.5, 1.01, f"full-N matches teacher: {100*recall:.0f}%  (mse {mse:.4f})", transform=ax.transAxes,
               fontsize=8.5, color=MUTED, ha="center", va="bottom")


def main():
    ds = build_dataset()
    lab, true_topk_np, snaps = train_and_snapshot(ds)
    n = len(lab)
    display_idx = pick_display_clusters(lab)
    lab_disp = lab[display_idx]
    g_vis = pca_2d(ds["G"][display_idx])

    fig, axes = plt.subplots(1, len(SNAP_EPOCHS) + 1, figsize=(3.9 * (len(SNAP_EPOCHS) + 1), 4.2),
                             facecolor=SURFACE)
    draw_panel(axes[0], g_vis, lab_disp, display_idx, true_topk_np, true_topk_np,
              "Teacher (ground truth)", None, None)
    for ax, ep in zip(axes[1:], SNAP_EPOCHS):
        s = snaps[ep]
        pos = pca_2d(s["z"][display_idx])
        draw_panel(ax, pos, lab_disp, display_idx, s["graph"], true_topk_np,
                  f"ARIAD, epoch {ep}", s["recall"], s["mse"])

    fig.suptitle(f"ARIAD's maintained k-NN graph converges to the teacher's  "
                f"(trained at N={n:,}, k={K}, sparse teacher; {len(display_idx)} pts / "
                f"{DISPLAY_CLUSTERS} clusters shown)",
                x=0.01, y=0.99, ha="left", va="top", fontsize=13.5, color=INK)
    fig.text(0.01, 0.01,
             f"Training uses the FULL N={n:,} tokens (real single-space ARIAD recipe, unmodified) -- this session "
             "found recall plateaus well below the real experiments' level\nwhen N is only ~100, regardless of "
             f"embedding dimension, since W_z/W_v are shared across all rows and a tiny N gives a noisy shared-"
             f"parameter gradient.\nOnly {DISPLAY_CLUSTERS} whole clusters ({len(display_idx)} of {n:,} points) "
             "are drawn for legibility; edges leaving this subset are not rendered, but the annotated "
             "recall/mse\nare the FULL-dataset values. Each edge is one directed ARIAD neighbor link: green if "
             f"also one of the teacher's true top-{K} neighbors, red if not.\nModel, search and training loop are "
             f"exactly ariad_train_single.py's; the student's actual embedding is {D_Z}-D, PCA-projected to 2D "
             "(fit on the drawn subset only,\nindependently per panel -- absolute orientation differs panel to "
             "panel; only the graph -- point colour + edge colour -- is the signal to read).",
             fontsize=7.3, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.16, 1, 0.88))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"kNN_convergence.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("recall@%d by epoch (full N=%d): %s" % (K, n, {ep: round(snaps[ep]["recall"], 3) for ep in SNAP_EPOCHS}))
    print("mse by epoch: %s" % ({ep: round(snaps[ep]["mse"], 4) for ep in SNAP_EPOCHS}))
    print(f"saved {OUT_DIR / 'kNN_convergence.png'} / .pdf")


if __name__ == "__main__":
    main()
