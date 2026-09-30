"""plot_results.py

Figures for the Dense / Exact top-k / ARIAD comparison.

  Figure 1: N vs estimated arithmetic FLOPs per training step (log-log),
            from the per-N summary CSVs of compare_dense_exact_ariad.py.
  Figure 2: training dynamics at fixed N, two panels sharing a log-epoch axis:
            (a) per-epoch neighbor replacement of each method's own graph in its
                own training run (graph_tracking.py);
            (b) MSE of Dense / Exact / ARIAD every --every epochs
                (compare_dense_exact_ariad.py log).
  Figure 3: ARIAD candidate-budget sweep at fixed N (budget_sweep.py).

Usage: python plot_results.py [--fig 1|2|3|all] [--teacher sparse|dense] [--k 32] [--epochs 500]
                              [--seed 0] [--n 10000] [--every 10]
Outputs: result/<name>.png and .pdf
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "result"
EXPECTED_N = [1008, 2000, 5008, 10000, 20000]

# Fixed categorical order (colour follows the method, never its rank);
# the first three slots of the reference palette validate all-pairs for CVD.
STYLE = {
    "dense": {"label": "Dense", "color": "#2a78d6", "marker": "o"},
    "exact": {"label": "Exact top-k", "color": "#eb6834", "marker": "s"},
    "ariad": {"label": "ARIAD", "color": "#1baf7a", "marker": "^"},
}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def load_summaries(teacher: str, k: int, epochs: int, seed: int) -> pd.DataFrame:
    t_tag = "_sparseT32" if teacher == "sparse" else ""
    pattern = str(HERE / f"compare_dense_exact_ariad_N*{t_tag}_k{k}_e{epochs}_seed{seed}_summary.csv")
    files = [f for f in glob.glob(pattern) if teacher == "sparse" or "_sparseT" not in f]
    if not files:
        raise FileNotFoundError(f"no summaries match {pattern}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["status"] == "ok"].sort_values(["method", "n"])
    missing = sorted(set(EXPECTED_N) - set(df["n"]))
    if missing:
        print(f"warning: missing N={missing} for teacher={teacher}")
    return df


def style_axes(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(False, which="minor")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, which="both", labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {OUT_DIR / name}.png/.pdf")


def fig_flops_vs_n(df: pd.DataFrame, teacher: str, name: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    style_axes(ax)

    for method in ("dense", "exact", "ariad"):
        d = df[df["method"] == method]
        s = STYLE[method]
        ax.plot(d["n"], d["est_arith_gflops"], color=s["color"], linewidth=2,
                marker=s["marker"], markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
                label=s["label"], zorder=3)
        # direct label at the line end, in ink (not series colour)
        last = d.iloc[-1]
        ax.annotate(f"{s['label']}  {last['est_arith_gflops']:.3g}", (last["n"], last["est_arith_gflops"]),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ns = sorted(df["n"].unique())
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n:,}" for n in ns])
    ax.set_xlim(ns[0] * 0.85, ns[-1] * 2.4)   # room for end labels
    ax.set_xlabel("N (number of tokens)")
    ax.set_ylabel("Estimated arithmetic GFLOPs per training step")
    t_desc = "sparse top-32 teacher" if teacher == "sparse" else "dense teacher"
    ax.set_title(f"Estimated training FLOPs vs N  (k=32, {t_desc})", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)

    # fitted log-log slopes, as a footnote
    slopes = []
    for method in ("dense", "exact", "ariad"):
        d = df[df["method"] == method]
        slope = np.polyfit(np.log(d["n"]), np.log(d["est_arith_gflops"]), 1)[0]
        slopes.append(f"{STYLE[method]['label']} {slope:.2f}")
    fig.text(0.01, 0.01, "Log-log slope: " + ", ".join(slopes) + ".\n"
             "Estimated arithmetic FLOPs (cost model, not measured); excludes softmax, top-k, dedup and memory traffic.",
             fontsize=7, color=MUTED, ha="left", va="bottom")

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, name)


def load_log(teacher: str, n: int, k: int, epochs: int, seed: int) -> pd.DataFrame:
    t_tag = "_sparseT32" if teacher == "sparse" else ""
    path = HERE / f"compare_dense_exact_ariad_N{n}{t_tag}_k{k}_e{epochs}_seed{seed}_log.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fig_training_dynamics(tr: pd.DataFrame, lg: pd.DataFrame, teacher: str, n: int, every: int,
                          name: str) -> None:
    """(a) per-epoch neighbor replacement of each method's own graph in its own training run
    (Dense: top-k of its learned scores; Exact: its exact top-k; ARIAD: its maintained graph at the
    default budget); (b) MSE of the three methods every `every` epochs."""
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(6.4, 6.4), dpi=150, sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1.25]})
    fig.patch.set_facecolor("#fcfcfb")
    ariad = tr[(tr["method"] == "ariad") & (tr["scale"] == 1.0)].sort_values("epoch")
    c_base = int(round(ariad["c_scored"].iloc[0]))
    churn = {
        "dense": (tr[tr["method"] == "dense"].sort_values("epoch"), "exact_churn",
                  "Dense (top-32 of its learned scores)"),
        "exact": (tr[tr["method"] == "exact"].sort_values("epoch"), "exact_churn",
                  "Exact top-k (its exact top-32 graph)"),
        "ariad": (ariad, "ariad_churn", f"ARIAD (maintained graph, $C_{{\\rm scored}}$={c_base})"),
    }

    # (a) replacement rates, each method's own graph in its own run
    style_axes(ax_a)
    for method, (d, col, label) in churn.items():
        ax_a.plot(d["epoch"], 100 * d[col], color=STYLE[method]["color"], linewidth=1.6, label=label)
    ax_a.set_ylabel("Neighbors replaced\nper epoch (%)")
    ax_a.set_ylim(bottom=0)
    ax_a.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    ax_a.set_title("(a) The neighbor graph changes during training", loc="left", fontsize=10, color=INK)

    # (b) MSE, sampled at epoch 1 and every `every` epochs (epoch 1 anchors the curve on the log axis)
    style_axes(ax_b)
    ends = []
    for method in ("dense", "exact", "ariad"):
        d = lg[(lg["method"] == method) & ((lg["epoch"] % every == 0) | (lg["epoch"] == 1))].sort_values("epoch")
        s = STYLE[method]
        ax_b.plot(d["epoch"], d["mse"], color=s["color"], linewidth=2, marker=s["marker"], markersize=4,
                  markevery=[0, 1, 2, 5, 10, 20, 50], markeredgecolor="#fcfcfb", markeredgewidth=1,
                  label=s["label"], zorder=3)
        ends.append((method, d.iloc[-1]))
    # end labels, spread apart vertically (log10 units) so near-equal finals don't collide
    ends.sort(key=lambda t: t[1]["mse"])
    label_y = [np.log10(r["mse"]) for _, r in ends]
    for i in range(1, len(label_y)):
        label_y[i] = max(label_y[i], label_y[i - 1] + 0.16)
    for (method, r), ly in zip(ends, label_y):
        ax_b.annotate(f"{STYLE[method]['label']}  {r['mse']:.2e}", (r["epoch"], r["mse"]),
                      xytext=(r["epoch"] * 1.08, 10 ** ly), textcoords="data",
                      va="center", fontsize=8, color=INK)
    ax_b.set_yscale("log")
    ax_b.set_ylabel("Training-set MSE (eval mode)")
    ax_b.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    ax_b.set_title("(b) MSE convergence", loc="left", fontsize=10, color=INK)

    # shared log-epoch axis
    e_max = int(max(tr["epoch"].max(), lg["epoch"].max()))
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if t <= e_max]
    ax_b.set_xscale("log")
    ax_b.set_xticks(ticks)
    ax_b.set_xticklabels([str(t) for t in ticks])
    ax_b.minorticks_off()
    ax_b.set_xlim(0.9, e_max * 3.2)   # room for (b)'s end labels
    ax_b.set_xlabel("Epoch (log scale)")

    t_desc = "sparse top-32 teacher" if teacher == "sparse" else "dense teacher"
    fig.suptitle(f"Training dynamics  (N={n:,}, k=32, {t_desc})", x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.01,
             "(a) Each method's own 32-neighbor graph S_t in its own run, audited on all N rows every epoch (not part of\n"
             "training cost); replacement = 1 - |S_t ∩ S_{t-1}| / 32, averaged over rows. Dense attends to all nodes; its "
             "line tracks\nthe top-32 of its learned scores. ARIAD's early peak is its first refreshes away from the random "
             "initial graph.\n"
             f"(b) MSE at epoch 1 and every {every} epochs. Both panels come from the same seed-0 training runs\n"
             "(full batch, 1 step = 1 epoch, Adam lr=1e-2, identical initialisation for all methods).",
             fontsize=7, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.11, 1, 0.96))
    save(fig, name)


def fig_budget_sweep(bs: pd.DataFrame, ref: pd.DataFrame, teacher: str, n: int, name: str) -> None:
    """Final MSE and Recall@32 of ARIAD vs candidate budget C_scored; Exact/Dense as reference lines."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    bs = bs.sort_values("c_scored")
    exact = ref[ref["method"] == "exact"].iloc[0]
    dense = ref[ref["method"] == "dense"].iloc[0]
    a = STYLE["ariad"]

    panels = [("final_mse", "Final MSE", "{:.3e}", False),
              ("final_teacher_match", "Final Recall@32", "{:.4f}", True)]
    for ax, (col, ylabel, fmt, show_dense) in zip(axes, panels):
        style_axes(ax)
        ax.plot(bs["c_scored"], bs[col], color=a["color"], linewidth=2, marker=a["marker"], markersize=7,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label="ARIAD", zorder=3)
        ax.axhline(exact[col], color=STYLE["exact"]["color"], linewidth=1.5, linestyle="--",
                   label=f"Exact top-k ({fmt.format(exact[col])})", zorder=2)
        if show_dense:
            ax.axhline(dense[col], color=STYLE["dense"]["color"], linewidth=1.5, linestyle=":",
                       label=f"Dense ({fmt.format(dense[col])})", zorder=2)
        ax.set_xscale("log")
        ax.set_xticks(bs["c_scored"])
        ax.set_xticklabels([f"{int(c)}\n×{s:g}" for c, s in zip(bs["c_scored"], bs["scale"])], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("ARIAD candidates scored per row, $C_{\\rm scored}$  (budget scale)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK,
                  loc="upper right" if col == "final_mse" else "lower right")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    t_desc = "sparse top-32 teacher" if teacher == "sparse" else "dense teacher"
    fig.suptitle(f"ARIAD candidate-budget sweep  (N={n:,}, k=32, {t_desc})", x=0.01, ha="left",
                 fontsize=11, color=INK)
    fig.text(0.01, 0.01,
             "Budget scale s multiplies every expansion source: c_o2 = 80s, c_i1 = k_rev = 32s; O(i) (k = 32) is always kept, "
             "so C_scored = 32 + 112s (measured).\n"
             f"Final epoch of 500, single seed. Dense final MSE = {dense['final_mse']:.2e} (off the left panel's scale).",
             fontsize=7, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    save(fig, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", choices=["1", "2", "3", "all"], default="all")
    ap.add_argument("--teacher", choices=["sparse", "dense"], default="sparse")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=10000, help="N for figs 2 and 3")
    ap.add_argument("--every", type=int, default=10, help="epoch spacing of the fig-2(b) MSE samples")
    args = ap.parse_args()
    suffix = f"{args.teacher}_k{args.k}_e{args.epochs}_seed{args.seed}"
    t_tag = "_sparseT32" if args.teacher == "sparse" else ""
    run_tag = f"N{args.n}{t_tag}_k{args.k}_e{args.epochs}_seed{args.seed}"

    if args.fig in ("1", "all"):
        df = load_summaries(args.teacher, args.k, args.epochs, args.seed)
        fig_flops_vs_n(df, args.teacher, f"fig1_flops_vs_n_{suffix}")
    if args.fig in ("2", "all"):
        tr = pd.read_csv(HERE / f"graph_tracking_{run_tag}.csv")
        lg = load_log(args.teacher, args.n, args.k, args.epochs, args.seed)
        fig_training_dynamics(tr, lg, args.teacher, args.n, args.every,
                              f"fig2_training_dynamics_N{args.n}_{suffix}")
    if args.fig in ("3", "all"):
        bs = pd.read_csv(HERE / f"budget_sweep_{run_tag}_summary.csv")
        ref = pd.read_csv(HERE / f"compare_dense_exact_ariad_{run_tag}_summary.csv")
        fig_budget_sweep(bs, ref, args.teacher, args.n, f"fig3_budget_sweep_N{args.n}_{suffix}")


if __name__ == "__main__":
    main()
