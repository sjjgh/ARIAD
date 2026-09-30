"""Aggregate result/raw/*.json (written by scaling_sweep.py) into the Table 1 / Fig. 1 / Fig. 2
analogues for the dual-space (QK) experiment. Everything lands in result/.

    python make_results.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULT_DIR = HERE / "result"
RAW_DIR = RESULT_DIR / "raw"

NS = [1008, 2000, 5008, 10000, 20000]
METHODS = ["dense", "exact_topk", "ariad"]
LABEL = {"dense": "Dense", "exact_topk": "Exact top-k", "ariad": "ARIAD"}
SEEDS = [0, 1, 2]

# --- palette: first three categorical slots of the reference palette (validated all-pairs) ---
COLOR = {"dense": "#2a78d6", "exact_topk": "#eb6834", "ariad": "#1baf7a"}
MARKER = {"dense": "o", "exact_topk": "s", "ariad": "^"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

# --- cost model (see README): dual-space projections + attention + search ---
D_IN, D_Z, D_V, K = 128, 64, 64, 32
# candidate slots scored per row per training step (measured from the configs, before dedup/self masking):
#   QQ graph: k + c_o2 + c_i1 = 64 + 80 + 32 = 176      (ariad_train_qk.QQ_CONFIG)
#   KK graph: k + c_o2 + c_i1 = 64 + 80 + 32 = 176      (ariad_train_qk.KK_CONFIG)
#   QK graph: k_final + b_kF + b_qF + b_kR = 32 + 32 + 32 + 32 = 128 (ariad_train_qk.QK_CONFIG, b_qR=0)
C_QQ, C_KK, C_QK = 176, 176, 128
C_SCORED = C_QQ + C_KK + C_QK


def flops(n: int, method: str) -> float:
    proj = 6 * n * D_IN * (2 * D_Z + D_V)                 # Wq, Wk, Wv (fwd+bwd ~ 3x fwd, 2 flops/MAC)
    p_attn = n * n if method == "dense" else n * K
    p_search = {"dense": 0, "exact_topk": n * n, "ariad": n * C_SCORED}[method]
    return proj + 6 * p_attn * (D_Z + D_V) + 2 * p_search * D_Z


def load_runs():
    runs = {}
    for f in RAW_DIR.glob("N*_*_s*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        runs[(r["n"], r["method"], r["seed"])] = r
    return runs


def mean_std(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else float("nan"))


def fmt(mean, std, prec, floor):
    """paper style: mean±std, with '<floor' when the std is below display precision."""
    s = f"<{floor}" if std < 10 ** (-prec) else f"{std:.{prec}f}"
    return f"{mean:.{prec}f}±{s}"


def build_table(runs):
    rows = []
    for n in NS:
        row = {"N": n}
        for m in METHODS:
            rs = [runs.get((n, m, s)) for s in SEEDS]
            oks = [r for r in rs if r and r["status"] == "ok"]
            if not oks:
                status = "OOM" if any(r and r["status"] == "OOM" for r in rs) else "missing"
                row[m] = {"status": status}
                continue
            mse = [r["final"]["mse"] * 1e4 for r in oks]
            rec = [r["final"]["recall"] for r in oks]
            exh = [r["final"]["match_exhaustive"] for r in oks]
            row[m] = {"status": "ok", "n_seeds": len(oks), "mse": mean_std(mse), "match": mean_std(rec),
                      "cur_recall": mean_std(exh), "mse_by_seed": mse, "match_by_seed": rec}
        rows.append(row)
    return rows


def write_table(rows):
    # ---- csv (full precision) ----
    with (RESULT_DIR / "table1.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N", "method", "n_seeds", "mse_x1e-4_mean", "mse_x1e-4_std", "teacher_match@32_mean",
                    "teacher_match@32_std", "current_graph_recall_mean", "status"])
        for row in rows:
            for m in METHODS:
                c = row[m]
                if c["status"] != "ok":
                    w.writerow([row["N"], m, 0, "", "", "", "", "", c["status"]])
                else:
                    w.writerow([row["N"], m, c["n_seeds"], f"{c['mse'][0]:.6f}", f"{c['mse'][1]:.6f}",
                                f"{c['match'][0]:.6f}", f"{c['match'][1]:.6f}", f"{c['cur_recall'][0]:.6f}", "ok"])

    def cell(c, key, prec, floor):
        return c["status"] if c["status"] != "ok" else fmt(*c[key], prec, floor)

    # ---- markdown ----
    md = ["| N | MSE ×10⁻⁴ Dense | MSE Exact top-k | MSE ARIAD | Match@32 Dense | Match@32 Exact top-k | Match@32 ARIAD |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        md.append(f"| {row['N']:,} | " + " | ".join(cell(row[m], "mse", 2, "0.01") for m in METHODS) + " | "
                  + " | ".join(cell(row[m], "match", 4, "0.0001") for m in METHODS) + " |")
    (RESULT_DIR / "table1.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- LaTeX (same layout as the paper's tab:single-quality) ----
    tex = [r"\begin{tabular}{r ccc ccc}", r"\toprule",
           r" & \multicolumn{3}{c}{MSE ($\times10^{-4}$)} & \multicolumn{3}{c}{Teacher match@32} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r"$N$ & Dense & Exact top-$k$ & ARIAD & Dense & Exact top-$k$ & ARIAD \\", r"\midrule"]
    for row in rows:
        cells = [cell(row[m], "mse", 2, "0.01") for m in METHODS] + [cell(row[m], "match", 4, "0.0001") for m in METHODS]
        cells = [c.replace("±", r"$\pm$").replace("<", r"$<$") for c in cells]
        tex.append(f"{row['N']:,} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (RESULT_DIR / "table1.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # ---- extra: ARIAD vs Exact paired comparison + current-graph recall ----
    ex = ["| N | ARIAD current-graph recall@32 (sampled rows) | ARIAD − Exact: paired MSE change | ARIAD − Exact: match@32 |",
          "|---:|---:|---:|---:|"]
    for row in rows:
        a, e = row["ariad"], row["exact_topk"]
        if a["status"] != "ok" or e["status"] != "ok":
            ex.append(f"| {row['N']:,} | n/a | n/a | n/a |")
            continue
        pct = [100.0 * (x - y) / y for x, y in zip(a["mse_by_seed"], e["mse_by_seed"])]
        dm = [x - y for x, y in zip(a["match_by_seed"], e["match_by_seed"])]
        ex.append(f"| {row['N']:,} | {a['cur_recall'][0]:.4f} | {np.mean(pct):+.2f}% (seeds: "
                  + ", ".join(f"{p:+.2f}%" for p in pct) + f") | {np.mean(dm):+.4f} |")
    (RESULT_DIR / "table1_extra.md").write_text("\n".join(ex) + "\n", encoding="utf-8")
    return md, ex


# ------------------------------------------------------------------ figures
def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig1_cost():
    fig, ax = plt.subplots(figsize=(6.2, 4.0), facecolor=SURFACE)
    style(ax)
    xs = np.array(NS, dtype=float)
    slopes, rows = {}, []
    for m in METHODS:
        ys = np.array([flops(n, m) / 1e9 for n in NS])
        slopes[m] = float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
        ax.plot(xs, ys, color=COLOR[m], marker=MARKER[m], markersize=6, linewidth=2, label=LABEL[m],
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(f"{LABEL[m]} {ys[-1]:.3g}", (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
        for n, y in zip(NS, ys):
            rows.append({"N": n, "method": m, "gflops_per_step": y})
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(NS); ax.set_xticklabels([f"{n:,}" for n in NS])
    ax.minorticks_off()
    ax.set_xlim(NS[0] * 0.85, NS[-1] * 1.9)
    ax.set_xlabel("N (number of tokens)", color=INK2, fontsize=9)
    ax.set_ylabel("Estimated arithmetic GFLOPs per training step", color=INK2, fontsize=9)
    ax.set_title("Estimated training FLOPs vs N  (dual-space QK, k=32, sparse top-32 teacher)", color=INK,
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.text(0.01, 0.005,
             f"Log-log slope: Dense {slopes['dense']:.2f}, Exact top-k {slopes['exact_topk']:.2f}, ARIAD {slopes['ariad']:.2f}. "
             f"ARIAD scores C={C_SCORED} slots/row/step (QQ {C_QQ} + KK {C_KK} + QK {C_QK}).\n"
             "Cost-model estimate, not measured; excludes softmax, top-k, dedup and memory traffic.",
             color=MUTED, fontsize=6.5, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"fig1_cost.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    with (RESULT_DIR / "fig1_cost.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["N", "method", "gflops_per_step"])
        w.writeheader(); w.writerows(rows)
    return slopes, {(r["N"], r["method"]): r["gflops_per_step"] for r in rows}


def fig2_dynamics(runs):
    hist = {}
    for m in METHODS:
        r = runs.get((10000, m, 0))
        if r and r["status"] == "ok" and "history" in r:
            hist[m] = r["history"]
    if not hist:
        return False
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.4, 6.4), facecolor=SURFACE, sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.25]})
    for ax in (a, b):
        style(ax)
    ends = []
    for m, h in hist.items():
        ep = [x["epoch"] for x in h if x["replace_pct"] is not None]
        rp = [x["replace_pct"] for x in h if x["replace_pct"] is not None]
        a.plot(ep, rp, color=COLOR[m], linewidth=1.6, label={"dense": "Dense (top-32 of its learned scores)",
                                                              "exact_topk": "Exact top-k (its exact top-32 graph)",
                                                              "ariad": f"ARIAD (maintained graph, $C_{{\\rm scored}}$={C_SCORED})"}[m])
        pts = [x for x in h if x["epoch"] == 1 or x["epoch"] % 10 == 0]
        b.plot([x["epoch"] for x in pts], [x["mse"] for x in pts], color=COLOR[m], linewidth=2, marker=MARKER[m],
               markersize=4, markevery=[0, 1, 2, 5, 10, 20, 50], markeredgecolor=SURFACE, markeredgewidth=1,
               label=LABEL[m], zorder=3)
        ends.append((m, pts[-1]))
    a.set_ylim(bottom=0)
    # end labels spread apart vertically (log10 units) so near-equal finals don't collide
    ends.sort(key=lambda t: t[1]["mse"])
    label_y = [math.log10(r["mse"]) for _, r in ends]
    for i in range(1, len(label_y)):
        label_y[i] = max(label_y[i], label_y[i - 1] + 0.16)
    for (m, r), ly in zip(ends, label_y):
        b.annotate(f"{LABEL[m]}  {r['mse']:.2e}", (r["epoch"], r["mse"]), xytext=(r["epoch"] * 1.08, 10 ** ly),
                   textcoords="data", color=INK, fontsize=8, va="center")
    a.set_ylabel("Neighbors replaced\nper epoch (%)", color=INK2, fontsize=9)
    a.set_title("(a) The neighbor graph changes during training", color=INK, fontsize=9.5, loc="left")
    a.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper right")
    b.set_yscale("log"); b.set_xscale("log")
    b.set_xlim(0.9, 500 * 3.2)
    b.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500]); b.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200", "500"])
    b.set_xlim(0.9, 500 * 3.2)   # room for the end labels
    b.minorticks_off()
    b.set_xlabel("Epoch (log scale)", color=INK2, fontsize=9)
    b.set_ylabel("Training-set MSE (eval mode)", color=INK2, fontsize=9)
    b.set_title("(b) MSE convergence", color=INK, fontsize=9.5, loc="left")
    b.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="lower left")
    fig.suptitle("Training dynamics  (N=10,000, k=32, sparse top-32 teacher)", color=INK, fontsize=11,
                 x=0.01, ha="left")
    fig.text(0.01, 0.01,
             "(a) Each method's own 32-neighbor graph H_t in its own run, audited on all N rows every epoch (not part of\n"
             "training cost); replacement = 1 - |H_t ∩ H_{t-1}| / 32, averaged over rows. Dense attends to all nodes; its line\n"
             "tracks the top-32 of its learned scores. ARIAD's early peak is its first refreshes away from the random initial graph.\n"
             "(b) MSE at epoch 1 and every 10 epochs. Both panels come from the same seed-0 training runs (full batch,\n"
             "1 step = 1 epoch, Adam lr=1e-2, identical initialisation for all methods; untied Wq/Wk, fixed values V).",
             color=MUTED, fontsize=7, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.11, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(RESULT_DIR / f"fig2_dynamics.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    with (RESULT_DIR / "fig2_dynamics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "epoch", "mse", "teacher_match@32", "current_graph_recall_sampled", "replace_pct"])
        for m, h in hist.items():
            for x in h:
                w.writerow([m, x["epoch"], x["mse"], x["recall"], x["match_exhaustive"],
                            "" if x["replace_pct"] is None else x["replace_pct"]])
    return True


def main():
    RESULT_DIR.mkdir(exist_ok=True)
    runs = load_runs()
    rows = build_table(runs)
    md, ex = write_table(rows)
    print("\n".join(md)); print(); print("\n".join(ex)); print()
    slopes, cost = fig1_cost()
    print("log-log slopes:", {k: round(v, 3) for k, v in slopes.items()})
    print("GFLOPs @N=20000:", {m: round(cost[(20000, m)], 2) for m in METHODS})
    print("fig2 written:", fig2_dynamics(runs))


if __name__ == "__main__":
    main()
