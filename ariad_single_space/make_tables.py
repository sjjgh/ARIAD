"""make_tables.py

Table 1: final-epoch MSE and recall@k (= teacher match@k) per N for
Dense / Exact top-k / ARIAD, read from the per-N summary CSVs written by
compare_dense_exact_ariad.py. With several --seeds, cells are
mean ± sample std (ddof=1) over training seeds (the dataset is the same for
every seed; the seed controls weight init, ARIAD's random initial graph and
its candidate sampling).

recall@k = mean_i |T_k(i) ∩ N_k(i)| / k, with T_k the teacher top-k (from S)
and N_k the student's neighbor set (see compare_dense_exact_ariad.py).

Usage: python make_tables.py [--teacher sparse|dense] [--k 32] [--epochs 500] [--seeds 0 1 2]
Outputs: result/<name>.csv (mean/std/n per cell), _per_seed.csv, .md, .tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from plot_results import STYLE, load_summaries

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "result"
METHODS = ["dense", "exact", "ariad"]
MSE_SCALE = 1e-4   # MSE cells are reported in units of 1e-4


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["n", "method"])
    out = g.agg(mse_mean=("final_mse", "mean"), mse_std=("final_mse", "std"),
                recall_mean=("final_teacher_match", "mean"), recall_std=("final_teacher_match", "std"),
                n_seeds=("seed", "nunique")).reset_index()
    out["o"] = out["method"].map({m: i for i, m in enumerate(METHODS)})
    return out.sort_values(["n", "o"]).drop(columns="o")


def cell(mean: float, std: float, fmt: str, pm: str) -> str:
    if pd.isna(std):
        return fmt.format(mean)
    s = fmt.format(std)
    if float(s) == 0.0:   # std below display precision: show an upper bound, not a misleading 0
        s = "<" + fmt.format(10 ** -int(fmt.split(".")[1][0]))
    return f"{fmt.format(mean)}{pm}{s}"


def to_markdown(agg: pd.DataFrame, k: int) -> str:
    head = ["N"] + [f"MSE (×1e-4) {STYLE[m]['label']}" for m in METHODS] + [f"Recall@{k} {STYLE[m]['label']}" for m in METHODS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---:|" * len(head)]
    for n, rows in agg.groupby("n", sort=True):
        r = rows.set_index("method")
        cells = [f"{int(n):,}"]
        cells += [cell(r.loc[m, "mse_mean"] / MSE_SCALE, r.loc[m, "mse_std"] / MSE_SCALE, "{:.2f}", " ± ") for m in METHODS]
        cells += [cell(r.loc[m, "recall_mean"], r.loc[m, "recall_std"], "{:.4f}", " ± ") for m in METHODS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def to_latex(agg: pd.DataFrame, k: int, caption: str) -> str:
    lab = [STYLE[m]["label"] for m in METHODS]
    out = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\small",
        r"\begin{tabular}{r ccc ccc}",
        r"\toprule",
        rf" & \multicolumn{{3}}{{c}}{{MSE ($\times 10^{{-4}}$)}} & \multicolumn{{3}}{{c}}{{Recall@{k}}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        "$N$ & " + " & ".join(lab + lab) + r" \\",
        r"\midrule",
    ]
    for n, rows in agg.groupby("n", sort=True):
        r = rows.set_index("method")
        mse = [cell(r.loc[m, "mse_mean"] / MSE_SCALE, r.loc[m, "mse_std"] / MSE_SCALE, "{:.2f}", r"$\pm$") for m in METHODS]
        rec = [cell(r.loc[m, "recall_mean"], r.loc[m, "recall_std"], "{:.4f}", r"$\pm$") for m in METHODS]
        # bare '<' prints as '¡' in LaTeX text mode; keep it inside the same math group as \pm (no '$$')
        cells = [c.replace(r"$\pm$<", r"$\pm{<}$") for c in mse + rec]
        out.append(f"{int(n):,} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=["sparse", "dense"], default="sparse")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    per_seed = pd.concat([load_summaries(args.teacher, args.k, args.epochs, s) for s in args.seeds],
                         ignore_index=True)
    counts = per_seed.groupby(["n", "method"])["seed"].nunique()
    if (counts != len(args.seeds)).any():
        print("warning: some (N, method) cells are missing seeds:\n", counts[counts != len(args.seeds)])
    agg = aggregate(per_seed)

    seed_tag = "seed" + "-".join(map(str, args.seeds)) if len(args.seeds) > 1 else f"seed{args.seeds[0]}"
    name = f"table1_mse_recall_{args.teacher}_k{args.k}_e{args.epochs}_{seed_tag}"
    t_desc = f"sparse top-{args.k} teacher" if args.teacher == "sparse" else "dense teacher"
    stat = (f"mean $\\pm$ std over {len(args.seeds)} training seeds" if len(args.seeds) > 1
            else f"seed {args.seeds[0]}")
    caption = f"Final-epoch MSE and Recall@{args.k} ({t_desc}, $k={args.k}$, {args.epochs} epochs; {stat})."

    OUT_DIR.mkdir(exist_ok=True)
    agg.to_csv(OUT_DIR / f"{name}.csv", index=False)
    per_seed[["n", "method", "seed", "final_mse", "final_teacher_match", "final_match_exhaustive"]].sort_values(
        ["n", "method", "seed"]).to_csv(OUT_DIR / f"{name}_per_seed.csv", index=False)
    (OUT_DIR / f"{name}.md").write_text(to_markdown(agg, args.k), encoding="utf-8")
    (OUT_DIR / f"{name}.tex").write_text(to_latex(agg, args.k, caption), encoding="utf-8")
    print(to_markdown(agg, args.k))
    print(f"saved {OUT_DIR / name}.csv/_per_seed.csv/.md/.tex")


if __name__ == "__main__":
    main()
