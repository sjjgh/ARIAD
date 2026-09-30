# Reproducing the experiments

Run commands from the repository root, using the environment installed in the
[quick start](../README.md#quick-start). Each experiment folder is self-contained;
the two `ariad_single.py` implementations are intentionally different. The
root runner launches separate Python processes to avoid import collisions.

## Regenerate published artifacts without training

```bash
python scripts/reproduce.py --suite figures
```

This reads the committed single-space CSVs and 45 QK JSON records and writes
tables, PNGs, and PDFs in each experiment's `result/` directory. The original
figure labels “Recall@32” refer to teacher match; current-graph recall is separate.

## Rerun training

```bash
# Five sizes, three training seeds, three methods; sequential execution.
python scripts/reproduce.py --suite single
python scripts/reproduce.py --suite qk

# Smaller selection, with the same 500-step configuration.
python scripts/reproduce.py --suite single --sizes 2000 --seeds 0
python scripts/reproduce.py --suite qk --sizes 2000 --seeds 0

# N=10000, seed=0, 500 steps.
python scripts/reproduce.py --suite budget
python scripts/reproduce.py --suite dynamics
python scripts/reproduce.py --suite figures
```

Training commands **overwrite matching result records**. Use a separate clone
or branch for new runs if you want to preserve the reference artifacts. The root
QK runner explicitly executes each requested run; unlike the legacy `--all`
command, it does not skip the JSON files already committed to the repository.
The figures suite requires the complete reference grid; a fresh partial sweep
alone cannot regenerate every figure.

Datasets are generated locally on demand and ignored by Git. The released code
contains no dataset download requirement. Existing generated datasets are reused;
if you change generation parameters, use a fresh dataset directory. The QK dataset
name encodes its generation settings, while the historical single-space name
does not encode every knob.

## Configuration

| Setting | Single space | QK |
|---|---|---|
| N | 1008, 2000, 5008, 10000, 20000 | Same |
| Cluster size | 16 | 16 |
| Input / projection dimensions | 128 / 64 | 128 / 64 |
| Jitter | 0.10 | 0.05 |
| Teacher scores | Globally standardized off-diagonal dot products | Dot products / √64 |
| Teacher temperature | 0.5 | 0.01 |
| Teacher / attention width | 32 / 32 | 32 / 32 |
| Search slots per row | 144, runtime counted | 480, configuration derived |
| Values | Learned projection | Frozen standardized-value selector |
| Optimization | Adam, lr=0.01, no weight decay, 500 steps | Same, constant LR |
| Data / training seeds | Data=0, training=0/1/2 | Same |
| Current-graph audit | All query rows | 1000 sampled rows |

The root single-space runner explicitly selects `--teacher-dense 0`, since
the raw generator's default is a dense teacher. The QK runner uses
`scaling_sweep.configure()` to select the fixed-value, constant-learning-rate
setup; launching `ariad_train_qk.py` directly uses different exploratory defaults.

## Resources and determinism

The 256-node CPU demos require no generated dataset. Full sweeps are substantially
heavier: data generation stores dense N×N arrays and Dense training materializes
quadratic attention tensors. A single float32 N×N matrix at N=20000 is about
1.6 GB, and several arrays/tensors can coexist. Start with a small run before
attempting the full sweep; exact hardware requirements depend on implementation
and device. The root runner executes sequentially to limit concurrent memory use.

The original run records do not provide a complete pinned software/hardware
environment. They are retained as reported evidence, not claimed as newly rerun
results. Seeds pair initial weights and data across methods, but floating-point
reductions, ties, device, and library versions can change trajectories. No
bitwise reproduction claim is made.

See [validation](validation.md) for checks performed for this release. CI runs
CPU tests and regenerates figures from the records, not the full training sweep.

## Result provenance

| Artifact | Source / regeneration |
|---|---|
| Single-space quality table | 15 summary CSVs × three methods; `make_tables.py` |
| Single-space cost / MSE curves | Seed-0 summaries and per-epoch CSVs; `plot_results.py` |
| Single-space turnover | `graph_tracking_N10000_sparseT32_k32_e500_seed0.csv` |
| Budget tradeoff | `budget_sweep_*_summary.csv` and `budget_sweep.py` |
| QK quality, cost, dynamics | 45 `result/raw/*.json`; `make_results.py` |
| Research narrative | Extracted experimental TeX in [research/](research/README.md) |

Original Chinese design notes mention exploratory ablations and predecessor
directories outside this curated release. Those historical references are not
required by the reproduction entry points documented here.
