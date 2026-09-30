# Watching the neighbor graph learn

The supplied visualization follows a single ARIAD teacher–student training run.
The student receives output-value supervision, not teacher edges. Its maintained
graph moves from about 3% teacher match at epoch 0 to about 96% at epoch 160.
This is approximate recovery of the teacher's neighbor structure, not exact
graph equality or a general convergence guarantee.

![Neighbor graph evolution](kNN_convergence.png)

## Configuration

| Parameter | Value |
|---|---:|
| Nodes | 2,006 (118 clusters × 17 nodes) |
| Displayed nodes | 102 (first 6 clusters) |
| Teacher geometry / value dimensions | 10 / 8 |
| Student embedding dimension | 32 |
| Teacher / attention width | 16 / 16 |
| Candidate slots per row | 64 = 16 retained + 32 outgoing two-hop + 16 incoming |
| Jitter / teacher temperature | 0.10 / 0.3 |
| Adam learning rate / steps / seed | 0.02 / 160 / 0 |

Both projections are learned. The graph is refreshed through local search at
every training forward pass. It is never rebuilt through exhaustive all-pairs
search for training. Evaluation here compares with the fixed teacher graph.
Epoch 0 is the first scored graph after random initialization, before optimizer
updates. All annotated match/MSE values use the full dataset; only edges between
displayed nodes are drawn. Colors identify clusters; green edges match teacher
edges and red edges do not. Independent PCA projections make absolute panel
orientations arbitrary.

This illustration has different dimensions, graph width, temperature, cluster
size, and optimization settings from the README's main experimental tables.
It must not be presented as one of those paired three-seed runs.

## Cost ratios

Using the existing single-space arithmetic model with d_in=18, d_z=32, d_v=8:

```text
Exact search scores = N²     = 4,024,036 per step
ARIAD search scores = N × 64 =   128,384 per step
ARIAD / Exact search = 64 / 2006 = 3.1904% (96.8096% fewer)

F = 6 N d_in (d_z + d_v) + 6 P_attn (d_z + d_v) + 2 P_search d_z
P_attn = N × 16 for both methods
Exact F = 273,907,264 FLOPs per step
ARIAD F =  24,585,536 FLOPs per step
ARIAD / Exact F = 8.97586% (91.02414% less)
```

The executable cost check is the source of truth for rounded ratios. The model
includes projections, attention, and search, but excludes softmax, top-k,
deduplication, reverse-buffer construction, gather/scatter, memory traffic, and
evaluation. It does not measure hardware speedup. The Exact comparison here is
a cost calculation, not a paired Exact training trajectory in the figure.

## Reproduce

From the repository root, with the README dependencies installed:

```bash
# Recalculate costs without training.
python ariad_single_space/highlight/cost_check.py

# Train the illustrative run on CPU and regenerate the PNG/PDF.
python ariad_single_space/highlight/highlight_kNN_convergence.py
```

The original supplied PNG/PDF and plotting script are retained. Figure regeneration
runs 160 steps and overwrites those images; numeric trajectories may vary with
library versions. The release check recalculates costs without rerunning training.
