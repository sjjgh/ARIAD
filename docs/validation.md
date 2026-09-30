# Release validation

The release was checked on Windows 11 with Python 3.13.13, PyTorch 2.11.0+cu128,
NumPy 2.5.2, pandas 3.0.6, and Matplotlib 3.11.2. The available CUDA device was
an NVIDIA GeForce RTX 5070 Ti. These describe the release checks, not the unknown
complete environment of the original 500-step experiments. Dependency lower
bounds in `requirements.txt` are not a tested compatibility matrix.

| Check | Result |
|---|---|
| `python -m pytest -q` | 8 tests passed: 6 original graph tests and 2 isolated model integration tests |
| Single-space CPU demo | 3 optimizer steps, finite gradients, valid nonself unique neighbors, eval graph reuse |
| QK CPU demo | Same checks, with independent Q/K and frozen value projection |
| N=128 CUDA teacher–student run | Data generation and 3 steps each of Dense, Exact, and ARIAD completed |
| `python scripts/reproduce.py --suite figures` | Both quality tables and all 5 figures regenerated from committed records |
| Numerical core provenance | 9 search/model/generator/comparison files matched the supplied research source byte for byte before Git newline normalization and removal of one trailing blank line |

The single-space plotting script now uses Matplotlib's noninteractive Agg
backend. The root reproduction runner gives Matplotlib a writable local cache.
Neither changes the numerical training algorithm. Full 500-step sweeps were not
rerun for the release; the README reports the author's retained experiment records.
