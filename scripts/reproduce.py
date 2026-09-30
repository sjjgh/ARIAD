"""Portable, sequential entry point for the recorded experiment settings."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(folder: str, script: str, *args: object) -> None:
    env = os.environ.copy()
    env.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs' / '.matplotlib'))
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    subprocess.run([sys.executable, script, *map(str, args)], cwd=ROOT / folder,
                   env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['figures', 'single', 'qk', 'budget', 'dynamics'], required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1008, 2000, 5008, 10000, 20000])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    if any(n < 80 or n % 16 for n in args.sizes):
        parser.error('sizes must be multiples of 16 and at least 80')
    single, qk = 'ariad_single_space', 'ariad_qk_space'
    if args.suite == 'figures':
        run(single, 'make_tables.py', '--teacher', 'sparse', '--k', 32, '--epochs', 500, '--seeds', 0, 1, 2)
        run(single, 'plot_results.py', '--fig', 'all', '--teacher', 'sparse', '--k', 32, '--epochs', 500, '--seed', 0)
        run(qk, 'make_results.py')
    elif args.suite == 'single':
        for n in args.sizes:
            for seed in args.seeds:
                run(single, 'compare_dense_exact_ariad.py', '--n-clusters', n // 16,
                    '--seed', seed, '--epochs', 500, '--teacher-dense', 0)
    elif args.suite == 'qk':
        for n in args.sizes:
            run(qk, 'scaling_sweep.py', '--gen', n)
            for method in ['dense', 'exact_topk', 'ariad']:
                for seed in args.seeds:
                    run(qk, 'scaling_sweep.py', '--run', n, method, seed)
    else:
        script = 'budget_sweep.py' if args.suite == 'budget' else 'graph_tracking.py'
        run(single, script, '--n-clusters', 625, '--epochs', 500, '--seed', 0, '--teacher-dense', 0)


if __name__ == '__main__':
    main()
