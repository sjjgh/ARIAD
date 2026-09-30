# Contributing

Bug reports, reproducibility checks, and focused improvements are welcome.
For a bug report, include the command, Python/PyTorch versions, device, seed,
and a small example that reproduces the issue.

Install `requirements-dev.txt`, then run `python -m pytest -q` from the root.
The two experiment directories intentionally contain different historical
versions of the single-space seeker. Run them in separate processes; do not
merge their imports or synchronize their implementations without rerunning
the affected experiments.

For algorithm changes, report quality and search cost against a paired Exact
top-k baseline. Separate teacher match from current-graph recall, and measured
runtime from modeled arithmetic. Preserve the provenance of committed results.
Do not commit generated datasets, model checkpoints, credentials, or local
environments. Use a new output directory for exploratory runs.
