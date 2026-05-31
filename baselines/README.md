# Baseline Evaluation

This directory contains the baseline evaluation wrapper and the included
per-method runners. `run_paper_baselines.py` calls scripts in
`paper_algorithms/` and uses the same input/output contract as the benchmark
evaluation scripts.

The recommended TabCausal path is still:

```bash
python -m tabcausal.cli predict-dir ...
```

List all registered baselines:

```bash
python -m baselines.run_paper_baselines --list
```

Baseline run:

```bash
python -m baselines.run_paper_baselines \
  --method pc \
  --input-dir examples/generated_synthetic/[gp_hard]_obs \
  --output-root results/baselines \
  --exp-name pc_gp_hard_obs \
  --regime obs \
  --max-per-f 10 \
  --save-preds
```

For a mixed/interventional method:

```bash
python -m baselines.run_paper_baselines \
  --method cdis \
  --input-dir examples/generated_synthetic/[gp_hard]_int \
  --output-root results/baselines \
  --exp-name cdis_gp_hard_int \
  --regime mixed \
  --max-per-f 10 \
  --save-preds
```

Run every included baseline runner as a smoke test:

```bash
python scripts/smoke_paper_baselines.py \
  --data-root examples/generated_smoke_7families \
  --output-root results/local_release_smoke/baselines \
  --datasets gp_hard \
  --max-per-f 1 \
  --timeout-seconds 300

cat results/local_release_smoke/baselines/baseline_smoke_manifest.csv
```

The smoke script runs observation-only methods on `[gp_hard]_obs` and
mixed/interventional methods on `[gp_hard]_int`. It records failures instead of
stopping at the first failed method, so the manifest is the best place to check
whether the local environment is ready. Every method subprocess uses the same
300-second wall-clock timeout by default; override it with
`--timeout-seconds <seconds>` or disable it with `--timeout-seconds 0`. The
manifest records the timeout applied to each method.

For a local machine to pass every method, install the top-level Python
requirements, make `Rscript` available with the R package `pcalg`, and provide
an AVICI pretrained-weight cache when running AVICI. SEA checkpoints can be
passed with `--sea-obs-checkpoint` and `--sea-int-checkpoint`. The smoke script
passes `--min_samples 1` to NoDAGS so the tiny one-graph smoke data is
sufficient to exercise the runner; formal benchmark runs keep NoDAGS' own
defaults.

DCDI uses a per-graph runtime budget internally: each graph has a 300-second
limit and, on timeout, the current learned graph is exported as that graph's
prediction. In the smoke script this internal DCDI limit follows the shared
`--timeout-seconds` value; for one-off direct runs, override it with
`--dcdi-time-limit-seconds` or `DCDI_TIME_LIMIT_SECONDS`.

## Included Baselines

The baseline wrapper includes runners for:

`avici`, `sea`, `dagma`, `sdcd`, `dcdi`, `gies`, `cdis`, `igsp`, `notears`,
`notears_mlp`, `lingam`, `pc`, `das`, `randomregress`, and `nodags`.

All methods above are launched through the same command-line wrapper and write
the same result files: `raw_metrics.csv`, `summary.csv`, and, when requested,
`predictions.npz`.

Install Python dependencies with the single top-level requirements file:

```bash
python -m pip install -r requirements.txt
```

Check the resolved baseline runtime:

```bash
python scripts/check_baseline_environment.py
```

The baseline wrapper follows the direct evaluation convention used for the
benchmark tables and applies threshold `0.5` for probability/score outputs.
Additional method-specific runner arguments can be appended after `--` when
needed for local debugging.

## Evaluation Conventions

The baseline evaluation uses the following conventions:

- Data values are read from channel `x[..., 0]`; intervention indicators are
  read from channel `x[..., 1]`.
- Value columns are z-scored per graph over all rows before classical baselines.
  Intervention flags are never standardized.
- PC, GIES/GES, and CDIS can output partially directed structures. In direct
  fixed-threshold evaluation, uncertain edges were not randomized: the default
  `liberal` strategy orients them from the smaller node index to the larger node
  index.
- PC, LiNGAM, NOTEARS, NOTEARS-MLP, DAGMA, DAS, and RandomRegress are
  observation-only baselines in the benchmark evaluation.
- GIES/GES, IGSP, CDIS, AVICI, SEA, SDCD, and DCDI can consume mixed
  observational/interventional files. When GIES receives no intervention
  targets, it is the observational GES setting.
- Fixed `0.5` graph-threshold evaluation was used for probabilistic methods
  unless a method naturally returned a discrete graph.
- The runners also accept release-style example files without `mask` and
  with two-dimensional `x`; for benchmark files that already store
  `mask` and `[value, intervention]` channels, this compatibility branch is not
  used and should not change benchmark behavior.

These notes are intended to make input formatting and output decoding visible to
readers.
