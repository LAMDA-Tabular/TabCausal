# TabCausal

This repository contains the public TabCausal package: model definitions, a
Python API, command-line prediction tools, benchmark runners, and data
generators for reproducing the synthetic benchmark inputs.

## Repository Layout

```text
tabcausal/                  Python package
  model/                    Model architecture
  inference.py              Predictor API
  preprocessing.py          Data preprocessing
  evaluate.py               Evaluation utilities
  metrics.py                Metric utilities
  cli.py                    Command-line interface
baselines/                  Baseline runners and included dependencies
  paper_algorithms/         Included baseline scripts/dependencies
scripts/
  generate_benchmark_suite.py
                             Generate all seven synthetic benchmark families
  evaluate_directory.py     Evaluate one benchmark directory
  evaluate_benchmark_suite.py
                             Evaluate the seven-family benchmark suite
  visualize_prediction.py   Plot probability/adjacency heatmaps and embedding PCA
  visualize_evaluation.py   Plot summary metrics and prediction heatmaps
examples/
  make_example_data.py      Create a small gp_hard smoke-test dataset
  minimal_predict.py        Minimal Python API example
checkpoints/
  tabcausal-base.pt         Public checkpoint, if included separately
data_engine/                Benchmark data-generation components
```

## Requirements

- Python 3.10 or newer
- PyTorch 2.1 or newer
- NumPy 1.x. Some PyTorch wheels are not compatible with NumPy 2.x.
- scikit-learn, for AUROC/AP metrics and PCA visualization
- pandas, for CSV/TSV/Parquet/Pickle table inputs
- matplotlib, for optional visualization scripts
- Optional for official SID: R with the CRAN `SID` package
- Paper baseline Python dependencies are included in `requirements.txt`.
- Some optional graph-learning dependencies may require an OpenMP runtime in
  the active environment.

Install dependencies with either:

```bash
pip install -r requirements.txt
```

If your environment already has NumPy 2.x and PyTorch reports
`RuntimeError: Numpy is not available`, reinstall NumPy 1.x inside the same
environment:

```bash
python -m pip install "numpy>=1.23,<2" --force-reinstall
```

or, if you want the `tabcausal` command installed:

```bash
pip install -e .
```

If you do not want to install the package, use `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
python -m tabcausal.cli --help
```

## Checkpoint

The examples below use the checkpoint path:

```text
checkpoints/tabcausal-base.pt
```

## Quick Smoke Test

This checks that imports, checkpoint loading, prediction, and metric writing
work end to end.

```bash
cd /path/to/TabCausal-release
export PYTHONPATH="$PWD:$PYTHONPATH"

python examples/make_example_data.py \
  --output-root examples/example_data \
  --num-graphs 10 \
  --variables 5 \
  --observations 1000

python -m tabcausal.cli predict-dir \
  --checkpoint checkpoints/tabcausal-base.pt \
  --input-dir 'examples/example_data/[gp_hard]_obs' \
  --output-dir examples/example_results \
  --mode auto \
  --threshold 0.5 \
  --batch-size 16 \
  --device cpu \
  --save-embeddings

cat examples/example_results/summary.csv
ls -lh examples/example_results

python scripts/visualize_evaluation.py \
  --prediction-npz examples/example_results/predictions.npz \
  --index 0 \
  --output-dir examples/example_results/figures

find examples/example_results/figures -type f | sort
```

The example data uses the `gp_hard` observational setting with `f=5`. The
command above creates 10 graphs and 1000 observational samples per graph.

Use `--device cuda:0` when a CUDA GPU is available.

## Predict One Graph

```bash
python -m tabcausal.cli predict \
  --checkpoint checkpoints/tabcausal-base.pt \
  --input /path/to/data_f20_000.npz \
  --output outputs/data_f20_000_predictions.npz \
  --device cuda:0 \
  --threshold 0.5
```

The output file contains:

- `logits`: raw directed-edge logits
- `probabilities`: sigmoid probabilities
- `adjacency`: binary directed graph after thresholding
- `embeddings`: final-layer node embeddings, when `--include-embeddings` or
  `--embedding-output` is used

You can also read NumPy/Pandas-style tables and write separate files for each
artifact:

```bash
python -m tabcausal.cli predict \
  --checkpoint checkpoints/tabcausal-base.pt \
  --input /path/to/table.csv \
  --output outputs/example_prediction.npz \
  --adjacency-output outputs/example_adjacency.csv \
  --probability-output outputs/example_probabilities.csv \
  --embedding-output outputs/example_embeddings.npy \
  --device cuda:0
```

For mixed observational/interventional inputs stored as plain tables, pass a
same-shaped binary intervention indicator table:

```bash
python -m tabcausal.cli predict \
  --checkpoint checkpoints/tabcausal-base.pt \
  --input /path/to/values.csv \
  --intervention-input /path/to/intervention_mask.csv \
  --output outputs/mixed_prediction.npz \
  --device cuda:0
```

Supported single-file inputs:

- `.npz`: benchmark schema or arrays under `x`, `X`, `data`, `values`, `table`
- `.npy`: NumPy array shaped `[observations, variables]` or
  `[observations, variables, 2]`
- `.csv`, `.tsv`, `.txt`: numeric columns are used as variables
- `.parquet`, `.pkl`, `.pickle`: numeric pandas columns are used as variables

## Evaluate One Benchmark Directory

```bash
python -m tabcausal.cli predict-dir \
  --checkpoint checkpoints/tabcausal-base.pt \
  --input-dir /path/to/benchmark_directory \
  --output-dir results/tabcausal_demo \
  --mode auto \
  --threshold 0.5 \
  --batch-size 1 \
  --device cuda:0
```

Equivalent script entry:

```bash
python scripts/evaluate_directory.py \
  --checkpoint checkpoints/tabcausal-base.pt \
  --data-root /path/to/benchmark_directory \
  --output-dir results/tabcausal_demo \
  --mode auto \
  --threshold 0.5 \
  --batch-size 1 \
  --device cuda:0
```

Outputs:

- `raw_metrics.csv`: per-graph metrics when ground truth is present
- `summary.csv`: mean and standard deviation grouped by graph size
- `predictions.npz`: paths, logits, probabilities, and thresholded adjacencies
- `adjacency_csv/`: per-graph thresholded adjacency matrices
- `probability_csv/`: per-graph edge probability matrices

Add `--save-embeddings` to directory-level evaluation if you want
`embedding_npy/` and embedding arrays in `predictions.npz`. Add
`--no-matrix-exports` if you only want compact CSV summaries and the compressed
prediction archive.

## Baseline Evaluation

The release includes baseline support in `baselines/`. Use the wrapper around
the included per-method runners:

```bash
python -m baselines.run_paper_baselines --list

python -m baselines.run_paper_baselines \
  --method pc \
  --input-dir 'examples/example_data/[gp_hard]_obs' \
  --output-root results/baselines \
  --exp-name pc_example \
  --regime obs \
  --max-per-f 1 \
  --save-preds
```

Here `--max-per-f` limits the number of graphs evaluated for each graph size
`f`; use `-1` to evaluate all available graphs.

To verify that every included baseline runner can execute, run the all-baseline
smoke script on one generated graph. It keeps going after a method failure and
writes a pass/fail manifest plus per-method stdout/stderr logs:

```bash
python scripts/smoke_paper_baselines.py \
  --data-root examples/generated_smoke_7families \
  --output-root results/local_release_smoke/baselines \
  --datasets gp_hard \
  --max-per-f 1 \
  --timeout-seconds 300

cat results/local_release_smoke/baselines/baseline_smoke_manifest.csv
```

This smoke command uses a relaxed NoDAGS `min_samples` setting so the tiny
one-graph smoke dataset can exercise the runner. By default every baseline
method subprocess gets the same 300-second wall-clock timeout; override it with
`--timeout-seconds <seconds>` or pass `--timeout-seconds 0` to disable the
smoke timeout. The manifest records the timeout used for each method; DCDI's
internal per-graph limit follows the same value during smoke runs. The official
AVICI source is included under `baselines/paper_algorithms/avici_official`; if
you already have another official AVICI checkout, pass it with
`--avici-root /path/to/official/avici`. AVICI's pretrained weights should be
available in its cache, or downloadable by `huggingface-hub`. SEA checkpoints
can be supplied with `--sea-obs-checkpoint` and `--sea-int-checkpoint`. GIES
requires an R runtime with `Rscript` and the R package `pcalg`.

The baseline runners cover `AVICI`, `SEA`, `DAGMA`, `SDCD`, `DCDI`, `GIES`,
`CDIS`, `IGSP`, `NOTEARS`, `NOTEARS-MLP`, `LiNGAM`, `PC`, `DAS`,
`RandomRegress`, and `NoDAGS`. Several heavy research baselines require
optional Python/R dependencies or checkpoints; see `baselines/README.md` for
the exact dependency notes.

The metrics include F1, SHD, and SID when ground truth is available. SID is
computed with the official R `SID` package when R/SID is installed; otherwise
the code falls back to a Python parent-adjustment approximation and marks the
source in `SID_source`. Use `--no-official-sid` to skip the R call or `--no-sid`
to disable SID entirely.

## Evaluate the Seven Synthetic Families

For the public TabCausal reproduction path, generate all seven synthetic
families with the bundled benchmark data generator:

```bash
python scripts/generate_benchmark_suite.py \
  --output-root examples/generated_synthetic \
  --regimes obs,int \
  --observations 1000 \
  --interventions 200 \
  --f-values 5,10,20 \
  --graphs-per-f 10 \
  --seed 0 \
  --overwrite
```

This creates folders named like `[gp_hard]_obs` and `[gp_hard]_int`,
compatible with the evaluator below. The bundled data-generation components
live under `data_engine/`; their Python dependencies are covered by the
top-level `requirements.txt`.

For the observation-only regime, `--observations 1000` produces 1000
observational rows. For the mixed-interventional regime, the default split is
800 observational rows plus 200 interventional rows, i.e. still 1000 total
rows. To override that split, pass `--mixed-observations` explicitly.

This creates:

```text
[gp_hard]_obs, [gp_hard]_int
[gp_simple]_obs, [gp_simple]_int
[linear_gauss]_obs, [linear_gauss]_int
[linear_graph]_obs, [linear_graph]_int
[linear_nongauss]_obs, [linear_nongauss]_int
[mul_noise]_obs, [mul_noise]_int
[pfn]_obs, [pfn]_int
```

The unified generator is the only public benchmark data-generation entry point.
Use `--families` to generate a subset, for example
`--families gp_hard,pfn`.

To evaluate either this generated data or your own data root:

```bash
python scripts/evaluate_benchmark_suite.py \
  --checkpoint checkpoints/tabcausal-base.pt \
  --suite-data-root examples/generated_synthetic \
  --output-dir results/benchmark_suite \
  --threshold 0.5 \
  --max-per-f 1 \
  --batch-size 1 \
  --device cuda:0
```

By default this evaluates:

```text
gp_hard, gp_simple, linear_gauss, linear_graph,
linear_nongauss, mul_noise, pfn
```

under both `obs` and `int` regimes. Increase or remove `--max-per-f` for
larger evaluations.

For a full benchmark-style run with 100 generated graphs per graph size:

```bash
python scripts/generate_benchmark_suite.py \
  --output-root examples/generated_synthetic_full100 \
  --regimes obs,int \
  --observations 1000 \
  --interventions 200 \
  --f-values 5,10,20 \
  --graphs-per-f 100 \
  --seed 0 \
  --overwrite

python scripts/evaluate_benchmark_suite.py \
  --checkpoint checkpoints/tabcausal-base.pt \
  --suite-data-root examples/generated_synthetic_full100 \
  --output-dir results/benchmark_suite_full100 \
  --families gp_hard,gp_simple,linear_gauss,linear_graph,linear_nongauss,mul_noise,pfn \
  --regimes obs,int \
  --threshold 0.5 \
  --max-per-f 100 \
  --batch-size 1 \
  --device cuda:0 \
  --max-observations 1000

cat results/benchmark_suite_full100/benchmark_summary.csv
```

If you already have prepared benchmark folders, skip generation and point
`--suite-data-root` to that directory.

Outputs:

- one result directory per dataset, for example `[gp_hard]_obs/summary.csv`
- `manifest.csv`
- `benchmark_summary.csv`

To visualize the aggregate results:

```bash
python scripts/visualize_evaluation.py \
  --summary-csv results/benchmark_suite_full100/benchmark_summary.csv \
  --output-dir figures/benchmark_suite_full100
```

To summarize a generated suite at the data-distribution level:

```bash
python scripts/compare_suite_statistics.py \
  --suite-root examples/generated_synthetic \
  --output-csv results/generated_synthetic_statistics.csv
```

To compare two suites, pass both `--reference-suite` and `--candidate-suite`
instead of `--suite-root`.

## Expected NPZ Format

The loader accepts several common key names:

- Data: `x`, `X`, `data`, `x_obs`, `X_obs`, `obs`, `x_int`, `X_int`, `int`
- Graph: `g`, `G`, `graph`, `dag`, `adjacency`, `A`, `target`

Data may be shaped as:

- `[observations, variables]`
- `[observations, variables, 2]`

For the two-channel format, channel 0 stores values and channel 1 stores
intervention indicators.

## Visualization and Analysis

Single-file prediction outputs can be visualized directly:

```bash
python -m tabcausal.cli predict \
  --checkpoint checkpoints/tabcausal-base.pt \
  --input 'examples/example_data/[gp_hard]_obs/data_f5_000.npz' \
  --output outputs/example_prediction.npz \
  --include-embeddings \
  --device cpu

python scripts/visualize_prediction.py \
  --prediction outputs/example_prediction.npz \
  --output-dir figures/example_prediction \
  --prefix example
```

This writes:

- `example_probabilities.png`: directed edge probability heatmap
- `example_adjacency.png`: thresholded adjacency heatmap
- `example_embedding_pca.png`: PCA projection of final-layer node embeddings, when
  embeddings are present

The embedding PCA is qualitative. It is useful for inspecting whether node
representations organize variables by role, but it should not be interpreted as
a literal geometric reconstruction of the DAG.

For a directory-level prediction archive, use:

```bash
python scripts/visualize_evaluation.py \
  --prediction-npz results/tabcausal_demo/predictions.npz \
  --index 0 \
  --output-dir figures/tabcausal_demo
```

This writes `probability_heatmap.png`, `adjacency_heatmap.png`, and a compact
`prediction_overview.png`.

## Python API

```python
from tabcausal import TabCausalPredictor
from tabcausal.preprocessing import load_input_file

predictor = TabCausalPredictor(
    "checkpoints/tabcausal-base.pt",
    device="cuda:0",
    max_observations=2000,
)

example = load_input_file("data_f20_000.npz")
probabilities = predictor.predict_proba(example.x)[0]
adjacency = predictor.predict_adjacency(example.x, threshold=0.5)[0]
embeddings = predictor.predict_embeddings(example.x)[0]
```

## Memory and Runtime Options

- `--batch-size`: number of graph instances evaluated together. Use `1` if
  memory is tight.
- `--max-observations`: deterministic row subsampling before inference. This is
  useful for very large tables because the encoder attends over observations.
- `--observation-seed`: seed used when `--max-observations` is active.
- `--dtype`: `float32`, `float16`, or `bfloat16`.
- `--no-amp`: disables CUDA autocast.

Self-loops are always cleared in output adjacency matrices.
