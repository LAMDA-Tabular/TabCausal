# TabCausal

TabCausal is a pretrained model for tabular causal discovery. Given an
observational or mixed observational/interventional table, it predicts a
directed causal graph as edge probabilities and a thresholded adjacency matrix.

This repository provides the public inference package, a release checkpoint,
synthetic benchmark tools, and wrappers for paper baselines.

## Links

- Paper: TabCausal: Pretraining Across Causal Environments for Tabular Causal Discovery
- arXiv: https://arxiv.org/abs/2605.31156
- Current checkpoint: [`checkpoints/tabcausal-base.pt`](https://huggingface.co/LAMDA-Tabular/TabCausal)
- Package entry point: `tabcausal`

## Repository Layout

```text
tabcausal/      Model, preprocessing, inference API, CLI, and metrics
checkpoints/    Released TabCausal checkpoint
examples/       Small data and prediction examples
scripts/        Benchmark generation, evaluation, and visualization scripts
data_engine/    Synthetic benchmark data-generation components
baselines/      Paper baseline wrappers and notes
```

## Requirements

TabCausal requires Python 3.10+, PyTorch 2.1+, and common scientific Python
packages. Install the full dependency list with:

```bash
pip install -r requirements.txt
```

For editable package installation:

```bash
pip install -e .
```

Some baselines require additional R/JAX dependencies or external checkpoints.
See [`baselines/README.md`](baselines/README.md) for details.

## Checkpoint

We provide `checkpoints/tabcausal-base.pt` as the current public release
checkpoint. The same checkpoint is hosted on Hugging Face:

https://huggingface.co/LAMDA-Tabular/TabCausal

The examples below use the local checkpoint path. We plan to update the
released checkpoint as newer TabCausal versions become available.

## Quick Start

Run a small CPU smoke test:

```bash
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
  --device cpu

cat examples/example_results/summary.csv
```

Use `--device cuda:0` for GPU inference.

## Predict Your Own Data

Predict one graph:

```bash
python -m tabcausal.cli predict \
  --checkpoint checkpoints/tabcausal-base.pt \
  --input /path/to/data.npz \
  --output outputs/prediction.npz \
  --device cuda:0 \
  --threshold 0.5
```

The output contains edge logits, edge probabilities, and a thresholded
adjacency matrix. `probabilities[i, j]` and `adjacency[i, j]` correspond to the
directed edge `i -> j`. Self-loops are always cleared.

Python API:

```python
from tabcausal import TabCausalPredictor
from tabcausal.preprocessing import load_input_file

predictor = TabCausalPredictor("checkpoints/tabcausal-base.pt", device="cuda:0")
example = load_input_file("data_f20_000.npz")
probabilities = predictor.predict_proba(example.x)[0]
adjacency = predictor.predict_adjacency(example.x, threshold=0.5)[0]
```

## Input and Output Format

TabCausal accepts `.npz`, `.npy`, `.csv`, `.tsv`, `.txt`, `.parquet`, `.pkl`,
and `.pickle` inputs.

Input arrays can be shaped as:

```text
[observations, variables]
[observations, variables, 2]
```

For the two-channel format, channel 0 stores observed values and channel 1
stores binary intervention indicators. For plain table inputs with interventions,
pass a same-shaped binary mask with `--intervention-input`.

For `.npz` files, common data keys such as `x`, `X`, `data`, `values`, `table`,
`obs`, and `int` are recognized. Ground-truth graph keys may be named `g`, `G`,
`graph`, `dag`, `adjacency`, `A`, or `target`.

Prediction outputs contain:

```text
logits          raw directed-edge logits
probabilities   sigmoid edge probabilities
adjacency       thresholded directed graph
embeddings      optional final-layer node embeddings
```

## Benchmark Evaluation

### Evaluate an Existing Benchmark Directory

Use `predict-dir` for a directory of benchmark files:

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

This writes `raw_metrics.csv`, `summary.csv`, `predictions.npz`, and optional
matrix exports when ground truth is available.

### Reproduce the Synthetic Benchmark Suite

Generate the seven public synthetic benchmark families:

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

Evaluate TabCausal on the generated suite:

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

The generator covers `gp_hard`, `gp_simple`, `linear_gauss`, `linear_graph`,
`linear_nongauss`, `mul_noise`, and `pfn` under both observational (`obs`) and
mixed interventional (`int`) regimes. For fuller runs, increase
`--graphs-per-f` and `--max-per-f`.

### Baselines

Baseline wrappers are provided for:

```text
AVICI, SEA, DAGMA, SDCD, DCDI, GIES, CDIS, IGSP,
NOTEARS, NOTEARS-MLP, LiNGAM, PC, DAS, RandomRegress, NoDAGS
```

List available methods:

```bash
python -m baselines.run_paper_baselines --list
```

See [`baselines/README.md`](baselines/README.md) for method-specific
dependencies, R requirements, AVICI/SEA checkpoint notes, and smoke tests.

## Visualization

Visualization scripts are available for single predictions and benchmark
summaries:

```bash
python scripts/visualize_prediction.py \
  --prediction outputs/prediction.npz \
  --output-dir figures/example_prediction \
  --prefix example

python scripts/visualize_evaluation.py \
  --summary-csv results/benchmark_suite/benchmark_summary.csv \
  --output-dir figures/benchmark_suite
```

## Runtime Tips

- Use `--batch-size 1` if GPU memory is tight.
- Use `--max-observations` to deterministically subsample rows before inference.
- Use `--dtype float16` or `--dtype bfloat16` for lower-memory CUDA inference.
- SID is computed with the official R `SID` package when available; otherwise
  a Python approximation is used and recorded in `SID_source`.

## Citation

If you use TabCausal, please cite:

```bibtex
@article{li2026tabcausal,
  title={TabCausal: Pretraining Across Causal Environments for Tabular Causal Discovery},
  author={Li, Zi-Rong and Liu, Si-Yang and Wang, Tian-Zuo and Ye, Han-Jia},
  journal={arXiv preprint arXiv:2605.31156},
  year={2026}
}
```

## License

Please see the repository license file. If no license file is included in your
checkout, consult the authors before redistribution or reuse.
