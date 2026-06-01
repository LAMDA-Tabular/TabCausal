# Examples

Create a small local `gp_hard` benchmark example:

```bash
python examples/make_example_data.py \
  --output-root examples/example_data \
  --num-graphs 10 \
  --variables 5 \
  --observations 1000
```

Run a CPU smoke test:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"

python -m tabcausal.cli predict-dir \
  --checkpoint checkpoints/tabcausal-base.pt \
  --input-dir 'examples/example_data/[gp_hard]_obs' \
  --output-dir examples/example_results \
  --mode auto \
  --threshold 0.5 \
  --batch-size 16 \
  --device cpu \
  --save-embeddings
```

Check that the summary schema and output files were written:

```bash
head -n 1 examples/example_results/summary.csv
ls -lh examples/example_results
```

Visualize the first prediction from the directory archive:

```bash
python scripts/visualize_evaluation.py \
  --prediction-npz examples/example_results/predictions.npz \
  --index 0 \
  --output-dir examples/example_results/figures

find examples/example_results/figures -type f | sort
```

The example data uses the same `gp_hard` observational setting as the synthetic
benchmark family, restricted to `f=5` for a fast local smoke test.

EMA checkpoint weights are preferred by default when present, matching the
reported benchmark runs.
