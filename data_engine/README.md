# Data Engine

This directory contains the synthetic data generation components used to create
benchmark datasets for TabCausal.

The inference package in `tabcausal/` does not depend on this directory. Use it
only when you want to regenerate synthetic benchmark datasets.

## Example

Generate all seven synthetic benchmark families:

```bash
python scripts/generate_benchmark_suite.py \
  --output-root examples/generated_synthetic \
  --regimes obs,int \
  --f-values 5,10,20 \
  --graphs-per-f 10 \
  --observations 1000 \
  --interventions 200 \
  --seed 0 \
  --overwrite
```

`scripts/generate_benchmark_suite.py` is the single public generator for all
seven families and writes a consistent benchmark directory layout such as
`[gp_hard]_obs` and `[gp_hard]_int`. Use `--families` to generate a subset, for
example `--families gp_hard,pfn`.
