"""Minimal TabCausal inference example."""

from __future__ import annotations

import argparse

from tabcausal import TabCausalPredictor
from tabcausal.preprocessing import load_input_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    example = load_input_file(args.input)
    predictor = TabCausalPredictor(args.checkpoint, device=args.device)
    probabilities = predictor.predict_proba(example.x)[0]
    adjacency = predictor.predict_adjacency(example.x, threshold=0.5)[0]
    print("probabilities:")
    print(probabilities)
    print("adjacency:")
    print(adjacency)


if __name__ == "__main__":
    main()
