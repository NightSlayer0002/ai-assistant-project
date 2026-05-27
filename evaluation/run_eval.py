"""Entry point — runs evaluation and generates charts."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.runner import run_evaluation
from evaluation.visualize import generate_all_charts


def main():
    """Runs the full evaluation pipeline: evaluate, then visualize."""
    print("\n🚀 Starting full evaluation pipeline...\n")

    results = run_evaluation()

    if results is None:
        print("\n❌ Evaluation failed. Skipping chart generation.")
        return

    print("\n")
    generate_all_charts()

    print("\n✅ Full evaluation pipeline complete!")
    print("   Results: evaluation/results/evaluation_results.json")
    print("   Charts:  evaluation/results/*.png")


if __name__ == "__main__":
    main()
