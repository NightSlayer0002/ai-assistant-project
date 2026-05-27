"""Visualization — generates comparison charts from evaluation results."""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = "evaluation/results/evaluation_results.json"
CHARTS_DIR = "evaluation/results"

# Colorblind-friendly palette
OSS_COLOR = "#2196F3"
FRONTIER_COLOR = "#FF9800"


def load_results() -> dict:
    """Loads evaluation results from JSON. Returns None if file missing."""
    if not os.path.exists(RESULTS_FILE):
        print(f"❌ Results file not found: {RESULTS_FILE}")
        print("   Run the evaluation first: python -m evaluation.runner")
        return None

    with open(RESULTS_FILE, "r") as f:
        return json.load(f)


def create_score_comparison_chart(results: dict):
    """Grouped bar chart comparing average scores between OSS and Frontier."""
    summary = results["summary"]

    dimensions = ["Relevance", "Accuracy", "Safety"]

    oss_scores = [
        summary["oss"]["avg_relevance"],
        summary["oss"]["avg_accuracy"],
        summary["oss"]["avg_safety"],
    ]
    frontier_scores = [
        summary["frontier"]["avg_relevance"],
        summary["frontier"]["avg_accuracy"],
        summary["frontier"]["avg_safety"],
    ]

    x = np.arange(len(dimensions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, oss_scores, width, label="OSS (Qwen2.5)", color=OSS_COLOR)
    bars2 = ax.bar(x + width/2, frontier_scores, width, label="Frontier (Gemini)", color=FRONTIER_COLOR)

    ax.set_ylabel("Average Score (1-5)", fontsize=12)
    ax.set_title("Model Comparison: Average Scores by Dimension", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    plt.tight_layout()

    chart_path = os.path.join(CHARTS_DIR, "score_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"✅ Score comparison chart saved to: {chart_path}")


def create_latency_comparison_chart(results: dict):
    """Bar chart comparing average response latency between models."""
    summary = results["summary"]

    models = ["OSS (Qwen2.5)", "Frontier (Gemini)"]
    latencies = [
        summary["oss"]["avg_latency_ms"],
        summary["frontier"]["avg_latency_ms"],
    ]
    colors = [OSS_COLOR, FRONTIER_COLOR]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(models, latencies, color=colors, width=0.5)

    ax.set_ylabel("Average Latency (ms)", fontsize=12)
    ax.set_title("Model Comparison: Average Response Time", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.0f}ms",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.tight_layout()

    chart_path = os.path.join(CHARTS_DIR, "latency_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"✅ Latency comparison chart saved to: {chart_path}")


def create_category_breakdown_chart(results: dict):
    """Grouped bar chart showing scores broken down by category."""
    detailed = results["detailed_results"]

    categories = ["factual", "adversarial", "bias"]
    category_labels = ["Factual", "Adversarial", "Bias"]

    oss_category_scores = {}
    frontier_category_scores = {}

    for cat in categories:
        cat_results = [r for r in detailed if r["category"] == cat]

        if cat_results:
            oss_avg = sum(
                (r["oss"]["scores"]["relevance"] + r["oss"]["scores"]["accuracy"] + r["oss"]["scores"]["safety"]) / 3
                for r in cat_results
            ) / len(cat_results)

            frontier_avg = sum(
                (r["frontier"]["scores"]["relevance"] + r["frontier"]["scores"]["accuracy"] + r["frontier"]["scores"]["safety"]) / 3
                for r in cat_results
            ) / len(cat_results)
        else:
            oss_avg = 0
            frontier_avg = 0

        oss_category_scores[cat] = round(oss_avg, 2)
        frontier_category_scores[cat] = round(frontier_avg, 2)

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    oss_vals = [oss_category_scores[c] for c in categories]
    frontier_vals = [frontier_category_scores[c] for c in categories]

    bars1 = ax.bar(x - width/2, oss_vals, width, label="OSS (Qwen2.5)", color=OSS_COLOR)
    bars2 = ax.bar(x + width/2, frontier_vals, width, label="Frontier (Gemini)", color=FRONTIER_COLOR)

    ax.set_ylabel("Average Overall Score (1-5)", fontsize=12)
    ax.set_title("Model Comparison: Scores by Category", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(category_labels, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    plt.tight_layout()

    chart_path = os.path.join(CHARTS_DIR, "category_breakdown.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"✅ Category breakdown chart saved to: {chart_path}")


def generate_all_charts():
    """Generates all visualization charts from evaluation results."""
    print("=" * 60)
    print("GENERATING VISUALIZATION CHARTS")
    print("=" * 60)

    results = load_results()
    if results is None:
        return

    create_score_comparison_chart(results)
    create_latency_comparison_chart(results)
    create_category_breakdown_chart(results)

    print(f"\n✅ All charts saved to: {CHARTS_DIR}/")
    print("   - score_comparison.png")
    print("   - latency_comparison.png")
    print("   - category_breakdown.png")


if __name__ == "__main__":
    generate_all_charts()
