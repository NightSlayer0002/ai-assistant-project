"""Evaluation runner — orchestrates prompts, assistants, and the judge."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from oss_assistant.assistant import OSSAssistant
from frontier_assistant.assistant import FrontierAssistant
from evaluation.prompts import FACTUAL_PROMPTS, ADVERSARIAL_PROMPTS, BIAS_PROMPTS
from evaluation.judge import judge_response
from observability.logger import SimpleLogger

RESULTS_DIR = "evaluation/results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "evaluation_results.json")
DELAY_BETWEEN_CALLS_SECONDS = 2


def run_evaluation():
    """Runs all prompts through both assistants, judges responses, and saves results."""
    print("=" * 60)
    print("STARTING EVALUATION")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger = SimpleLogger()

    # Initialize both assistants
    try:
        oss = OSSAssistant()
        print("✅ OSS Assistant (Qwen2.5) initialized")
    except ValueError as e:
        print(f"❌ Could not initialize OSS Assistant: {e}")
        print("   Skipping OSS evaluations — will use placeholder responses.")
        oss = None

    try:
        frontier = FrontierAssistant()
        print("✅ Frontier Assistant (Gemini) initialized")
    except ValueError as e:
        print(f"❌ Could not initialize Frontier Assistant: {e}")
        print("   Skipping Frontier evaluations — will use placeholder responses.")
        frontier = None

    all_prompts = []
    for p in FACTUAL_PROMPTS:
        all_prompts.append({**p, "category": "factual"})
    for p in ADVERSARIAL_PROMPTS:
        all_prompts.append({**p, "category": "adversarial"})
    for p in BIAS_PROMPTS:
        all_prompts.append({**p, "category": "bias"})

    results = []

    for i, prompt_data in enumerate(all_prompts):
        prompt_id = prompt_data["id"]
        prompt_text = prompt_data["prompt"]
        category = prompt_data["category"]

        print(f"\n[{i+1}/{len(all_prompts)}] Evaluating: {prompt_text[:60]}...")

        # Get OSS response
        if oss:
            oss.clear_history()
            start_time = time.time()
            try:
                oss_response = oss.send_message(prompt_text)
            except Exception as e:
                oss_response = f"Error: {str(e)}"
            oss_latency = (time.time() - start_time) * 1000
            logger.log_interaction("Qwen2.5-7B-Instruct", prompt_text, oss_response, oss_latency)
        else:
            oss_response = "OSS assistant not available — missing API key."
            oss_latency = 0.0

        print(f"  OSS response ({oss_latency:.0f}ms): {oss_response[:80]}...")
        time.sleep(DELAY_BETWEEN_CALLS_SECONDS)

        # Get Frontier response
        if frontier:
            frontier.clear_history()
            start_time = time.time()
            try:
                frontier_response = frontier.send_message(prompt_text)
            except Exception as e:
                frontier_response = f"Error: {str(e)}"
            frontier_latency = (time.time() - start_time) * 1000
            logger.log_interaction("gemini-2.5-flash", prompt_text, frontier_response, frontier_latency)
        else:
            frontier_response = "Frontier assistant not available — missing API key."
            frontier_latency = 0.0

        print(f"  Frontier response ({frontier_latency:.0f}ms): {frontier_response[:80]}...")
        time.sleep(DELAY_BETWEEN_CALLS_SECONDS)

        # Judge both responses
        print("  Judging responses...")

        oss_scores = judge_response(prompt_text, oss_response, category)
        time.sleep(DELAY_BETWEEN_CALLS_SECONDS)

        frontier_scores = judge_response(prompt_text, frontier_response, category)
        time.sleep(DELAY_BETWEEN_CALLS_SECONDS)

        result = {
            "prompt_id": prompt_id,
            "category": category,
            "prompt": prompt_text,
            "oss": {
                "response": oss_response,
                "latency_ms": round(oss_latency, 2),
                "scores": oss_scores
            },
            "frontier": {
                "response": frontier_response,
                "latency_ms": round(frontier_latency, 2),
                "scores": frontier_scores
            }
        }
        results.append(result)

        print(f"  OSS scores:      R={oss_scores['relevance']} A={oss_scores['accuracy']} S={oss_scores['safety']}")
        print(f"  Frontier scores:  R={frontier_scores['relevance']} A={frontier_scores['accuracy']} S={frontier_scores['safety']}")

    summary = calculate_summary(results)

    final_output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_prompts": len(all_prompts),
            "oss_model": "Qwen2.5-7B-Instruct",
            "frontier_model": "gemini-2.5-flash",
            "judge_model": "Qwen2.5-7B-Instruct"
        },
        "summary": summary,
        "detailed_results": results
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"EVALUATION COMPLETE")
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"{'=' * 60}")
    print(f"\n📊 SUMMARY:")
    print(f"  OSS Average Scores:      Relevance={summary['oss']['avg_relevance']:.1f}  "
          f"Accuracy={summary['oss']['avg_accuracy']:.1f}  Safety={summary['oss']['avg_safety']:.1f}")
    print(f"  Frontier Average Scores:  Relevance={summary['frontier']['avg_relevance']:.1f}  "
          f"Accuracy={summary['frontier']['avg_accuracy']:.1f}  Safety={summary['frontier']['avg_safety']:.1f}")
    print(f"  OSS Avg Latency:      {summary['oss']['avg_latency_ms']:.0f}ms")
    print(f"  Frontier Avg Latency: {summary['frontier']['avg_latency_ms']:.0f}ms")

    return final_output


def calculate_summary(results: list) -> dict:
    """Calculates average scores and latencies for each assistant."""
    oss_scores = {"relevance": [], "accuracy": [], "safety": [], "latency": []}
    frontier_scores = {"relevance": [], "accuracy": [], "safety": [], "latency": []}

    for r in results:
        oss_scores["relevance"].append(r["oss"]["scores"]["relevance"])
        oss_scores["accuracy"].append(r["oss"]["scores"]["accuracy"])
        oss_scores["safety"].append(r["oss"]["scores"]["safety"])
        oss_scores["latency"].append(r["oss"]["latency_ms"])

        frontier_scores["relevance"].append(r["frontier"]["scores"]["relevance"])
        frontier_scores["accuracy"].append(r["frontier"]["scores"]["accuracy"])
        frontier_scores["safety"].append(r["frontier"]["scores"]["safety"])
        frontier_scores["latency"].append(r["frontier"]["latency_ms"])

    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "oss": {
            "avg_relevance": round(safe_avg(oss_scores["relevance"]), 2),
            "avg_accuracy": round(safe_avg(oss_scores["accuracy"]), 2),
            "avg_safety": round(safe_avg(oss_scores["safety"]), 2),
            "avg_latency_ms": round(safe_avg(oss_scores["latency"]), 2),
        },
        "frontier": {
            "avg_relevance": round(safe_avg(frontier_scores["relevance"]), 2),
            "avg_accuracy": round(safe_avg(frontier_scores["accuracy"]), 2),
            "avg_safety": round(safe_avg(frontier_scores["safety"]), 2),
            "avg_latency_ms": round(safe_avg(frontier_scores["latency"]), 2),
        }
    }


if __name__ == "__main__":
    run_evaluation()
