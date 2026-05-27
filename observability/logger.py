"""JSON interaction logger — records prompts, responses, latency, and token estimates to JSONL."""

import json
import os
from datetime import datetime

DEFAULT_LOG_FILE = "observability/logs/interactions.jsonl"


class SimpleLogger:
    """Logs AI interactions to a JSONL file for observability and cost tracking."""
    
    def __init__(self, log_file: str = DEFAULT_LOG_FILE):
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
    
    def log_interaction(self, model_name: str, prompt: str, response: str, latency_ms: float):
        """Records one interaction to the log file."""
        total_words = len(prompt.split()) + len(response.split())
        estimated_tokens = int(total_words * 1.3)  # rough approximation
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "prompt_preview": prompt[:100],
            "response_length": len(response),
            "latency_ms": round(latency_ms, 2),
            "estimated_tokens": estimated_tokens
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def get_latency_summary(self) -> dict:
        """Returns latency statistics grouped by model."""
        if not os.path.exists(self.log_file):
            return {}
        
        latencies_by_model = {}
        
        with open(self.log_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        model = entry["model_name"]
                        latency = entry["latency_ms"]
                        
                        if model not in latencies_by_model:
                            latencies_by_model[model] = []
                        latencies_by_model[model].append(latency)
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        summary = {}
        for model, latencies in latencies_by_model.items():
            summary[model] = {
                "avg": round(sum(latencies) / len(latencies), 2),
                "min": round(min(latencies), 2),
                "max": round(max(latencies), 2),
                "count": len(latencies)
            }
        
        return summary
