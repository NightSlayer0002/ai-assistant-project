# 🤖 AI Personal Assistant: Comparative Study

> **💰 Completely Free**: Both APIs (Google AI and HuggingFace) have free tiers. No credit card needed.

## Overview

This project builds and evaluates **two AI personal assistants**:
1. **OSS Assistant** - Qwen2.5-7B (open-source) via HuggingFace Inference API
2. **Frontier Assistant** - Gemini 2.5 Flash (proprietary) via Google AI API

Both support **multi-turn conversations** with short-term memory. The Frontier assistant also supports **tool use** (time lookup, calculator).

We evaluate both on **hallucination**, **bias**, and **content safety** using an automated **LLM-as-judge** framework (Qwen2.5-7B as judge), then visualize the comparison with charts.

## 📁 Project Structure

```
ai-assistant-project/
├── requirements.txt
├── .env.example
│
├── oss_assistant/
│   ├── assistant.py          # Qwen2.5-7B via HuggingFace InferenceClient
│   ├── guardrails.py         # Input/output safety filters
│   └── app.py                # Streamlit chat UI
│
├── frontier_assistant/
│   ├── assistant.py          # Gemini 2.5 Flash via google-genai SDK
│   ├── tools.py              # Function calling (time, calculator)
│   └── app.py                # Streamlit chat UI
│
├── evaluation/
│   ├── prompts.py            # 13 test prompts (factual, adversarial, bias)
│   ├── judge.py              # LLM-as-judge using Qwen2.5-7B
│   ├── runner.py             # Evaluation orchestrator
│   ├── visualize.py          # Chart generation (matplotlib)
│   ├── run_eval.py           # Entry point: runs everything
│   └── results/              # Generated scores + charts
│
└── observability/
    └── logger.py             # JSONL interaction logger
```

## 🚀 Setup

### Prerequisites
- Python 3.10+
- Google AI API key → [Get free key](https://aistudio.google.com/apikey)
- HuggingFace API key → [Get free key](https://huggingface.co/settings/tokens)

### Installation
```bash
git clone <repo-url>
cd ai-assistant-project

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

cp .env.example .env
# Add your API keys to .env
```

## ▶️ Usage

```bash
# Run the OSS assistant
streamlit run oss_assistant/app.py

# Run the Frontier assistant
streamlit run frontier_assistant/app.py

# Run the full evaluation pipeline
python evaluation/run_eval.py
```

## 📊 Evaluation Results

Both models were tested on 13 prompts across 3 categories and scored by Qwen2.5-7B as an LLM judge:

| Metric | OSS (Qwen2.5-7B) | Frontier (Gemini 2.5 Flash) |
|--------|-------------------|----------------------------|
| Relevance | 3.8 / 5 | 3.5 / 5 |
| Accuracy | 4.5 / 5 | 4.3 / 5 |
| Safety | 4.5 / 5 | 4.5 / 5 |
| Avg Latency | ~1446ms | ~789ms |

**Key Findings:**
- Both models score similarly on safety, modern OSS models have strong alignment
- Frontier model is ~2x faster due to optimized infrastructure
- OSS model edges ahead slightly on relevance and accuracy for straightforward tasks

## 🏗️ Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| UI Framework | Streamlit | Built-in chat components, minimal boilerplate |
| OSS Model | Qwen2.5-7B via HuggingFace | Free tier, instruction-tuned, no GPU needed |
| Frontier Model | Gemini 2.5 Flash | Free tier, fast, native function calling |
| Memory | List of dicts / Content objects | Matches API format directly |
| Judge Model | Qwen2.5-7B via HuggingFace | Avoids self-evaluation bias (using a different model than the Frontier assistant) |
| Evaluation | LLM-as-judge + custom prompts | Scalable, consistent, reproducible |

## ⚠️ Known Limitations

1. **Basic guardrails** - keyword-based filtering is easy to bypass
2. **Small eval set** - 13 prompts is sufficient for demonstration but not rigorous benchmarking
3. **Session-only memory** - conversation resets on app restart
4. **No streaming** - responses appear all at once
