#!/usr/bin/env bash
# Full pipeline. The LLM step runs only if ANTHROPIC_API_KEY is set.
set -e
cd "$(dirname "$0")"
python src/generate_invoices.py
python src/extract_rules.py
if [ -n "$ANTHROPIC_API_KEY" ]; then python src/extract_llm.py; else echo "Skipping LLM step (no ANTHROPIC_API_KEY)"; fi
python src/process.py
