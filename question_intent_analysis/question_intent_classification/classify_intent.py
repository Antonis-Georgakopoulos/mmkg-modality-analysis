#!/usr/bin/env python3
"""
Zero-Shot Question Intent Classification
==========================================

Classifies questions from both benchmarks (MMLongBench-Doc and LongDocURL)
into intent categories using GPT-5 with a zero-shot prompt.

Requirements:
  - OpenAI API key set as `openai_taxonomy_key` in .env or environment
  - pip install openai python-dotenv

Input files (relative to this script's directory):
  - MMLongBench: ../../mmlongbench/samples_2modalities.json
  - LongDocURL:  ../../longdocurl/LongDocURL_public_cleaned_2modalities.jsonl

Output files (saved in the parent directory, question_intent_analysis/):
  - samples_2modalities_zero_shot_gpt_5_final.json
  - LongDocURL_public_cleaned_2modalities_zero_shot_gpt_5_final.jsonl
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROMPT_FILE = SCRIPT_DIR / "prompt.txt"
OUTPUT_DIR = SCRIPT_DIR.parent  # question_intent_analysis/

REPO_ROOT = SCRIPT_DIR.parent.parent

BENCHMARKS = [
    {
        "name": "MMLongBench-Doc",
        "input": REPO_ROOT / "mmlongbench" / "data" / "samples_2modalities.json",
        "output": OUTPUT_DIR / "samples_2modalities_zero_shot_gpt_5_final.json",
        "format": "json",
        "intent_field": "question_intent",
    },
    {
        "name": "LongDocURL",
        "input": REPO_ROOT / "longdocurl" / "LongDocURL_public_cleaned_2modalities.jsonl",
        "output": OUTPUT_DIR / "LongDocURL_public_cleaned_2modalities_zero_shot_gpt_5_final.jsonl",
        "format": "jsonl",
        "intent_field": "question_intent",
    },
]

VALID_INTENTS = {
    "Description", "Process", "Advice", "Opinion", "Verification",
    "Attribute", "Reason", "Location", "Quantity", "Entity",
    "Language", "Temporal", "List", "Calculation", "Weather", "Resource",
}


# ── Classification ────────────────────────────────────────────────────

def load_prompt():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()


def classify_question(client: OpenAI, prompt_template: str, question: str) -> str:
    prompt = prompt_template.format(question)
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=32000,
    )
    intent = response.choices[0].message.content.strip().rstrip(".")
    # Strip "Intent:" prefix if present
    if intent.lower().startswith("intent:"):
        intent = intent[len("intent:"):].strip()
    # Try to match to a valid intent (case-insensitive)
    for valid in VALID_INTENTS:
        if intent.lower() == valid.lower():
            return valid
    # If multi-word, take the first word and try again
    first_word = intent.split()[0] if intent.split() else intent
    for valid in VALID_INTENTS:
        if first_word.lower() == valid.lower():
            return valid
    return intent


# ── I/O helpers ───────────────────────────────────────────────────────

def load_data(path: Path, fmt: str) -> list:
    if fmt == "json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:  # jsonl
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data


def save_data(data: list, path: Path, fmt: str):
    if fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    else:  # jsonl
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment / .env")

    client = OpenAI(api_key=api_key)
    prompt_template = load_prompt()

    for bench in BENCHMARKS:
        name = bench["name"]
        input_path = bench["input"]
        output_path = bench["output"]
        fmt = bench["format"]
        intent_field = bench["intent_field"]

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        if not input_path.exists():
            print(f"  INPUT NOT FOUND: {input_path}")
            print(f"  Skipping {name}.")
            continue

        data = load_data(input_path, fmt)
        total = len(data)
        print(f"  Classifying {total} questions with gpt-5 …\n")

        for i, item in enumerate(data, start=1):
            question = item["question"]
            intent = classify_question(client, prompt_template, question)
            item[intent_field] = intent
            print(f"  [{i}/{total}] {item.get('question_id', '?')} → {intent}")

        save_data(data, output_path, fmt)
        print(f"\n  Saved: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
