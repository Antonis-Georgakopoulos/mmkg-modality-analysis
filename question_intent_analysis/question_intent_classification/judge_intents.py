import json
import os
import re
from pathlib import Path

from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# ---------- Paths ----------
SCRIPT_DIR = Path(__file__).parent
PROMPT_FILE = SCRIPT_DIR / "judge_prompt.txt"
OUTPUT_DIR = SCRIPT_DIR.parent  # question_intent_analysis/


def load_prompt() -> str:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()

VALID_INTENTS = {
    "Description", "Process", "Advice", "Opinion", "Verification",
    "Attribute", "Reason", "Location", "Quantity", "Entity",
    "Language", "Temporal", "List", "Calculation", "Weather", "Resource",
}

# ---------- Model configs ----------
OPENAI_MODEL = "gpt-5.5"
ANTHROPIC_MODEL = "claude-opus-4-7"

# ---------- I/O ----------
SAMPLES_FILE = OUTPUT_DIR / "samples_2modalities_zero_shot_gpt_5_final.json"
LONGDOCURL_FILE = OUTPUT_DIR / "LongDocURL_public_cleaned_2modalities_zero_shot_gpt_5_final.jsonl"

GPT_OUTPUT_FILE = OUTPUT_DIR / "gpt5_5_judge_suggested_intents.json"
CLAUDE_OUTPUT_FILE = OUTPUT_DIR / "claude_opus_4_7_judge_suggested_intents.json"


def load_questions():
    """Load questions from both datasets, returning a unified list of dicts
    with keys: question_id, question, annotated_intent, source_file."""
    questions = []

    # 1. samples JSON
    with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
        samples = json.load(f)
    for item in samples:
        questions.append({
            "question_id": item["question_id"],
            "question": item["question"],
            "annotated_intent": item["question_intent"],
            "source_file": str(SAMPLES_FILE),
        })

    # 2. LongDocURL JSONL
    with open(LONGDOCURL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            questions.append({
                "question_id": item["question_id"],
                "question": item["question"],
                "annotated_intent": item["question_intent"],
                "source_file": str(LONGDOCURL_FILE),
            })

    return questions


def parse_intent(raw: str) -> str:
    """Normalize a model response to a valid intent string.

    Strategy:
    1. If the response is a single valid intent (possibly with a prefix), return it.
    2. Otherwise scan the full response for valid intents; return the *last*
       one mentioned (models tend to state the final answer at the end).
    3. Fall back to the raw text if nothing matches.
    """
    text = raw.strip().rstrip(".")
    # Strip common prefixes
    for prefix in ("suggested intent:", "intent:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    # Exact match (case-insensitive)
    for valid in VALID_INTENTS:
        if text.lower() == valid.lower():
            return valid
    # First-word fallback
    first_word = text.split()[0] if text.split() else text
    for valid in VALID_INTENTS:
        if first_word.lower() == valid.lower():
            return valid
    # Scan full response: find the last valid intent mentioned as a whole word
    last_match = None
    lower_text = raw.lower()
    for valid in VALID_INTENTS:
        pattern = r'\b' + re.escape(valid.lower()) + r'\b'
        for m in re.finditer(pattern, lower_text):
            if last_match is None or m.start() > last_match[1]:
                last_match = (valid, m.start())
    if last_match:
        return last_match[0]
    return text


# ---------- Model callers ----------

def call_openai(client: OpenAI, prompt_template: str, question: str, annotated_intent: str) -> str:
    prompt = prompt_template.format(question=question, annotated_intent=annotated_intent)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=4096,
    )
    return parse_intent(response.choices[0].message.content)


def call_anthropic(client: Anthropic, prompt_template: str, question: str, annotated_intent: str) -> str:
    prompt = prompt_template.format(question=question, annotated_intent=annotated_intent)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return parse_intent(response.content[0].text)


def run_judge(model_name: str, call_fn, client, prompt_template: str, questions: list, output_file: str):
    """Run a single model as judge over all questions and write results."""
    results = []
    total = len(questions)
    print(f"\n{'='*60}")
    print(f"Running {model_name} as judge on {total} questions …")
    print(f"{'='*60}")

    for i, q in enumerate(questions, start=1):
        try:
            suggested = call_fn(client, prompt_template, q["question"], q["annotated_intent"])
        except Exception as e:
            print(f"  [ERROR] {q['question_id']}: {e}")
            suggested = f"ERROR: {e}"

        agrees = suggested.lower() == q["annotated_intent"].lower()

        results.append({
            "question_id": q["question_id"],
            "source_file": q["source_file"],
            "question": q["question"],
            "annotated_intent": q["annotated_intent"],
            "suggested_intent": suggested,
            "agrees": agrees,
        })

        status = "✓" if agrees else "✗"
        print(f"  [{i}/{total}] {status}  {q['question_id']}  "
              f"annotated={q['annotated_intent']}  suggested={suggested}")

    # Write results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    agree_count = sum(1 for r in results if r["agrees"])
    print(f"\n{model_name} finished: {agree_count}/{total} agreed "
          f"({100*agree_count/total:.1f}%). Results → {output_file}")

    return results


def main():
    # --- API keys ---
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in .env")

    openai_client = OpenAI(api_key=openai_key)
    anthropic_client = Anthropic(api_key=anthropic_key)

    # --- Load prompt & data ---
    prompt_template = load_prompt()
    questions = load_questions()
    print(f"Loaded {len(questions)} questions total "
          f"({SAMPLES_FILE.name} + {LONGDOCURL_FILE.name})")

    # --- Run both judges ---
    if not GPT_OUTPUT_FILE.exists():
        run_judge(OPENAI_MODEL, call_openai, openai_client, prompt_template, questions, GPT_OUTPUT_FILE)
    else:
        print(f"Skipping {OPENAI_MODEL} — {GPT_OUTPUT_FILE} already exists.")

    if not CLAUDE_OUTPUT_FILE.exists():
        run_judge(ANTHROPIC_MODEL, call_anthropic, anthropic_client, prompt_template, questions, CLAUDE_OUTPUT_FILE)
    else:
        print(f"Skipping {ANTHROPIC_MODEL} — {CLAUDE_OUTPUT_FILE} already exists.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
