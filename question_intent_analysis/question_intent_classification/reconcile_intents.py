"""
Reconcile intent annotations based on judge consensus.

Where both judges (GPT-5.5 and Claude Opus 4.7) disagree with the original
annotation AND agree with each other on the replacement, update question_intent
in the source files and save as *_final.* versions.
"""
import json
from pathlib import Path

# ---------- Paths ----------
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent  # question_intent_analysis/

# ---------- Load judge outputs ----------
gpt_data = json.load(open(OUTPUT_DIR / "gpt5_5_judge_suggested_intents.json"))
claude_data = json.load(open(OUTPUT_DIR / "claude_opus_4_7_judge_suggested_intents.json"))

gpt_by_id = {r["question_id"]: r for r in gpt_data}
claude_by_id = {r["question_id"]: r for r in claude_data}

# ---------- Find consensus changes ----------
# Both judges disagree with annotation AND suggest the same new intent
consensus_changes = {}
for qid in gpt_by_id:
    g = gpt_by_id[qid]
    c = claude_by_id.get(qid)
    if c is None:
        continue
    if not g["agrees"] and not c["agrees"] and g["suggested_intent"] == c["suggested_intent"]:
        consensus_changes[qid] = {
            "old_intent": g["annotated_intent"],
            "new_intent": g["suggested_intent"],
        }

print(f"Found {len(consensus_changes)} consensus changes (both judges agree on a different intent)\n")

for qid, change in list(consensus_changes.items())[:5]:
    print(f"  {qid}: {change['old_intent']} → {change['new_intent']}")
print("  ...")

# ---------- Update samples file ----------
SAMPLES_IN = OUTPUT_DIR / "samples_2modalities_zero_shot_gpt_5_final.json"
SAMPLES_OUT = OUTPUT_DIR / "samples_2modalities_zero_shot_reconciled.json"

with open(SAMPLES_IN) as f:
    samples = json.load(f)

samples_updated = 0
for entry in samples:
    qid = entry["question_id"]
    if qid in consensus_changes:
        entry["question_intent"] = consensus_changes[qid]["new_intent"]
        samples_updated += 1

with open(SAMPLES_OUT, "w") as f:
    json.dump(samples, f, indent=4, ensure_ascii=False)

print(f"\n{SAMPLES_IN}: updated {samples_updated} intents → {SAMPLES_OUT}")

# ---------- Update LongDocURL file ----------
LONGDOC_IN = OUTPUT_DIR / "LongDocURL_public_cleaned_2modalities_zero_shot_gpt_5_final.jsonl"
LONGDOC_OUT = OUTPUT_DIR / "LongDocURL_public_cleaned_2modalities_zero_shot_reconciled.jsonl"

longdoc_updated = 0
output_lines = []
with open(LONGDOC_IN) as f:
    for line in f:
        entry = json.loads(line)
        qid = entry["question_id"]
        if qid in consensus_changes:
            entry["question_intent"] = consensus_changes[qid]["new_intent"]
            longdoc_updated += 1
        output_lines.append(json.dumps(entry, ensure_ascii=False))

with open(LONGDOC_OUT, "w") as f:
    f.write("\n".join(output_lines) + "\n")

print(f"{LONGDOC_IN}: updated {longdoc_updated} intents → {LONGDOC_OUT}")
print(f"\nTotal updated: {samples_updated + longdoc_updated} / {len(consensus_changes)} consensus changes applied")
