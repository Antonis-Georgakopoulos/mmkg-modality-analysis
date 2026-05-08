import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# ---------- Paths ----------
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR.parent  # question_intent_analysis/

# ---------- Load & align ----------
gpt_data = json.load(open(OUTPUT_DIR / "gpt5_5_judge_suggested_intents.json"))
claude_data = json.load(open(OUTPUT_DIR / "claude_opus_4_7_judge_suggested_intents.json"))

# Index by question_id
gpt_by_id = {r["question_id"]: r for r in gpt_data}
claude_by_id = {r["question_id"]: r for r in claude_data}

# Align on shared question_ids
shared_ids = sorted(set(gpt_by_id) & set(claude_by_id))
print(f"Shared question IDs: {len(shared_ids)}")

# Filter out rows where either judge had an ERROR
aligned = []
for qid in shared_ids:
    g = gpt_by_id[qid]
    c = claude_by_id[qid]
    aligned.append({
        "question_id": qid,
        "question": g["question"],
        "annotated_intent": g["annotated_intent"],
        "gpt_suggested": g["suggested_intent"],
        "claude_suggested": c["suggested_intent"],
        "gpt_agrees_annotator": g["agrees"],
        "claude_agrees_annotator": c["agrees"],
    })

print(f"Total: {len(aligned)} questions\n")

gpt_labels = [r["gpt_suggested"] for r in aligned]
claude_labels = [r["claude_suggested"] for r in aligned]

# ---------- 1. Cohen's kappa ----------
kappa = cohen_kappa_score(gpt_labels, claude_labels)
print("=" * 60)
print(f"Cohen's kappa (GPT-5.5 vs Claude Opus 4.7): {kappa:.4f}")
print("=" * 60)

# Raw agreement
raw_agree = sum(1 for g, c in zip(gpt_labels, claude_labels) if g == c)
print(f"Raw agreement: {raw_agree}/{len(aligned)} ({100 * raw_agree / len(aligned):.1f}%)\n")

# ---------- 2. Among items where BOTH judges suggested a change ----------
both_changed = [r for r in aligned
                if not r["gpt_agrees_annotator"] and not r["claude_agrees_annotator"]]

print("=" * 60)
print("Among items where BOTH judges suggested a change from the annotation:")
print("=" * 60)
print(f"Total such items: {len(both_changed)}")

same_change = [r for r in both_changed if r["gpt_suggested"] == r["claude_suggested"]]
diff_change = [r for r in both_changed if r["gpt_suggested"] != r["claude_suggested"]]

print(f"  Same suggested change: {len(same_change)} ({100 * len(same_change) / len(both_changed):.1f}%)")
print(f"  Different suggested change: {len(diff_change)} ({100 * len(diff_change) / len(both_changed):.1f}%)")

if same_change:
    print(f"\n  Examples where both judges agree on the change:")
    for r in same_change[:5]:
        print(f"    [{r['question_id']}]")
        print(f"      annotated={r['annotated_intent']}  →  both suggested={r['gpt_suggested']}")
        print(f"      Q: {r['question'][:100]}...")

if diff_change:
    print(f"\n  Examples where both judges changed but to different intents:")
    for r in diff_change[:5]:
        print(f"    [{r['question_id']}]")
        print(f"      annotated={r['annotated_intent']}  GPT→{r['gpt_suggested']}  Claude→{r['claude_suggested']}")
        print(f"      Q: {r['question'][:100]}...")

# ---------- 3. Additional breakdown ----------
print(f"\n{'=' * 60}")
print("Additional breakdown:")
print("=" * 60)
only_gpt_changed = sum(1 for r in aligned if not r["gpt_agrees_annotator"] and r["claude_agrees_annotator"])
only_claude_changed = sum(1 for r in aligned if r["gpt_agrees_annotator"] and not r["claude_agrees_annotator"])
neither_changed = sum(1 for r in aligned if r["gpt_agrees_annotator"] and r["claude_agrees_annotator"])
print(f"  Both agree with annotation: {neither_changed}")
print(f"  Only GPT-5.5 changed: {only_gpt_changed}")
print(f"  Only Claude changed: {only_claude_changed}")
print(f"  Both changed: {len(both_changed)}")

# ---------- 4. Confusion matrix ----------
all_labels = sorted(set(gpt_labels) | set(claude_labels))
cm = confusion_matrix(gpt_labels, claude_labels, labels=all_labels)
df_cm = pd.DataFrame(cm, index=all_labels, columns=all_labels)

print(f"\n{'=' * 60}")
print("Confusion matrix (rows = GPT-5.5, columns = Claude Opus 4.7)")
print("=" * 60)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(df_cm.to_string())

# Highlight off-diagonal disagreements
print(f"\nDiagonal (agreement) total: {sum(cm[i][i] for i in range(len(all_labels)))}")
print(f"Off-diagonal (disagreement) total: {cm.sum() - sum(cm[i][i] for i in range(len(all_labels)))}")

# Top disagreement pairs
print(f"\nTop 10 disagreement cells (GPT_label → Claude_label : count):")
pairs = []
for i, l1 in enumerate(all_labels):
    for j, l2 in enumerate(all_labels):
        if i != j and cm[i][j] > 0:
            pairs.append((l1, l2, cm[i][j]))
pairs.sort(key=lambda x: -x[2])
for l1, l2, count in pairs[:10]:
    print(f"  {l1:15s} → {l2:15s} : {count}")

# ---------- 5. Save confusion matrix as heatmap image ----------
fig, ax = plt.subplots(figsize=(14, 11))

# Use log1p scale for color so small off-diagonal cells remain visible
sns.heatmap(
    df_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=0.5,
    linecolor="grey",
    square=True,
    cbar_kws={"label": "Count"},
    ax=ax,
)

ax.set_xlabel("Claude Opus 4.7  (suggested intent)", fontsize=16, labelpad=10)
ax.set_ylabel("GPT-5.5  (suggested intent)", fontsize=16, labelpad=10)
ax.set_title(
    f"Cohen's κ = {kappa:.4f}  |  Raw agreement = {raw_agree}/{len(aligned)} ({100*raw_agree/len(aligned):.1f}%)",
    fontsize=16,
    pad=15,
)
ax.tick_params(axis="x", labelsize=14, rotation=45)
ax.tick_params(axis="y", labelsize=14, rotation=0)

plt.tight_layout()
output_path = OUTPUT_DIR / "confusion_matrix_gpt55_vs_claude_opus47.png"
plt.savefig(output_path, dpi=200)
print(f"\nConfusion matrix heatmap saved → {output_path}")
plt.close()
