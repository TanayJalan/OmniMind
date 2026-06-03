# ─────────────────────────────────────────────────────────────────
#  validate_dataset.py — Check dataset quality before fine-tuning
#
#  Run this after clean_and_format.py to verify your dataset
#  is ready for fine-tuning. It prints a full quality report.
#
#  Run: python data_pipeline/validate_dataset.py
# ─────────────────────────────────────────────────────────────────

import json
from collections import Counter
import os


def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def print_section(title: str):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


def validate_split(records: list, split_name: str):
    """Run quality checks on one dataset split and print a report."""
    print_section(f"{split_name.upper()} SPLIT  ({len(records)} records)")

    if not records:
        print("  ⚠ Empty split!")
        return

    # ── Check required fields ────────────────────────────────────
    missing_text  = sum(1 for r in records if not r.get("text"))
    missing_q     = sum(1 for r in records if not r.get("question"))
    missing_a     = sum(1 for r in records if not r.get("answer"))
    print(f"\n  Field completeness:")
    print(f"    text field     : {'✔' if missing_text == 0 else f'✘ {missing_text} missing'}")
    print(f"    question field : {'✔' if missing_q    == 0 else f'✘ {missing_q} missing'}")
    print(f"    answer field   : {'✔' if missing_a    == 0 else f'✘ {missing_a} missing'}")

    # ── Length statistics ────────────────────────────────────────
    q_lengths = [len(r["question"]) for r in records if r.get("question")]
    a_lengths = [len(r["answer"])   for r in records if r.get("answer")]
    t_lengths = [len(r["text"])     for r in records if r.get("text")]

    def stats(values):
        if not values: return "N/A"
        return f"min={min(values)}  avg={sum(values)//len(values)}  max={max(values)}"

    print(f"\n  Length stats (chars):")
    print(f"    Questions : {stats(q_lengths)}")
    print(f"    Answers   : {stats(a_lengths)}")
    print(f"    Full text : {stats(t_lengths)}")

    # ── Mistral format check ─────────────────────────────────────
    correct_format = sum(
        1 for r in records
        if r.get("text", "").startswith("<s>[INST]") and r.get("text", "").endswith("</s>")
    )
    print(f"\n  Mistral format : {correct_format}/{len(records)} correctly formatted")

    # ── Topic distribution ───────────────────────────────────────
    topics = Counter(r.get("topic", "unknown") for r in records)
    print(f"\n  Topics covered : {len(topics)}")
    print(f"  Most common    :")
    for topic, count in topics.most_common(5):
        bar = "█" * count
        print(f"    {count:3}× {topic[:45]}")

    # ── Duplicate check ──────────────────────────────────────────
    questions = [r.get("question", "").lower() for r in records]
    duplicates = len(questions) - len(set(questions))
    print(f"\n  Duplicates : {duplicates} {'✔ none' if duplicates == 0 else '⚠ found!'}")


def main():
    print("\n" + "═"*50)
    print("   OmniMind Phase 3 — Dataset Validation Report")
    print("═"*50)

    splits = {
        "train": "data/train.jsonl",
        "val":   "data/val.jsonl",
        "test":  "data/test.jsonl",
    }

    total = 0
    for split_name, path in splits.items():
        if not os.path.exists(path):
            print(f"\n  ⚠ {path} not found — run clean_and_format.py first")
            continue
        records = load_jsonl(path)
        total += len(records)
        validate_split(records, split_name)

    print(f"\n{'═'*50}")
    print(f"  TOTAL: {total} records across all splits")

    # ── Final recommendation ─────────────────────────────────────
    if total < 100:
        print("  ⚠  Less than 100 samples — consider generating more data")
    elif total < 500:
        print("  ⚠  Dataset is small — fine-tuning will work but more data = better results")
    else:
        print("  ✔  Dataset size looks good for fine-tuning!")

    print("═"*50 + "\n")


if __name__ == "__main__":
    main()
