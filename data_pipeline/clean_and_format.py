# ─────────────────────────────────────────────────────────────────
#  clean_and_format.py — Clean and format dataset for Mistral
#
#  WHY THIS STEP MATTERS:
#    Raw generated data has noise: duplicate questions, very short
#    answers, formatting inconsistencies. Cleaning directly impacts
#    fine-tuning quality — garbage in, garbage out.
#
#  MISTRAL INSTRUCTION FORMAT:
#    Mistral expects this exact template for instruction tuning:
#    <s>[INST] {question} [/INST] {answer} </s>
#    The model learns to complete the pattern: given [INST]...[/INST]
#    it should produce what comes after (the answer).
#
#  OUTPUT:
#    data/train.jsonl  (80%)
#    data/val.jsonl    (10%)
#    data/test.jsonl   (10%)
#
#  Run: python data_pipeline/clean_and_format.py
# ─────────────────────────────────────────────────────────────────

import json
import re
import random
from collections import Counter

# ── Quality thresholds ───────────────────────────────────────────
MIN_QUESTION_LEN = 15    # characters — filters out "What is AI?"
MIN_ANSWER_LEN   = 80    # characters — filters out one-liners
MAX_ANSWER_LEN   = 2000  # characters — filters out walls of text

# ── Train/val/test split ─────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 0.10  (remainder)

random.seed(42)   # reproducible splits every run


def clean_text(text: str) -> str:
    """
    Normalise a text string:
      - Strip leading/trailing whitespace
      - Collapse multiple spaces/newlines into one
      - Remove zero-width characters
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)            # collapse whitespace
    text = re.sub(r"[\u200b\u200c\u200d]", "", text)  # zero-width chars
    return text


def is_quality_pair(pair: dict) -> tuple:
    """
    Check if a Q&A pair meets quality standards.
    Returns (is_valid: bool, reason: str)
    """
    q = pair.get("question", "").strip()
    a = pair.get("answer",   "").strip()

    if len(q) < MIN_QUESTION_LEN:
        return False, f"question too short ({len(q)} chars)"
    if len(a) < MIN_ANSWER_LEN:
        return False, f"answer too short ({len(a)} chars)"
    if len(a) > MAX_ANSWER_LEN:
        return False, f"answer too long ({len(a)} chars)"
    if not q.endswith("?") and not any(q.lower().startswith(w) for w in
                                        ["what", "how", "why", "when", "explain",
                                         "describe", "define", "compare"]):
        return False, "question doesn't look like a question"
    return True, "ok"


def format_for_mistral(question: str, answer: str) -> str:
    """
    Format a Q&A pair as Mistral's instruction template.

    This is the EXACT format Mistral was pre-trained with, so using
    it for fine-tuning helps the model generalise better.

    Template: <s>[INST] question [/INST] answer </s>
    """
    return f"<s>[INST] {question} [/INST] {answer} </s>"


def load_jsonl(path: str) -> list:
    """Load all records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list, path: str):
    """Save records to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records → {path}")


def main():
    raw_path = "data/raw_dataset.jsonl"

    print("\n Cleaning and formatting dataset\n")

    # ── Load raw data ────────────────────────────────────────────
    raw = load_jsonl(raw_path)
    print(f" Loaded {len(raw)} raw pairs from {raw_path}")

    # ── Clean & filter ───────────────────────────────────────────
    cleaned = []
    skipped = Counter()

    for pair in raw:
        pair["question"] = clean_text(pair.get("question", ""))
        pair["answer"]   = clean_text(pair.get("answer",   ""))

        valid, reason = is_quality_pair(pair)
        if valid:
            cleaned.append(pair)
        else:
            skipped[reason] += 1

    print(f" After cleaning: {len(cleaned)} pairs  ({len(raw)-len(cleaned)} removed)")
    if skipped:
        for reason, count in skipped.most_common():
            print(f"   - {reason}: {count}")

    # ── Deduplicate by question ──────────────────────────────────
    seen_questions = set()
    deduped = []
    for pair in cleaned:
        q_lower = pair["question"].lower()
        if q_lower not in seen_questions:
            seen_questions.add(q_lower)
            deduped.append(pair)

    print(f" After deduplication: {len(deduped)} pairs")

    # ── Format as Mistral instruction template ───────────────────
    formatted = []
    for pair in deduped:
        formatted.append({
            "text":     format_for_mistral(pair["question"], pair["answer"]),
            "question": pair["question"],
            "answer":   pair["answer"],
            "topic":    pair.get("topic", ""),
        })

    # ── Shuffle & split ──────────────────────────────────────────
    random.shuffle(formatted)
    n      = len(formatted)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train = formatted[:n_train]
    val   = formatted[n_train : n_train + n_val]
    test  = formatted[n_train + n_val:]

    print(f"\n Split: train={len(train)}  val={len(val)}  test={len(test)}")

    # ── Save splits ──────────────────────────────────────────────
    save_jsonl(train, "data/train.jsonl")
    save_jsonl(val,   "data/val.jsonl")
    save_jsonl(test,  "data/test.jsonl")

    # ── Dataset stats ────────────────────────────────────────────
    if not formatted:
        print("\n ⚠ No records after cleaning.")
        print("  → Did you run generate_dataset.py first?")
        print("  → Run: python data_pipeline/generate_dataset.py")
        return

    avg_q = sum(len(p["question"]) for p in formatted) / len(formatted)
    avg_a = sum(len(p["answer"])   for p in formatted) / len(formatted)
    print(f"\n Stats:")
    print(f"  Avg question length : {avg_q:.0f} chars")
    print(f"  Avg answer length   : {avg_a:.0f} chars")
    print(f"  Topics covered      : {len(set(p['topic'] for p in formatted))}")
    print(f"\n Dataset ready for fine-tuning!")


if __name__ == "__main__":
    main()