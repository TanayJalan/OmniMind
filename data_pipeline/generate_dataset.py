# ─────────────────────────────────────────────────────────────────
#  generate_dataset.py — Generate AI/ML Q&A pairs using Groq
#
#  WHY SYNTHETIC DATA?
#    Fine-tuning needs thousands of (question, answer) pairs in
#    your target domain. Collecting real data takes months.
#    We use an LLM (Groq) to generate high-quality synthetic
#    Q&A pairs across 40+ AI/ML topics — this takes ~15 minutes
#    and costs nothing with the free Groq tier.
#
#  OUTPUT: data/raw_dataset.jsonl
#    Each line is a JSON object: {"question": "...", "answer": "..."}
#
#  Run: python data_pipeline/generate_dataset.py
# ─────────────────────────────────────────────────────────────────

import os
import json
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── 40+ AI/ML topics to generate Q&A for ────────────────────────
TOPICS = [
    # Fundamentals
    "gradient descent and learning rate",
    "backpropagation in neural networks",
    "overfitting and regularisation techniques",
    "bias-variance tradeoff",
    "train/validation/test split",
    "batch normalisation",
    "dropout regularisation",
    "activation functions (ReLU, sigmoid, tanh)",
    "loss functions in deep learning",
    "optimisers (Adam, SGD, RMSprop)",

    # Architectures
    "convolutional neural networks (CNNs)",
    "recurrent neural networks and LSTMs",
    "transformer architecture",
    "self-attention mechanism",
    "BERT and masked language modelling",
    "GPT and autoregressive language modelling",
    "encoder-decoder architecture",
    "residual connections and skip connections",
    "positional encoding in transformers",
    "multi-head attention",

    # LLMs & Modern AI
    "large language models (LLMs)",
    "retrieval augmented generation (RAG)",
    "fine-tuning vs pre-training",
    "instruction tuning and RLHF",
    "LoRA and parameter-efficient fine-tuning",
    "quantisation (INT8, INT4, QLoRA)",
    "prompt engineering techniques",
    "chain-of-thought prompting",
    "vector embeddings and similarity search",
    "tokenisation in NLP",

    # ML Engineering
    "MLOps and model deployment",
    "model evaluation metrics",
    "cross-validation techniques",
    "feature engineering",
    "transfer learning",
    "data augmentation strategies",
    "class imbalance handling",
    "hyperparameter tuning",
    "model interpretability and SHAP",
    "A/B testing for ML models",

    # Practical Tools
    "HuggingFace Transformers library",
    "LangChain for LLM applications",
    "vector databases (ChromaDB, Pinecone, Weaviate)",
    "PyTorch vs TensorFlow",
    "experiment tracking with MLflow or W&B",
]

PAIRS_PER_TOPIC = 5   # 5 Q&A pairs × 44 topics = ~220 total pairs

SYSTEM_PROMPT = """You are an expert AI/ML educator creating a high-quality Q&A dataset 
for fine-tuning a language model. Generate clear, accurate, and educational Q&A pairs."""

def generate_qa_for_topic(topic: str) -> list:
    """
    Ask Groq to generate Q&A pairs for one topic.
    We ask for JSON output so we can parse it reliably.
    """
    user_prompt = f"""Generate {PAIRS_PER_TOPIC} question-answer pairs about: {topic}

Rules:
- Questions should range from beginner to intermediate level
- Answers should be thorough but concise (2-5 sentences)
- Cover different angles of the topic
- Be technically accurate

Respond ONLY with a valid JSON array, no other text:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.7,     # some creativity for diverse questions
            max_tokens=2048,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if Groq adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        pairs = json.loads(raw)
        return pairs

    except json.JSONDecodeError as e:
        print(f"  [!] JSON parse error for '{topic}': {e}")
        return []
    except Exception as e:
        print(f"  [!] API error for '{topic}': {e}")
        return []


def main():
    os.makedirs("data", exist_ok=True)
    output_path = "data/raw_dataset.jsonl"

    all_pairs = []
    print(f"\n Generating AI/ML Q&A dataset")
    print(f" Topics: {len(TOPICS)}  ·  Pairs per topic: {PAIRS_PER_TOPIC}")
    print(f" Target: ~{len(TOPICS) * PAIRS_PER_TOPIC} Q&A pairs\n")

    for i, topic in enumerate(TOPICS, 1):
        print(f"[{i:2}/{len(TOPICS)}] {topic}...", end=" ", flush=True)

        pairs = generate_qa_for_topic(topic)

        # Tag each pair with its topic (useful for filtering later)
        for pair in pairs:
            pair["topic"] = topic
            all_pairs.append(pair)

        print(f"✔ {len(pairs)} pairs")

        # Respect Groq rate limits — 30 requests/min on free tier
        time.sleep(2)

    # Save as JSONL (one JSON object per line)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n Done! {len(all_pairs)} Q&A pairs saved to {output_path}")


if __name__ == "__main__":
    main()
