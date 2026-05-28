# 🧠 OmniMind — Phase 1

A RAG chatbot that answers questions from your PDFs and websites. Powered by Groq (free) + ChromaDB + sentence-transformers.

## Setup

```bash
# 1. Clone / enter the project
cd omnimind

# 2. Activate your virtual environment
source omnimind-env/bin/activate

# 3. Copy env file and add your Groq key
cp .env.example .env
# Open .env and paste your key from https://console.groq.com

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

## Project structure

```
omnimind/
├── app.py                  ← Streamlit UI (start here)
├── requirements.txt
├── .env.example            ← copy to .env and add your key
├── src/
│   ├── ingestion.py        ← loads PDFs and URLs, splits into chunks
│   ├── vectorstore.py      ← embeds chunks and stores in ChromaDB
│   └── rag_chain.py        ← RAG pipeline + memory + Groq LLM
└── data/
    ├── uploads/            ← temp PDF storage
    └── chroma_db/          ← vector DB (auto-created)
```

## What you learned in Phase 1

- **RAG** — Retrieval Augmented Generation: find relevant text, then answer
- **Embeddings** — turning text into numbers that capture meaning
- **Vector search** — finding similar chunks using cosine similarity
- **LangChain** — chaining LLM calls, memory, and retrieval together
- **Streamlit** — building interactive ML apps with pure Python
