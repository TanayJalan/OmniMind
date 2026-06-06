---
title: OmniMind
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🧠 OmniMind
### AI Research & Analysis Platform

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/TanayJalan/omnimind)
[![Model](https://img.shields.io/badge/🤗%20Model-Mistral%207B%20Fine--tuned-orange)](https://huggingface.co/TanayJalan/omnimind-mistral-7b-mlqa)
[![Python](https://img.shields.io/badge/Python-3.12-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

A production-ready, multi-modal AI platform combining document intelligence, computer vision, and a custom fine-tuned LLM — deployed with full MLOps infrastructure.

</div>

---

## Overview

OmniMind is a multi-modal AI research and analysis platform that enables users to extract insights from documents and images using state-of-the-art AI models. Built with a modular architecture, it supports PDF ingestion, web crawling, OCR, object detection, visual Q&A, and conversational document chat — all powered by free, open-source models.

---

## Features

### 💬 Document Chat (RAG)
- Upload PDFs or paste any URL for instant ingestion
- Semantic search across documents using vector embeddings
- Multi-turn conversational memory
- Source citations with page-level references
- Powered by LangChain + ChromaDB + Groq LLaMA 3.1 (free)

### 🖼️ Computer Vision
- **OCR** — Extract text from any image using Tesseract
- **Object Detection** — Detect and label objects with bounding boxes (YOLOv8)
- **Visual Q&A** — Ask natural language questions about any image (Gemini Flash)
- Push OCR results directly into the document chat vector store

### 🔧 Fine-tuned Model
- Custom Mistral 7B fine-tuned on AI/ML Q&A dataset
- Trained with QLoRA (4-bit quantisation + LoRA adapters) on Google Colab T4
- Published on HuggingFace Hub: [TanayJalan/omnimind-mistral-7b-mlqa](https://huggingface.co/TanayJalan/omnimind-mistral-7b-mlqa)

### ⚙️ MLOps
- FastAPI REST backend with auto-generated Swagger docs
- MLflow experiment tracking and model registry
- Dockerised deployment with docker-compose
- GitHub Actions CI/CD — auto-deploys to HuggingFace Spaces on every push

---

## Architecture

```
                        ┌─────────────────────────────┐
                        │       Streamlit UI           │
                        │   (Document Chat + Vision)   │
                        └──────────────┬──────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
    ┌─────────▼────────┐   ┌──────────▼─────────┐   ┌─────────▼────────┐
    │   RAG Pipeline   │   │  Vision Pipeline   │   │   FastAPI REST   │
    │                  │   │                    │   │     Backend      │
    │ ingestion.py     │   │ ocr.py             │   │ /api/chat        │
    │ vectorstore.py   │   │ detection.py       │   │ /api/vision/*    │
    │ rag_chain.py     │   │ visual_qa.py       │   │ /api/health      │
    └────────┬─────────┘   └──────────┬─────────┘   └─────────┬────────┘
             │                        │                        │
    ┌────────▼────┐        ┌──────────▼────┐        ┌─────────▼────────┐
    │  ChromaDB   │        │  YOLOv8 nano  │        │     MLflow       │
    │  + Groq LLM │        │  + Tesseract  │        │  Experiment      │
    │  + Embedder │        │  + Gemini API │        │  Tracking        │
    └─────────────┘        └───────────────┘        └──────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq (LLaMA 3.1 8B) — free tier |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) — local |
| **Vector DB** | ChromaDB — local persistence |
| **RAG** | LangChain · langchain-community · langchain-groq |
| **OCR** | Tesseract · pytesseract |
| **Object Detection** | Ultralytics YOLOv8 nano |
| **Vision LLM** | Google Gemini 1.5 Flash — free tier |
| **Fine-tuning** | HuggingFace Transformers · PEFT · TRL · bitsandbytes |
| **Backend API** | FastAPI · Pydantic · uvicorn |
| **Experiment Tracking** | MLflow |
| **Containerisation** | Docker · docker-compose |
| **CI/CD** | GitHub Actions → HuggingFace Spaces |
| **Frontend** | Streamlit |

---

## Project Structure

```
OmniMind/
├── api/
│   ├── main.py                     ← FastAPI app (5 endpoints)
│   └── schemas.py                  ← Pydantic request/response models
├── src/
│   ├── ingestion.py                ← PDF & URL loader with chunking
│   ├── vectorstore.py              ← ChromaDB + HuggingFace embeddings
│   ├── rag_chain.py                ← RAG pipeline + conversational memory
│   └── vision/
│       ├── ocr.py                  ← Tesseract OCR pipeline
│       ├── detection.py            ← YOLOv8 object detection
│       └── visual_qa.py            ← Gemini Flash visual Q&A
├── data_pipeline/
│   ├── generate_dataset.py         ← Synthetic Q&A generation via Groq
│   ├── clean_and_format.py         ← Cleaning + Mistral instruction format
│   └── validate_dataset.py         ← Dataset quality validation
├── mlflow_tracking/
│   └── tracker.py                  ← MLflow experiment wrapper
├── .github/workflows/
│   └── deploy.yml                  ← CI/CD to HuggingFace Spaces
├── app.py                          ← Streamlit UI entry point
├── Dockerfile
├── docker-compose.yml
├── OmniMind_Phase3_Finetune.ipynb  ← Colab fine-tuning notebook
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Homebrew (macOS) or apt (Linux)
- Free API keys: [Groq](https://console.groq.com) · [Google AI Studio](https://aistudio.google.com/apikey)

### Installation

```bash
# Clone
git clone https://github.com/TanayJalan/OmniMind.git
cd OmniMind

# Install Tesseract OCR engine
brew install tesseract          # macOS
# sudo apt install tesseract-ocr  # Ubuntu/Debian

# Create virtual environment
python3 -m venv omnimind-env
source omnimind-env/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...
```

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`

---

## API

Start the FastAPI backend:

```bash
uvicorn api.main:app --reload --port 8000
```

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | RAG chatbot — ask questions about documents |
| `POST` | `/api/vision/ocr` | Extract text from an image |
| `POST` | `/api/vision/detect` | Detect objects in an image |
| `POST` | `/api/vision/qa` | Visual Q&A on an image |
| `GET` | `/api/health` | Health check |

Interactive docs: `http://localhost:8000/docs`

---

## Docker

```bash
# Start full stack (app + MLflow)
docker-compose up --build

# App:     http://localhost:8501
# MLflow:  http://localhost:5001
```

---

## MLflow Experiment Tracking

```bash
python mlflow_tracking/tracker.py
mlflow ui --port 5001
```

Tracks RAG parameters (chunk size, retrieval k, model) and metrics (response time, answer length) across runs.

---

## Fine-tuning (Phase 3)

Generate and prepare the dataset locally:

```bash
python data_pipeline/generate_dataset.py   # generates ~220 Q&A pairs
python data_pipeline/clean_and_format.py   # formats as Mistral instruction template
python data_pipeline/validate_dataset.py   # quality report
```

Then open `OmniMind_Phase3_Finetune.ipynb` in [Google Colab](https://colab.research.google.com) with a T4 GPU runtime and follow the notebook cells.

---

## Deployment

Deployment is fully automated via GitHub Actions.

Every push to `main` triggers `.github/workflows/deploy.yml` which uploads the app to HuggingFace Spaces using the `huggingface_hub` Python API.

**Required GitHub secrets:**

| Secret | Description |
|---|---|
| `HF_TOKEN` | HuggingFace write token |
| `HF_USERNAME` | HuggingFace username |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | Free LLM API — [console.groq.com](https://console.groq.com) |
| `GOOGLE_API_KEY` | ✅ | Gemini Vision — [aistudio.google.com](https://aistudio.google.com/apikey) |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built by <a href="https://github.com/TanayJalan">Tanay Jalan</a>
</div>
