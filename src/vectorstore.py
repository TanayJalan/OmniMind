# ─────────────────────────────────────────────────────────────────
#  vectorstore.py — Store and search document chunks using vectors
#
#  HOW RAG WORKS (simplified):
#    1. We turn every text chunk into a vector (list of numbers)
#       that captures its *meaning*
#    2. When the user asks a question, we turn the question into
#       a vector too
#    3. We find the chunks whose vectors are closest to the
#       question vector — those are the most relevant chunks
#    4. We send those chunks + the question to the LLM
#
#  WHY sentence-transformers?
#    It runs 100% locally on your Mac. Free, fast, no API needed.
#    Model: all-MiniLM-L6-v2 — small (80MB) but very capable.
# ─────────────────────────────────────────────────────────────────

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# Persist the vector DB to disk so it survives app restarts
CHROMA_DB_PATH = "./data/chroma_db"

# Embedding model — downloaded once, cached locally after that
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    """
    Load the sentence-transformers embedding model.
    First run: downloads ~80MB from HuggingFace (one time only).
    After that: loads instantly from local cache.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # use "mps" on Apple Silicon for speed boost
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore():
    """
    Open (or create) the persistent ChromaDB vector store.
    ChromaDB stores the vectors as files under data/chroma_db/
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
    )
    return vectorstore


def add_documents(chunks: list) -> int:
    """
    Embed a list of chunks and add them to ChromaDB.

    LangChain handles the embedding step automatically:
    it calls embeddings.embed_documents() on each chunk's text
    and stores (vector, text, metadata) in ChromaDB.

    Returns the total number of documents now in the store.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )
    count = vectorstore._collection.count()
    print(f"[vectorstore] Added {len(chunks)} chunks. Total in DB: {count}")
    return count


def get_retriever(k: int = 4):
    """
    Returns a retriever that fetches the top-k most relevant chunks.

    k=4 means: for each user question, retrieve the 4 chunks
    whose embedding is closest to the question's embedding.
    """
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
