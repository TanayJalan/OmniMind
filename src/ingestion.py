# ─────────────────────────────────────────────────────────────────
#  ingestion.py — Load documents from PDFs or URLs, then chunk them
#
#  WHY CHUNKING?
#  LLMs have a limited context window (how much text they can read
#  at once). We split large documents into small overlapping chunks
#  so we can find and send only the relevant pieces to the LLM.
# ─────────────────────────────────────────────────────────────────

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import os


def get_text_splitter():
    """
    RecursiveCharacterTextSplitter splits text in this order:
      paragraphs → sentences → words → characters
    It stops as soon as chunks are small enough.

    chunk_size=800   → each chunk holds ~800 characters
    chunk_overlap=100 → chunks share 100 characters with their
                        neighbour so context isn't lost at edges
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )


def load_pdf(uploaded_file) -> list:
    """
    Load a PDF uploaded through Streamlit and split into chunks.

    Streamlit gives us a file-like object, but PyPDFLoader needs
    a real file path — so we save it to a temp file first.

    Returns: list of LangChain Document objects (chunk + metadata)
    """
    # Save the uploaded file to a temporary location on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()              # one Document per page
        splitter = get_text_splitter()
        chunks = splitter.split_documents(pages)

        # Tag each chunk with the original filename so we can cite it later
        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name

        print(f"[ingestion] PDF '{uploaded_file.name}' → {len(pages)} pages → {len(chunks)} chunks")
        return chunks

    finally:
        os.unlink(tmp_path)                # clean up the temp file


def load_url(url: str) -> list:
    """
    Fetch a webpage and split into chunks.

    WebBaseLoader uses requests + BeautifulSoup under the hood
    to strip HTML tags and extract clean text.

    Returns: list of LangChain Document objects
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = get_text_splitter()
    chunks = splitter.split_documents(docs)

    print(f"[ingestion] URL '{url}' → {len(docs)} page(s) → {len(chunks)} chunks")
    return chunks
