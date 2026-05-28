# ─────────────────────────────────────────────────────────────────
#  app.py — OmniMind Streamlit UI
#
#  Run with:  streamlit run app.py
#
#  Streamlit re-runs this entire script top-to-bottom every time
#  the user interacts with the app. We use st.session_state to
#  keep data (chat history, the RAG chain) between re-runs.
# ─────────────────────────────────────────────────────────────────

import streamlit as st
from src.ingestion import load_pdf, load_url
from src.vectorstore import add_documents

from src.rag_chain import get_rag_chain, ask


# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniMind",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 OmniMind")
st.caption("Upload PDFs or paste a URL — then ask anything about them.")


# ── Session state init ───────────────────────────────────────────
# session_state persists across Streamlit re-runs (like a global dict)
if "messages" not in st.session_state:
    st.session_state.messages = []      # chat history for display

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None   # built once after docs are loaded

if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False


# ── Sidebar: document loading ─────────────────────────────────────
with st.sidebar:
    st.header("📂 Load Documents")

    # ── PDF upload ────────────────────────────────────────────────
    st.subheader("Upload PDF(s)")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # ── URL input ─────────────────────────────────────────────────
    st.subheader("Or paste a URL")
    url_input = st.text_input("Website URL", placeholder="https://example.com")

    # ── Process button ────────────────────────────────────────────
    if st.button("⚡ Process Documents", type="primary", use_container_width=True):
        all_chunks = []

        # Load each uploaded PDF
        if uploaded_files:
            for f in uploaded_files:
                with st.spinner(f"Reading {f.name}..."):
                    chunks = load_pdf(f)
                    all_chunks.extend(chunks)
                    st.success(f"✔ {f.name} — {len(chunks)} chunks")

        # Load URL if provided
        if url_input.strip():
            with st.spinner(f"Fetching {url_input}..."):
                try:
                    chunks = load_url(url_input.strip())
                    all_chunks.extend(chunks)
                    st.success(f"✔ URL loaded — {len(chunks)} chunks")
                except Exception as e:
                    st.error(f"Could not load URL: {e}")

        if all_chunks:
            with st.spinner("Building vector store..."):
                total = add_documents(all_chunks)
            st.info(f"Vector store ready — {total} total chunks indexed")

            # Build the RAG chain now that documents are loaded
            with st.spinner("Initialising RAG chain..."):
                st.session_state.rag_chain = get_rag_chain()
                st.session_state.docs_loaded = True

            st.success("✅ Ready! Ask a question below.")
        else:
            st.warning("Please upload a PDF or enter a URL first.")

    # ── Reset button ──────────────────────────────────────────────
    if st.session_state.docs_loaded:
        st.divider()
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.rag_chain = get_rag_chain()
            st.rerun()


# ── Main area: chat ───────────────────────────────────────────────

# Show a prompt if no docs loaded yet
if not st.session_state.docs_loaded:
    st.info("👈 Load a PDF or URL in the sidebar to get started.")
    st.stop()   # don't render the chat input below

# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show source citations for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources used"):
                for i, doc in enumerate(msg["sources"], 1):
                    source_name = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "")
                    label = f"**Source {i}** — {source_name}"
                    if page != "":
                        label += f", page {int(page) + 1}"
                    st.markdown(label)
                    st.caption(doc.page_content[:300] + "...")

# Chat input box at the bottom
if question := st.chat_input("Ask a question about your documents..."):

    # Show the user's message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer from the RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(st.session_state.rag_chain, question)

        answer = result["answer"]
        sources = result["sources"]

        st.markdown(answer)

        # Show source citations
        if sources:
            with st.expander("📄 Sources used"):
                for i, doc in enumerate(sources, 1):
                    source_name = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "")
                    label = f"**Source {i}** — {source_name}"
                    if page != "":
                        label += f", page {int(page) + 1}"
                    st.markdown(label)
                    st.caption(doc.page_content[:300] + "...")

    # Save assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
