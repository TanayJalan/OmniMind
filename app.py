# ─────────────────────────────────────────────────────────────────
#  app.py — OmniMind Streamlit UI  (Phase 1 + Phase 2)
#
#  Phase 1: RAG chatbot (PDF / URL Q&A)
#  Phase 2: Vision tab (OCR, Object Detection, Visual Q&A)
#
#  Run with:  streamlit run app.py
# ─────────────────────────────────────────────────────────────────

import streamlit as st
from PIL import Image

# Phase 1 imports
from src.ingestion import load_pdf, load_url
from src.vectorstore import add_documents
from src.rag_chain import get_rag_chain, ask

# Phase 2 imports
from src.vision.ocr import extract_text
from src.vision.detection import detect_objects
from src.vision.visual_qa import ask_about_image, describe_image


# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniMind",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 OmniMind")
st.caption("Phase 1: Document Q&A  ·  Phase 2: Computer Vision")


# ── Session state ────────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "rag_chain"   not in st.session_state: st.session_state.rag_chain   = None
if "docs_loaded" not in st.session_state: st.session_state.docs_loaded = False


# ════════════════════════════════════════════════════════════════
#  TAB LAYOUT
# ════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["💬 Document Chat", "🖼️ Vision"])


# ════════════════════════════════════════════════════════════════
#  TAB 1 — RAG Chatbot (Phase 1, unchanged)
# ════════════════════════════════════════════════════════════════
with tab1:
    with st.sidebar:
        st.header("📂 Load Documents")

        uploaded_files = st.file_uploader(
            "Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        url_input = st.text_input("Or paste a URL", placeholder="https://example.com")

        if st.button("⚡ Process Documents", type="primary", use_container_width=True):
            all_chunks = []
            if uploaded_files:
                for f in uploaded_files:
                    with st.spinner(f"Reading {f.name}..."):
                        chunks = load_pdf(f)
                        all_chunks.extend(chunks)
                        st.success(f"✔ {f.name} — {len(chunks)} chunks")

            if url_input.strip():
                with st.spinner(f"Fetching URL..."):
                    try:
                        chunks = load_url(url_input.strip())
                        all_chunks.extend(chunks)
                        st.success(f"✔ URL — {len(chunks)} chunks")
                    except Exception as e:
                        st.error(f"Could not load URL: {e}")

            if all_chunks:
                with st.spinner("Building vector store..."):
                    total = add_documents(all_chunks)
                with st.spinner("Initialising RAG chain..."):
                    st.session_state.rag_chain   = get_rag_chain()
                    st.session_state.docs_loaded = True
                st.success(f"✅ Ready — {total} chunks indexed")
            else:
                st.warning("Upload a PDF or enter a URL first.")

        if st.session_state.docs_loaded:
            st.divider()
            if st.button("🔄 Reset Chat", use_container_width=True):
                st.session_state.messages  = []
                st.session_state.rag_chain = get_rag_chain()
                st.rerun()

    if not st.session_state.docs_loaded:
        st.info("👈 Load a PDF or URL in the sidebar to get started.")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("📄 Sources used"):
                        for i, doc in enumerate(msg["sources"], 1):
                            source = doc.metadata.get("source", "Unknown")
                            page   = doc.metadata.get("page", "")
                            label  = f"**Source {i}** — {source}"
                            if page != "": label += f", page {int(page)+1}"
                            st.markdown(label)
                            st.caption(doc.page_content[:300] + "...")

        if question := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result  = ask(st.session_state.rag_chain, question)
                answer  = result["answer"]
                sources = result["sources"]
                st.markdown(answer)
                if sources:
                    with st.expander("📄 Sources used"):
                        for i, doc in enumerate(sources, 1):
                            source = doc.metadata.get("source", "Unknown")
                            page   = doc.metadata.get("page", "")
                            label  = f"**Source {i}** — {source}"
                            if page != "": label += f", page {int(page)+1}"
                            st.markdown(label)
                            st.caption(doc.page_content[:300] + "...")

            st.session_state.messages.append({
                "role": "assistant", "content": answer, "sources": sources,
            })


# ════════════════════════════════════════════════════════════════
#  TAB 2 — Vision (Phase 2)
# ════════════════════════════════════════════════════════════════
with tab2:
    st.header("🖼️ Vision Analysis")
    st.caption("Upload an image to extract text, detect objects, or ask questions about it.")

    # ── Image upload ─────────────────────────────────────────────
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="vision_upload",
    )

    if uploaded_image:
        # Load image once and reuse across all features
        image = Image.open(uploaded_image).convert("RGB")

        # Show original image
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with col_info:
            w, h = image.size
            st.metric("Width",  f"{w}px")
            st.metric("Height", f"{h}px")
            st.metric("Mode",   image.mode)

        st.divider()

        # ── Three feature sections ────────────────────────────────
        feat1, feat2, feat3 = st.tabs([
            "📝 Extract Text (OCR)",
            "🔍 Detect Objects",
            "💬 Visual Q&A",
        ])

        # ── Feature 1: OCR ────────────────────────────────────────
        with feat1:
            st.subheader("Extract Text from Image")
            st.caption("Uses Tesseract OCR — works best on screenshots, scanned docs, signs.")

            if st.button("📝 Extract Text", type="primary"):
                with st.spinner("Running OCR..."):
                    result = extract_text(image)

                if result["word_count"] > 0:
                    st.success(f"Found {result['word_count']} words  ·  Confidence: {result['confidence']}%")
                    st.text_area("Extracted Text", result["text"], height=200)

                    # Bonus: add extracted text to vector store
                    st.divider()
                    if st.button("➕ Add this text to Document Chat"):
                        from langchain.schema import Document
                        from src.vectorstore import add_documents
                        from src.rag_chain import get_rag_chain

                        doc = Document(
                            page_content=result["text"],
                            metadata={"source": uploaded_image.name, "type": "ocr"},
                        )
                        add_documents([doc])
                        st.session_state.rag_chain   = get_rag_chain()
                        st.session_state.docs_loaded = True
                        st.success("Text added! Switch to 💬 Document Chat to ask questions about it.")
                else:
                    st.warning("No text found. Try a clearer image with higher contrast.")

        # ── Feature 2: Object Detection ───────────────────────────
        with feat2:
            st.subheader("Detect Objects")
            st.caption("Uses YOLOv8 nano — detects 80 common object classes.")

            if st.button("🔍 Detect Objects", type="primary"):
                with st.spinner("Running YOLOv8... (first run downloads model ~6MB)"):
                    detections, annotated_image, summary = detect_objects(image)

                if detections:
                    st.success(f"Detected: {summary}")

                    col_orig, col_ann = st.columns(2)
                    with col_orig:
                        st.image(image,            caption="Original",   use_container_width=True)
                    with col_ann:
                        st.image(annotated_image,  caption="Annotated",  use_container_width=True)

                    # Show detections as a table
                    st.subheader("Detections")
                    table_data = [
                        {"Object": d["label"], "Confidence": f"{d['confidence']:.0%}"}
                        for d in sorted(detections, key=lambda x: x["confidence"], reverse=True)
                    ]
                    st.table(table_data)
                else:
                    st.warning("No objects detected. Try a different image.")

        # ── Feature 3: Visual Q&A ─────────────────────────────────
        with feat3:
            st.subheader("Ask Questions About the Image")
            st.caption("Uses Google Gemini Flash — ask anything about what's in the image.")

            # Auto-describe button
            if st.button("🔎 Auto-describe this image"):
                with st.spinner("Gemini is analysing the image..."):
                    description = describe_image(image)
                st.info(description)

            st.divider()

            # Custom question input
            vqa_question = st.text_input(
                "Or ask your own question",
                placeholder="What colour is the car? How many people are there? What's written on the sign?",
            )
            if st.button("💬 Ask", type="primary") and vqa_question:
                with st.spinner("Gemini is thinking..."):
                    result = ask_about_image(image, vqa_question)
                st.markdown(f"**Q:** {result['question']}")
                st.markdown(f"**A:** {result['answer']}")

    else:
        st.info("⬆️ Upload an image above to get started.")
