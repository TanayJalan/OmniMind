# ─────────────────────────────────────────────────────────────────
#  rag_chain.py — The brain of OmniMind
#
#  This file wires together:
#    Retriever  →  finds relevant chunks from ChromaDB
#    Memory     →  remembers what was said earlier in the chat
#    LLM        →  Groq (free) running LLaMA 3 or Mixtral
#    Prompt     →  instructs the LLM how to answer
#
#  The flow for each user message:
#    1. Retrieve the top-4 relevant chunks from ChromaDB
#    2. Build a prompt: context chunks + chat history + question
#    3. Send to Groq → get answer
#    4. Save question + answer to memory for next turn
#
#  Updated for LangChain v1.x (LCEL-based approach)
# ─────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.vectorstore import get_retriever

load_dotenv()   # reads GROQ_API_KEY from your .env file


# ── Prompt template ──────────────────────────────────────────────
# {context}  → the relevant chunks retrieved from ChromaDB
# {chat_history} → previous turns in this conversation
# {question} → the user's latest message
SYSTEM_TEMPLATE = """You are OmniMind, a helpful AI assistant that answers questions
based on the provided documents. Be concise and accurate.

If the answer is not in the context below, say "I couldn't find that in the
uploaded documents" — do not make things up.

Context from documents:
{context}"""


def get_llm():
    """
    Groq gives you free access to:
      - llama-3.1-8b-instant  (fast, great for most tasks)
      - llama-3.3-70b-versatile (smarter, slightly slower)
      - mixtral-8x7b-32768   (large context window)

    temperature=0.2 → mostly factual, slight creativity allowed
    """
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


def _format_docs(docs):
    """Join retrieved document chunks into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


class RAGChain:
    """
    A conversational RAG chain using LangChain v1.x LCEL.

    Keeps the last 5 exchanges in memory (window memory).
    Uses the retriever to find relevant chunks, then sends
    context + chat history + question to the LLM.
    """

    def __init__(self, retriever, llm, max_history: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.max_history = max_history
        self.chat_history: list = []  # list of (HumanMessage, AIMessage) pairs

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        self.chain = (
            {
                "context": lambda x: _format_docs(
                    self.retriever.invoke(x["question"])
                ),
                "chat_history": lambda x: x["chat_history"],
                "question": lambda x: x["question"],
                "docs": lambda x: self.retriever.invoke(x["question"]),
            }
            | RunnablePassthrough.assign(
                answer=self.prompt | self.llm | StrOutputParser()
            )
        )

    def invoke(self, question: str) -> dict:
        """
        Send a question through the RAG chain.

        Returns a dict with:
          "answer"  → the LLM's response string
          "sources" → list of Document chunks used (for citations)
        """
        # Flatten chat history into message objects for the prompt
        history_messages = []
        for human_msg, ai_msg in self.chat_history[-self.max_history:]:
            history_messages.append(human_msg)
            history_messages.append(ai_msg)

        # Retrieve docs separately so we can return them as sources
        docs = self.retriever.invoke(question)

        result = (self.prompt | self.llm | StrOutputParser()).invoke({
            "context": _format_docs(docs),
            "chat_history": history_messages,
            "question": question,
        })

        answer = result

        # Save to memory
        self.chat_history.append((
            HumanMessage(content=question),
            AIMessage(content=answer),
        ))

        return {
            "answer": answer,
            "sources": docs,
        }


def get_rag_chain():
    """
    Build the full Conversational RAG chain.

    Uses a simple list-based window memory (last 5 exchanges).
    """
    retriever = get_retriever(k=4)
    llm = get_llm()
    return RAGChain(retriever=retriever, llm=llm, max_history=5)


def ask(chain, question: str) -> dict:
    """
    Send a question through the RAG chain.

    Returns a dict with:
      "answer"           → the LLM's response string
      "sources"          → list of Document chunks used (for citations)
    """
    return chain.invoke(question)
