"""Starbucks Decision Intelligence — Streamlit chatbot UI."""

import os
from typing import Optional

import requests
import streamlit as st

# ── page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="☕ Starbucks Decision Intelligence",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API endpoint ──────────────────────────────────────────────────────────────
API_URL: str = os.getenv("API_URL", "http://localhost:8000")

# ── quick query definitions ───────────────────────────────────────────────────
QUICK_QUERIES: list[str] = [
    "Why are sales dropping in Pacific Northwest?",
    "What are the top customer churn drivers?",
    "Which inventory items have the most stockouts?",
    "How is the loyalty program performing?",
    "Compare store performance across regions",
    "What is ROI of the Happy Hour promotion?",
]

# ── session state initialisation ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "query_count" not in st.session_state:
    st.session_state["query_count"] = 0


# ── helper functions ──────────────────────────────────────────────────────────

def send_query(question: str) -> dict:
    """POST a question to the RAG API and return the JSON response.

    Args:
        question: The business question to send.

    Returns:
        Parsed JSON response dict, or a dict with an ``"error"`` key on failure.
    """
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question, "top_k": 5},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Start uvicorn on port 8000."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out."}
    except Exception as exc:
        return {"error": str(exc)}


def handle_response(question: str) -> None:
    """Send *question* to the API, render the exchange, and persist it to history.

    Appends both the user turn and the assistant turn (with metrics) to
    ``st.session_state["messages"]`` and updates the sidebar metric cache.

    Args:
        question: The business question entered or selected by the user.
    """
    # ── user turn ─────────────────────────────────────────────────────────────
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.write(question)

    # ── assistant turn ────────────────────────────────────────────────────────
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching enterprise data..."):
            data = send_query(question)

        if "error" in data:
            st.error(data["error"])
            st.session_state["messages"].append(
                {"role": "assistant", "content": data["error"], "is_error": True}
            )
            return

        answer: str = data.get("answer", "")
        latency_ms: int = data.get("latency_ms", 0)
        chunks_retrieved: int = data.get("chunks_retrieved", 0)
        groundedness: float = data.get("groundedness_score", 0.0)
        intent: str = data.get("intent", "general")
        sources: list[str] = data.get("sources", [])
        confidence_pct: int = round(groundedness * 100)

        st.write(answer)

        c1, c2, c3 = st.columns(3)
        c1.metric("⏱ Latency", f"{latency_ms} ms")
        c2.metric("📦 Chunks", chunks_retrieved)
        c3.metric("🎯 Confidence", f"{confidence_pct}%")

        if sources:
            st.caption(f"📄 Sources: {' · '.join(sources)}")

    # ── update session state ──────────────────────────────────────────────────
    st.session_state["query_count"] += 1
    st.session_state["last_metrics"] = {
        "latency_ms": latency_ms,
        "chunks_retrieved": chunks_retrieved,
        "confidence_pct": confidence_pct,
        "intent": intent,
    }
    st.session_state["last_sources"] = sources
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "latency_ms": latency_ms,
            "chunks_retrieved": chunks_retrieved,
            "groundedness_score": groundedness,
            "intent": intent,
            "sources": sources,
            "is_error": False,
        }
    )


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☕ Starbucks RAG Intelligence")
    st.markdown("**Enterprise Decision Platform**")
    st.divider()

    # quick queries
    st.markdown("### ⚡ Quick Queries")
    for query_text in QUICK_QUERIES:
        if st.button(query_text, use_container_width=True):
            st.session_state["pending_query"] = query_text

    st.divider()

    # pipeline metrics (2 × 2 grid)
    st.markdown("### 📊 Pipeline Metrics")
    if "last_metrics" in st.session_state:
        m = st.session_state["last_metrics"]
        col_a, col_b = st.columns(2)
        col_a.metric("⏱ Latency", f"{m['latency_ms']} ms")
        col_b.metric("📦 Chunks", m["chunks_retrieved"])
        col_a.metric("🎯 Confidence", f"{m['confidence_pct']}%")
        col_b.metric("🧠 Intent", m["intent"])
    else:
        st.caption("Run a query to see metrics")

    st.divider()

    # retrieved sources
    st.markdown("### 📄 Retrieved Sources")
    if "last_sources" in st.session_state and st.session_state["last_sources"]:
        for src in st.session_state["last_sources"]:
            st.markdown(f"• `{src}`")
    else:
        st.caption("Sources will appear after a query")

    st.divider()

    # session query count
    count = st.session_state["query_count"]
    st.caption(f"🔢 Total queries this session: **{count}**")


# ── main area ─────────────────────────────────────────────────────────────────
st.title("☕ Starbucks Decision Intelligence")
st.markdown(
    "Ask natural-language questions about Starbucks operational data — "
    "sales, inventory, loyalty, and store performance."
)
st.divider()

# replay full message history
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(msg["content"])
            if not msg.get("is_error"):
                c1, c2, c3 = st.columns(3)
                c1.metric("⏱ Latency", f"{msg['latency_ms']} ms")
                c2.metric("📦 Chunks", msg["chunks_retrieved"])
                c3.metric("🎯 Confidence", f"{round(msg['groundedness_score'] * 100)}%")
                if msg.get("sources"):
                    st.caption(f"📄 Sources: {' · '.join(msg['sources'])}")

# ── pending quick query ───────────────────────────────────────────────────────
if "pending_query" in st.session_state:
    pending = st.session_state.pop("pending_query")
    handle_response(pending)
    st.rerun()

# ── chat input ────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask about sales, inventory, loyalty, store performance..."):
    handle_response(question)
    st.rerun()
