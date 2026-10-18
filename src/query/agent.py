"""
src/query/agent.py
LangGraph agent untuk Themis RAG query pipeline.

Graph:
  [start] → retrieve → rerank → confidence_check → generate → [end]
                                      ↓ (low confidence)
                                   [no_answer]

State:
  question      : pertanyaan pengguna
  chunks        : chunk setelah reranking
  answer        : jawaban final dari LLM
  sources       : list metadata sumber untuk ditampilkan di UI
  confident     : apakah ada chunk dengan score ≥ threshold (PRD F7)
"""

from __future__ import annotations

import logging
from typing import List, TypedDict

logger = logging.getLogger(__name__)


# ─── State Schema ─────────────────────────────────────────────────────────────

class RAGState(TypedDict, total=False):
    question: str
    chunks: List[dict]
    answer: str
    sources: List[dict]
    confident: bool
    provider: str | None


# ─── Node Functions ───────────────────────────────────────────────────────────

def node_retrieve(state: RAGState) -> RAGState:
    """Node 1: Semantic search ke Milvus."""
    from src.query.retriever import retrieve
    from src.query.prompts import RETRIEVAL_TOP_K

    chunks = retrieve(state["question"], top_k=RETRIEVAL_TOP_K)
    return {**state, "chunks": chunks}


def node_rerank(state: RAGState) -> RAGState:
    """Node 2: Reranking dengan BGE-Reranker-v2-M3."""
    from src.query.reranker import rerank
    from src.query.prompts import RERANK_TOP_K

    top_chunks = rerank(state["question"], state["chunks"], top_k=RERANK_TOP_K)
    return {**state, "chunks": top_chunks}


def node_confidence_check(state: RAGState) -> RAGState:
    """Node 3: Cek apakah ada chunk cukup relevan (PRD F7)."""
    from src.query.prompts import LOW_CONFIDENCE_THRESHOLD

    chunks = state["chunks"]
    if not chunks:
        return {**state, "confident": False}

    # Gunakan reranker_score jika ada, fallback ke cosine score
    best_score = chunks[0].get("reranker_score", chunks[0].get("score", 0.0))
    confident = best_score >= LOW_CONFIDENCE_THRESHOLD

    if not confident:
        logger.info(
            f"[AGENT] Confidence rendah (score={best_score:.4f} < {LOW_CONFIDENCE_THRESHOLD}). "
            "Jawaban ditolak."
        )

    return {**state, "confident": confident}


def _build_messages(question: str, chunks: List[dict]):
    """Susun pesan (system + user) untuk LLM dari pertanyaan dan chunk."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.query.prompts import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, build_context

    context = build_context(chunks)
    user_message = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    return SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message)


def _extract_content(response) -> str:
    """Ekstrak teks jawaban dari respons LLM.

    `content` bisa str (normal) atau list of parts (beberapa provider).
    """
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return (content or "").strip()


def _collect_unique_sources(chunks: List[dict]) -> List[dict]:
    """Kumpulkan metadata sumber unik (per pasal) untuk ditampilkan di UI."""
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk.get("nomor_peraturan", ""), chunk.get("nomor_pasal", ""))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "nomor_peraturan": chunk.get("nomor_peraturan", ""),
            "judul_peraturan": chunk.get("judul_peraturan", ""),
            "nomor_pasal": chunk.get("nomor_pasal", ""),
            "nomor_ayat": chunk.get("nomor_ayat", ""),
            "url_sumber": chunk.get("url_sumber", ""),
            "halaman_pdf": chunk.get("halaman_pdf", 0),
            "text": chunk.get("text", ""),
            "reranker_score": round(chunk.get("reranker_score", 0.0), 4),
        })
    return sources


def node_generate(state: RAGState) -> RAGState:
    """Node 4: Generate jawaban dengan LLM + sitasi."""
    if not state.get("confident", False):
        from src.query.prompts import NO_CONTEXT_RESPONSE
        return {**state, "answer": NO_CONTEXT_RESPONSE, "sources": []}

    answer = _generate_with_llm(
        state["question"], state["chunks"], provider=state.get("provider")
    )
    return {**state, "answer": answer, "sources": _collect_unique_sources(state["chunks"])}


def _generate_with_llm(question: str, chunks: List[dict], provider: str | None = None) -> str:
    """Panggil LLM; retry 1x bila konten kosong; pesan aman bila gagal total.

    `provider` opsional (additive): None → provider default dari .env (jalur
    lama identik); bila diisi, model yang dipilih UI dipakai.
    """
    from llm_config import get_model
    from src.reliability.observability import get_langfuse_callback

    system_msg, user_msg = _build_messages(question, chunks)
    llm = get_model(temperature=0, provider=provider)

    # Callback Langfuse opsional (PRD N3) — tidak boleh menggagalkan jawaban.
    callbacks = []
    try:
        cb = get_langfuse_callback()
        if cb:
            callbacks.append(cb)
    except Exception:
        pass
    invoke_config = {"callbacks": callbacks} if callbacks else {}

    try:
        answer = _extract_content(llm.invoke([system_msg, user_msg], config=invoke_config))
        # Pengaman: budget max_tokens bisa habis untuk reasoning tersembunyi
        # sehingga content kosong — coba satu kali lagi.
        if not answer:
            logger.warning("[AGENT] LLM mengembalikan konten kosong, mencoba ulang...")
            answer = _extract_content(llm.invoke([system_msg, user_msg], config=invoke_config))
        if not answer:
            return (
                "Maaf, model gagal menghasilkan jawaban untuk pertanyaan ini "
                "pada percobaan ini. Silakan coba lagi."
            )
        return answer
    except Exception as e:
        logger.error(f"[AGENT] LLM error: {e}")
        return (
            f"Terjadi kesalahan saat memproses pertanyaan Anda. "
            f"Silakan coba kembali. (Error: {type(e).__name__})"
        )


# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_graph():
    """Membangun dan mengkompilasi LangGraph agent graph."""
    from langgraph.graph import StateGraph, END

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", node_retrieve)
    graph.add_node("rerank", node_rerank)
    graph.add_node("confidence_check", node_confidence_check)
    graph.add_node("generate", node_generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "confidence_check")
    graph.add_edge("confidence_check", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ─── Public API ───────────────────────────────────────────────────────────────

_graph = None


def get_graph():
    """Singleton compiled graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(question: str, provider: str | None = None) -> dict:
    """
    Jalankan query RAG end-to-end.

    Args:
        question: Pertanyaan dalam bahasa natural
        provider: (opsional) nama provider LLM dari UI; None → default .env

    Returns:
        Dict dengan keys: answer (str), sources (list), confident (bool)
    """
    from src.reliability.defense import validate_query

    # Validasi input (PRD — defense layer)
    valid, reason = validate_query(question)
    if not valid:
        return {
            "answer": f"Pertanyaan tidak valid: {reason}",
            "sources": [],
            "confident": False,
        }

    initial_state: RAGState = {
        "question": question,
        "chunks": [],
        "answer": "",
        "sources": [],
        "confident": False,
    }
    if provider:
        initial_state["provider"] = provider

    graph = get_graph()
    final_state = graph.invoke(initial_state)

    return {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "confident": final_state.get("confident", False),
    }


def run_rag_agent(query: str, top_k: int = 5, provider: str | None = None) -> dict:
    """
    Alias yang dipakai FastAPI (/api/chat) & frontend Streamlit.
    Menjalankan query RAG end-to-end dan menormalkan struktur sumber respons.
    `provider` opsional (additive): None → model default .env.
    """
    result = run_query(query, provider=provider)

    norm_sources = []
    for i, src in enumerate(result.get("sources", [])[:top_k], start=1):
        norm_sources.append({
            "index": i,
            "nomor_peraturan": src.get("nomor_peraturan", ""),
            "judul_peraturan": src.get("judul_peraturan", ""),
            "nomor_pasal": src.get("nomor_pasal", ""),
            "nomor_ayat": src.get("nomor_ayat", ""),
            "url_sumber": src.get("url_sumber", ""),
            "halaman_pdf": src.get("halaman_pdf", 0),
            "score": src.get("reranker_score", 0.0),
            "text": (src.get("text", "") or "")[:500],
        })

    return {
        "answer": result.get("answer", ""),
        "sources": norm_sources,
        "confidence_passed": result.get("confident", False),
    }
