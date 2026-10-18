"""
src/query/reranker.py
Reranker berbasis BGE-Reranker-v2-M3 (BAAI/bge-reranker-v2-m3).
Cross-encoder: score setiap (query, chunk) pair secara langsung.

Input: 20 kandidat chunk dari retriever
Output: 5 chunk terbaik dengan skor reranker
"""

from __future__ import annotations

import logging
import os
from typing import List

# Redam progress bar Hugging Face / loading weights (di-set sebelum
# huggingface_hub & transformers diimpor oleh sentence-transformers).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)

_reranker_instance = None


def _get_reranker():
    """Lazy-load BGE-Reranker model (singleton). ~1.1GB, diunduh otomatis."""
    global _reranker_instance
    if _reranker_instance is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers belum terinstall. "
                "Jalankan: pip install sentence-transformers"
            )
        logger.info(
            "Memuat BGE-Reranker-v2-M3... "
            "Model ~1.1GB, akan diunduh otomatis jika belum ada."
        )
        _reranker_instance = CrossEncoder("BAAI/bge-reranker-v2-m3")
        logger.info("BGE-Reranker-v2-M3 berhasil dimuat.")
    return _reranker_instance


def warmup() -> None:
    """Paksa lazy-load model cross-encoder sekali (dipanggil saat startup aplikasi)."""
    _get_reranker()


def rerank(query: str, chunks: List[dict], top_k: int = 5) -> List[dict]:
    """
    Rerank kandidat chunk menggunakan BGE-Reranker-v2-M3.

    Args:
        query: Pertanyaan asli pengguna
        chunks: List chunk dari retriever (biasanya 20)
        top_k: Jumlah chunk terbaik yang dikembalikan (biasanya 5)

    Returns:
        List chunk, diurutkan berdasarkan reranker score (descending).
        Setiap chunk mendapat field tambahan 'reranker_score'.
    """
    if not chunks:
        return []

    reranker = _get_reranker()

    # Buat pasangan (query, text) untuk setiap chunk
    pairs = [(query, chunk["text"]) for chunk in chunks]

    # Score semua pasangan sekaligus
    scores = reranker.predict(pairs)

    # Gabungkan score ke chunk dan urutkan
    for chunk, score in zip(chunks, scores):
        chunk["reranker_score"] = float(score)

    reranked = sorted(chunks, key=lambda x: x["reranker_score"], reverse=True)
    top = reranked[:top_k]

    logger.info(
        f"[RERANK] {len(chunks)} → {len(top)} chunk. "
        f"Score tertinggi: {top[0]['reranker_score']:.4f}" if top else "[RERANK] Kosong."
    )

    return top
