"""
src/query/retriever.py
Semantic search ke Milvus menggunakan BGE-M3 query embedding.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)


def retrieve(query: str, top_k: int = 20) -> List[dict]:
    """
    Embed query dengan BGE-M3 dan lakukan similarity search ke Milvus.

    Args:
        query: Pertanyaan pengguna dalam bahasa natural
        top_k: Jumlah kandidat chunk yang dikembalikan sebelum reranking

    Returns:
        List chunk dict, diurutkan berdasarkan cosine similarity (descending).
        Setiap dict memiliki keys: text, score, nomor_peraturan, nomor_pasal,
        nomor_ayat, judul_peraturan, url_sumber, halaman_pdf
    """
    from src.embeddings.bge_m3 import embed_query
    from src.vector_store.milvus_client import get_client

    logger.info(f"[RETRIEVE] Query: '{query[:80]}...' — top_k={top_k}")

    query_embedding = embed_query(query)
    client = get_client()
    results = client.search(query_embedding, top_k=top_k)

    logger.info(
        f"[RETRIEVE] {len(results)} hasil ditemukan. "
        f"Score tertinggi: {results[0]['score']:.4f}" if results else "[RETRIEVE] Tidak ada hasil."
    )

    return results
