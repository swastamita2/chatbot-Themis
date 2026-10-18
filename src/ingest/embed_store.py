"""
src/ingest/embed_store.py
Utilitas bersama tahap akhir pipeline ingest: chunk → embedding (per batch)
→ insert ke vector store.

Dipakai sama oleh pipeline URL (src/ingest/pipeline.py) dan CLI ingest PDF
lokal (scripts/ingest_local_pdf.py) agar perilaku batch-embedding konsisten.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, List

logger = logging.getLogger(__name__)

# Tipe: menerima satu batch baris siap insert, mengembalikan jumlah baris
# yang berhasil disimpan.
InsertFn = Callable[[List[dict]], int]


def embed_and_insert(
    chunks: Iterable,
    insert_fn: InsertFn,
    batch_embed_size: int = 16,
) -> int:
    """
    Embed chunk per batch (mengontrol penggunaan memori) lalu insert.

    Args:
        chunks: Iterable objek Chunk (harus punya atribut `text` dan metode
                `to_dict()`).
        insert_fn: Fungsi insert vector store (mis. MilvusClient.insert_chunks).
        batch_embed_size: Jumlah chunk per batch saat embedding.

    Returns:
        Total chunk yang berhasil diinsert.
    """
    rows: List[dict] = []
    for chunk in chunks:
        d = chunk.to_dict()
        rows.append(d)

    from src.embeddings.bge_m3 import embed_texts

    total = 0
    for i in range(0, len(rows), batch_embed_size):
        batch_rows = rows[i:i + batch_embed_size]
        embeddings = embed_texts([r["text"] for r in batch_rows])
        for row, emb in zip(batch_rows, embeddings):
            row["embedding"] = emb
        inserted = insert_fn(batch_rows)
        total += inserted
        logger.info(
            f"[INGEST] Batch {i // batch_embed_size + 1}: "
            f"{inserted} chunk diinsert (total {total})."
        )
    return total
