"""
src/warmup.py
Prakisi (warmup) komponen berat sekali saat startup aplikasi agar query
pertama tidak menahan latensi:
- model embedding BGE-M3  (~40 dtk load dari cache)
- reranker BGE-Reranker-v2-M3 (cross-encoder, ~45 dtk load)
- koneksi + load koleksi Milvus

Panggilan berulang aman (semua loader memakai pola singleton-guard).
"""

from __future__ import annotations

import logging
import os
import time

# Redam progress bar unduhan/loade Hugging Face (harus di-set sebelum
# huggingface_hub diimpor agar terbaca saat modul diinisialisasi).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

logger = logging.getLogger(__name__)


def warmup_all() -> dict:
    """
    Muat embedding, reranker, dan koneksi Milvus.
    Mengembalikan dict waktu tiap tahap (detik) untuk ditampilkan/audit.
    """
    timings: dict = {}

    t0 = time.time()
    from src.embeddings.bge_m3 import warmup as warmup_embed
    warmup_embed()
    timings["embed"] = round(time.time() - t0, 2)
    logger.info(f"[WARMUP] BGE-M3 siap ({timings['embed']}s)")

    t0 = time.time()
    from src.query.reranker import warmup as warmup_rerank
    warmup_rerank()
    timings["reranker"] = round(time.time() - t0, 2)
    logger.info(f"[WARMUP] BGE-Reranker siap ({timings['reranker']}s)")

    t0 = time.time()
    from src.vector_store.milvus_client import warmup as warmup_milvus
    warmup_milvus()
    timings["milvus"] = round(time.time() - t0, 2)
    logger.info(f"[WARMUP] Milvus siap ({timings['milvus']}s)")

    return timings
