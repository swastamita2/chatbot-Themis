"""
src/embeddings/bge_m3.py
Wrapper singleton untuk BGE-M3 embedding model (BAAI/bge-m3).
Dijalankan in-process via sentence-transformers — tidak sebagai microservice terpisah.
Output dimension: 1024 (dense vector).
"""

from __future__ import annotations

import logging
import os
from typing import List

import numpy as np

# Redam progress bar Hugging Face / loading weights (di-set sebelum
# huggingface_hub & transformers diimpor oleh sentence-transformers).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)

_model_instance = None


def _get_model():
    """Lazy-load BGE-M3 model (singleton). Download otomatis saat pertama kali dipanggil."""
    global _model_instance
    if _model_instance is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers belum terinstall. "
                "Jalankan: pip install sentence-transformers"
            )
        logger.info("Memuat BGE-M3 embedding model (BAAI/bge-m3)... "
                    "Model ~570MB, akan diunduh otomatis jika belum ada.")
        _model_instance = SentenceTransformer("BAAI/bge-m3")
        logger.info("BGE-M3 berhasil dimuat.")
    return _model_instance


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Mengubah list teks menjadi list embedding vektor (dimensi 1024).

    Args:
        texts: List string yang akan di-embed.
        batch_size: Ukuran batch untuk inference (sesuaikan dengan memori GPU/CPU).

    Returns:
        List of float lists, shape: (len(texts), 1024)
    """
    if not texts:
        return []
    model = _get_model()
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # Cosine similarity siap pakai
        show_progress_bar=len(texts) > 10,
    )
    return embeddings.tolist()


def warmup() -> None:
    """Paksa lazy-load model BGE-M3 sekali (dipanggil saat startup aplikasi)."""
    _get_model()


def embed_query(text: str) -> List[float]:
    """
    Mengubah satu teks query menjadi embedding vektor.
    Menggunakan query instruction prefix untuk BGE-M3.
    """
    # BGE-M3 merekomendasikan prefix untuk query retrieval
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    result = embed_texts([prefixed])
    return result[0]
