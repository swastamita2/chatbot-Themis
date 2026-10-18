"""
src/reliability/defense.py
Defense layer untuk Themis RAG.
- Validasi input query (panjang, karakter)
- Graceful degradation saat layanan eksternal down
"""

from __future__ import annotations

import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)

# Konfigurasi validasi
MIN_QUERY_LENGTH = 5
MAX_QUERY_LENGTH = 500


def validate_query(query: str) -> Tuple[bool, str]:
    """
    Validasi input query dari pengguna.

    Returns:
        Tuple (is_valid: bool, reason: str)
        reason kosong jika valid.
    """
    if not query or not query.strip():
        return False, "Pertanyaan tidak boleh kosong."

    query = query.strip()

    if len(query) < MIN_QUERY_LENGTH:
        return False, f"Pertanyaan terlalu pendek (minimum {MIN_QUERY_LENGTH} karakter)."

    if len(query) > MAX_QUERY_LENGTH:
        return False, (
            f"Pertanyaan terlalu panjang (maksimum {MAX_QUERY_LENGTH} karakter). "
            f"Anda memasukkan {len(query)} karakter."
        )

    # Cegah query yang hanya berisi simbol/angka tanpa kata bermakna
    words = re.findall(r"[a-zA-Z\u00C0-\u024F\u0100-\u017F\u4E00-\u9FFF\u0900-\u097F]+", query)
    if len(words) < 1:
        return False, "Pertanyaan harus mengandung setidaknya satu kata."

    return True, ""


def check_milvus_health() -> Tuple[bool, str]:
    """Cek koneksi Milvus."""
    try:
        from src.vector_store.milvus_client import get_client
        client = get_client()
        count = client.count()
        return True, f"OK ({count:,} chunk tersimpan)"
    except Exception as e:
        return False, str(e)


def check_llm_health() -> Tuple[bool, str]:
    """Cek koneksi LLM (quick ping)."""
    try:
        from llm_config import get_model
        llm = get_model()
        llm.invoke("ping")
        return True, "OK"
    except Exception as e:
        return False, str(e)


def check_langfuse_health() -> Tuple[bool, str]:
    """Cek koneksi Langfuse."""
    try:
        from langfuse import Langfuse
        import os
        lf = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        ok = lf.auth_check()
        return (True, "OK") if ok else (False, "Auth gagal")
    except Exception as e:
        return False, str(e)


def get_system_health() -> dict:
    """
    Cek kesehatan semua komponen sistem.
    Digunakan oleh FastAPI /health endpoint.

    Returns:
        Dict status setiap komponen. Sistem tetap "degraded" bukan "down"
        jika hanya Langfuse/Alerting yang bermasalah (PRD N3).
    """
    milvus_ok, milvus_msg = check_milvus_health()
    llm_ok, llm_msg = check_llm_health()
    langfuse_ok, langfuse_msg = check_langfuse_health()

    # Sistem "operational" selama Milvus + LLM OK
    # Langfuse opsional — boleh down (PRD N3)
    operational = milvus_ok and llm_ok

    return {
        "status": "operational" if operational else "degraded",
        "components": {
            "milvus": {"ok": milvus_ok, "message": milvus_msg},
            "llm": {"ok": llm_ok, "message": llm_msg},
            "langfuse": {"ok": langfuse_ok, "message": langfuse_msg, "required": False},
        },
    }
