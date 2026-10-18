"""
src/reliability/observability.py
Integrasi Langfuse untuk observability Themis RAG.

- Log setiap request (prompt, token, latency) via LangChain callback
- Graceful degradation: jika Langfuse down/tidak dikonfigurasi →
  sistem tetap jalan tanpa tracing (PRD N3)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

_callback_handler = None


def get_langfuse_callback():
    """
    Mengembalikan LangChain CallbackHandler untuk Langfuse (singleton).
    Mengembalikan None jika Langfuse tidak dikonfigurasi (graceful degradation).
    """
    global _callback_handler
    if _callback_handler is not None:
        return _callback_handler

    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    if not pk or not sk:
        logger.info(
            "[OBS] LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY belum dikonfigurasi. "
            "Observability tidak aktif."
        )
        return None

    try:
        # Langfuse v3 (langfuse.callback) — API lama.
        from langfuse.callback import CallbackHandler  # type: ignore

        _callback_handler = CallbackHandler(
            public_key=pk,
            secret_key=sk,
            host=host,
            session_id="themis-rag",
        )
        logger.info(f"[OBS] Langfuse terhubung ke {host}")
        return _callback_handler
    except ImportError:
        pass

    try:
        # Langfuse v4 (langfuse.langchain) — kredensial dibaca dari environment.
        from langfuse.langchain import CallbackHandler  # type: ignore

        os.environ["LANGFUSE_PUBLIC_KEY"] = pk
        os.environ["LANGFUSE_SECRET_KEY"] = sk
        os.environ["LANGFUSE_HOST"] = host
        _callback_handler = CallbackHandler()
        logger.info(f"[OBS] Langfuse terhubung ke {host}")
        return _callback_handler
    except Exception as e:
        logger.warning(f"[OBS] Gagal membuat Langfuse callback: {e}")
        return None
