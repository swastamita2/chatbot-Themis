"""
src/ingest/pipeline.py
Orchestrator pipeline ingest end-to-end (sequential mode).
Flow: URL → Scraper → Parser → Chunker → BGE-M3 Embedder → Milvus

Untuk pengembangan lanjut (multi-dokumen besar): tambah RQ job dispatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Hasil ingest satu dokumen."""
    url: str
    nomor_peraturan: str = ""
    judul_peraturan: str = ""
    pdf_path: Optional[str] = None
    num_pages: int = 0
    num_chunks: int = 0
    num_inserted: int = 0
    success: bool = False
    error: Optional[str] = None


def ingest_document(url: str, batch_embed_size: int = 16) -> IngestResult:
    """
    Ingest satu dokumen dari URL ke Milvus.

    Args:
        url: URL halaman detail atau PDF langsung dari JDIH ESDM
        batch_embed_size: Jumlah chunk per batch saat embedding

    Returns:
        IngestResult dengan statistik dan status
    """
    from src.ingest.scraper import JdihScraper
    from src.ingest.parser import extract_pages
    from src.ingest.chunker import chunk_document
    from src.ingest.embed_store import embed_and_insert
    from src.vector_store.milvus_client import get_client

    result = IngestResult(url=url)

    try:
        # ── Step 1: Scrape & Download PDF ─────────────────────────────────
        logger.info(f"[INGEST] Step 1/4: Scraping → {url}")
        scraper = JdihScraper()
        pdf_path, meta = scraper.process_url(url)

        result.nomor_peraturan = meta.nomor_peraturan
        result.judul_peraturan = meta.judul

        if pdf_path is None:
            result.error = "Gagal mengunduh PDF"
            return result

        result.pdf_path = str(pdf_path)

        # ── Step 2: Parse PDF ──────────────────────────────────────────────
        logger.info(f"[INGEST] Step 2/4: Parsing PDF → {pdf_path.name}")
        pages = extract_pages(str(pdf_path))
        result.num_pages = len(pages)

        if not pages:
            result.error = "Tidak ada teks yang berhasil diekstrak dari PDF"
            return result

        # ── Step 3: Chunk ──────────────────────────────────────────────────
        logger.info(f"[INGEST] Step 3/4: Chunking ({len(pages)} halaman) → '{meta.nomor_peraturan}'")
        chunks = chunk_document(
            pages=pages,
            nomor_peraturan=meta.nomor_peraturan,
            judul_peraturan=meta.judul,
            url_sumber=url,
        )
        result.num_chunks = len(chunks)

        if not chunks:
            result.error = "Tidak ada chunk yang dihasilkan dari dokumen"
            return result

        logger.info(f"[INGEST] {len(chunks)} chunk dihasilkan.")

        # ── Step 4: Embed + Insert ke Milvus ──────────────────────────────
        logger.info(f"[INGEST] Step 4/4: Embedding {len(chunks)} chunk (batch={batch_embed_size})...")
        client = get_client()
        result.num_inserted = embed_and_insert(chunks, client.insert_chunks, batch_embed_size)
        result.success = True
        logger.info(
            f"[INGEST] ✓ Selesai: '{meta.nomor_peraturan}' — "
            f"{result.num_pages} hal, {result.num_chunks} chunk, "
            f"{result.num_inserted} diinsert."
        )

    except Exception as e:
        result.error = str(e)
        logger.error(f"[INGEST] ✗ Gagal ingest '{url}': {e}", exc_info=True)

        # Kirim alert jika ingest pipeline gagal (PRD F9)
        try:
            from alerting import alert_manager
            alert_manager.send_alert(
                subject="[Themis RAG] Ingest Pipeline Gagal",
                message=f"URL: {url}\nError: {e}",
            )
        except Exception as alert_err:
            logger.warning(f"Gagal kirim alert: {alert_err}")

    return result
