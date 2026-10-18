"""
scripts/run_ingest.py
CLI runner untuk pipeline ingest dokumen Themis RAG.

Penggunaan:
    # Ingest satu URL:
    python scripts/run_ingest.py --url "https://jdih.esdm.go.id/dokumen/view?id=2428"

    # Ingest dari file daftar URL:
    python scripts/run_ingest.py --file data/urls.txt

    # Atur batch size embedding:
    python scripts/run_ingest.py --file data/urls.txt --batch-size 32

Skrip ini bersifat IDEMPOTEN: sebelum insert, chunk lama milik peraturan yang
sama (ditandai metadata nomor_peraturan) dihapus dulu, sehingga re-ingest
dokumen yang sama tidak menghasilkan duplikasi di Milvus.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Pastikan root direktori ada di sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_ingest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI Ingestion Pipeline untuk Themis RAG")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--url",
        type=str,
        help="URL detail peraturan atau file PDF langsung dari JDIH ESDM",
    )
    group.add_argument(
        "--file",
        type=str,
        help="Path ke file teks berisi daftar URL (satu URL per baris)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Ukuran batch chunk untuk proses embedding (default: 16)",
    )
    parser.add_argument(
        "--keep-stale",
        action="store_true",
        help="SKIP pembersihan chunk lama (default: chunk lama peraturan yang sama dihapus)",
    )
    return parser.parse_args()


def read_urls(path: Path) -> list[str]:
    """Baca daftar URL dari file (baris kosong & komentar '#' diabaikan)."""
    if not path.exists():
        logger.error(f"File tidak ditemukan: {path}")
        sys.exit(1)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def ingest_one(url: str, batch_size: int, clean_first: bool) -> bool:
    """Ingest satu URL; kembalikan True jika sukses. Bersihkan chunk lama dulu."""
    from src.ingest.pipeline import ingest_document
    from src.ingest.scraper import JdihScraper
    from src.vector_store.milvus_client import get_client

    # Idempotensi: hapus chunk lama milik peraturan yang sama (jika ada)
    # SEBELUM insert, agar re-ingest tidak menghasilkan duplikasi.
    if clean_first:
        try:
            meta = JdihScraper().get_metadata(url)
            if meta.nomor_peraturan:
                get_client().delete_by_peraturan(meta.nomor_peraturan)
                logger.info(f"[CLEAN] Chunk lama '{meta.nomor_peraturan}' dibersihkan.")
        except Exception as e:
            logger.warning(f"[CLEAN] Pembersihan dilewati: {e}")

    result = ingest_document(url, batch_embed_size=batch_size)
    if result.success:
        logger.info(
            f"[SUKSES] '{result.nomor_peraturan}' | {result.num_pages} hal, "
            f"{result.num_chunks} chunk, {result.num_inserted} diinsert"
        )
        return True
    logger.error(f"[GAGAL] {url} | Error: {result.error}")
    return False


def main() -> None:
    args = parse_args()
    clean_first = not args.keep_stale

    urls = [args.url.strip()] if args.url else read_urls(Path(args.file))
    if not urls:
        logger.warning("Tidak ada URL valid untuk di-ingest.")
        sys.exit(0)

    logger.info(f"Memulai ingest {len(urls)} dokumen (batch_size={args.batch_size})...")
    ok = 0
    for i, url in enumerate(urls, start=1):
        logger.info(f"({i}/{len(urls)}) → {url}")
        if ingest_one(url, args.batch_size, clean_first):
            ok += 1

    logger.info(f"Selesai: {ok}/{len(urls)} dokumen berhasil di-ingest.")
    if ok < len(urls):
        sys.exit(1)


if __name__ == "__main__":
    main()
