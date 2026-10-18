"""
scripts/ingest_local_pdf.py
CLI ingest untuk file PDF LOCAAL (tanpa scraping JDIH) ke Milvus.

Dipakai untuk migrasi korpus Themis dari regulasi ketenagalistrikan ke
regulasi pasar modal (POJK OJK + peraturan IDX), yang sudah diunduh sebagai
file PDF ke folder corpus_pasar_modal/.

Reuse komponen pipeline yang sudah ada:
  parser (extract_pages) -> chunker (chunk_document) -> BGE-M3 (embed_texts)
  -> Milvus (insert_chunks). Idempoten per nomor_peraturan (hapus chunk lama
  milik peraturan yang sama sebelum insert).

Penggunaan:
    python scripts/ingest_local_pdf.py --all --corpus "D:/Magang/yuta_agent/corpus_pasar_modal"
    python scripts/ingest_local_pdf.py --file "path/ke.pdf" --nomor "POJK No. 33 Tahun 2024" \
        --judul "Pengelolaan Investasi di Pasar Modal" --url "https://..."
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

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
logger = logging.getLogger("ingest_local_pdf")

# ── Metadata korpus pasar modal (5 dokumen) ─────────────────────────────────
# nomor_peraturan dipakai sebagai kunci idempotensi + sitasi di UI.
# url_sumber = URL publik sumber (untuk sitasi & traceability), bukan target scrape.
CORPUS = [
    {
        "file": "POJK-3-2026-Perusahaan-Efek.pdf",
        "nomor_peraturan": "POJK No. 3 Tahun 2026",
        "judul_peraturan": "Penyelenggaraan Kegiatan Usaha Perusahaan Efek yang Melakukan "
                           "Kegiatan Usaha sebagai Penjamin Emisi Efek dan Perantara Pedagang Efek",
        "url_sumber": "https://www.ojk.go.id/id/regulasi/halaman/regulasi",
    },
    {
        "file": "POJK-33-2024-Pengelolaan-Investasi.pdf",
        "nomor_peraturan": "POJK No. 33 Tahun 2024",
        "judul_peraturan": "Pengembangan dan Penguatan Pengelolaan Investasi di Pasar Modal",
        "url_sumber": "https://www.ojk.go.id/id/regulasi/halaman/regulasi",
    },
    {
        "file": "POJK-45-2024-Emiten-Perusahaan-Publik.pdf",
        "nomor_peraturan": "POJK No. 45 Tahun 2024",
        "judul_peraturan": "Pengembangan dan Penguatan Emiten dan Perusahaan Publik",
        "url_sumber": "https://www.ojk.go.id/id/regulasi/halaman/regulasi",
    },
    {
        "file": "IDX-II-A-Perdagangan-Efek-Ekuitas.pdf",
        "nomor_peraturan": "Peraturan No. II-A (Perubahan) tentang Perdagangan Efek Bersifat Ekuitas",
        "judul_peraturan": "Perubahan Peraturan Nomor II-A tentang Perdagangan Efek Bersifat Ekuitas "
                           "(Kep-00108/BEI/12-2020)",
        "url_sumber": "https://www.idx.co.id/perseroan-tercatat/peraturan-bursa",
    },
    {
        "file": "IDX-III-A-Keanggotaan-Bursa.pdf",
        "nomor_peraturan": "Peraturan No. III-A (Perubahan) tentang Keanggotaan Bursa",
        "judul_peraturan": "Perubahan Peraturan Nomor III-A tentang Keanggotaan Bursa "
                           "(Kep-00184/BEI/12-2018)",
        "url_sumber": "https://www.idx.co.id/perseroan-tercatat/peraturan-bursa",
    },
]


def ingest_local_pdf(
    pdf_path: str,
    nomor_peraturan: str,
    judul_peraturan: str,
    url_sumber: str,
    batch_embed_size: int = 16,
    clean_first: bool = True,
) -> dict:
    """Ingest satu PDF lokal ke Milvus. Kembalikan dict hasil."""
    from src.ingest.parser import extract_pages
    from src.ingest.chunker import chunk_document
    from src.ingest.embed_store import embed_and_insert
    from src.vector_store.milvus_client import get_client

    path = Path(pdf_path)
    out = {
        "file": path.name,
        "nomor_peraturan": nomor_peraturan,
        "success": False,
        "num_pages": 0,
        "num_chunks": 0,
        "num_inserted": 0,
        "error": None,
        "elapsed": 0.0,
    }
    t0 = time.time()
    try:
        # Idempotensi: hapus chunk lama milik peraturan yang sama dulu.
        client = get_client()
        if clean_first:
            client.delete_by_peraturan(nomor_peraturan)
            logger.info(f"[CLEAN] Chunk lama '{nomor_peraturan}' dibersihkan.")

        logger.info(f"[INGEST] Parse PDF -> {path.name}")
        pages = extract_pages(str(path))
        out["num_pages"] = len(pages)
        if not pages:
            out["error"] = "Tidak ada teks dari PDF"
            return out

        logger.info(f"[INGEST] Chunking -> '{nomor_peraturan}'")
        chunks = chunk_document(
            pages=pages,
            nomor_peraturan=nomor_peraturan,
            judul_peraturan=judul_peraturan,
            url_sumber=url_sumber,
        )
        out["num_chunks"] = len(chunks)
        if not chunks:
            out["error"] = "Tidak ada chunk"
            return out

        out["num_inserted"] = embed_and_insert(chunks, client.insert_chunks, batch_embed_size)
        out["success"] = True
        logger.info(f"[INGEST] OK '{nomor_peraturan}': {len(pages)} hal, "
                    f"{len(chunks)} chunk, {out['num_inserted']} diinsert.")
    except Exception as e:
        out["error"] = str(e)
        logger.error(f"[INGEST] GAGAL '{path.name}': {e}", exc_info=True)
    finally:
        out["elapsed"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest PDF lokal ke Milvus")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="Ingest seluruh korpus CORPUS")
    g.add_argument("--file", type=str, help="Path satu file PDF")
    ap.add_argument("--corpus", type=str, default=r"D:/Magang/yuta_agent/corpus_pasar_modal",
                    help="Folder berisi PDF korpus (default: corpus_pasar_modal)")
    ap.add_argument("--nomor", type=str, help="Nomor peraturan (mode --file)")
    ap.add_argument("--judul", type=str, help="Judul peraturan (mode --file)")
    ap.add_argument("--url", type=str, default="", help="URL sumber (mode --file)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--keep-stale", action="store_true",
                    help="Skip pembersihan chunk lama")
    args = ap.parse_args()

    clean = not args.keep_stale
    results = []

    if args.all:
        corpus_dir = Path(args.corpus)
        for meta in CORPUS:
            p = corpus_dir / meta["file"]
            if not p.exists():
                logger.warning(f"[SKIP] File tidak ditemukan: {p}")
                results.append({"file": meta["file"], "success": False,
                                "error": "file tidak ditemukan"})
                continue
            logger.info(f"===== {meta['file']} =====")
            results.append(ingest_local_pdf(
                str(p), meta["nomor_peraturan"], meta["judul_peraturan"],
                meta["url_sumber"], args.batch_size, clean,
            ))
    else:
        if not args.nomor:
            ap.error("--nomor wajib untuk mode --file")
        results.append(ingest_local_pdf(
            args.file, args.nomor, args.judul or args.nomor,
            args.url, args.batch_size, clean,
        ))

    # Ringkasan
    logger.info("=" * 60)
    ok = 0
    tot_chunks = 0
    for r in results:
        flag = "OK " if r.get("success") else "GAGAL"
        logger.info(f"[{flag}] {r['file']}: {r.get('num_pages',0)} hal, "
                    f"{r.get('num_chunks',0)} chunk, {r.get('num_inserted',0)} insert"
                    f"{(' - ' + str(r.get('error'))) if r.get('error') else ''}")
        if r.get("success"):
            ok += 1
            tot_chunks += r.get("num_inserted", 0)
    logger.info(f"Selesai: {ok}/{len(results)} dokumen sukses, total {tot_chunks} chunk.")
    if ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
