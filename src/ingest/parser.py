"""
src/ingest/parser.py
Ekstraksi teks dari PDF (digital dan hasil scan/OCR).

- Digital PDF: via pdfplumber (teks langsung, akurat)
- Scanned PDF: via Tesseract OCR + Pillow (render ke gambar dulu, lalu OCR)
- Output per halaman: {'text': str, 'page': int}

Dependensi:
    pip install pdfplumber pytesseract Pillow
    # Juga perlu Tesseract binary + bahasa Indonesia:
    # Windows: https://github.com/UB-Mannheim/tesseract/wiki
    # Tambahkan tesseract-ocr-ind language pack
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Minimum karakter per halaman agar dianggap digital (bukan scan kosong)
MIN_CHARS_DIGITAL = 50


def extract_pages(pdf_path: str) -> List[dict]:
    """
    Ekstrak teks dari semua halaman PDF.
    Secara otomatis mendeteksi apakah PDF digital atau scan.

    Args:
        pdf_path: Path absolut ke file PDF

    Returns:
        List dict {'text': str, 'page': int} — satu entry per halaman
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {pdf_path}")

    try:
        pages = _extract_digital(str(path))
    except Exception as e:
        # pdfminer/pdfplumber kadang menolak PDF valid (mis. xref object stream).
        # Coba fallback pypdf yang lebih toleran.
        logger.warning(f"pdfplumber gagal ({type(e).__name__}), mencoba fallback pypdf...")
        pages = _extract_pypdf(str(path))
        if not pages:
            raise
    # Hitung halaman yang berhasil diekstrak teksnya
    good_pages = sum(1 for p in pages if len(p["text"].strip()) >= MIN_CHARS_DIGITAL)

    if good_pages < len(pages) * 0.5:
        # Lebih dari 50% halaman kosong → kemungkinan scan → gunakan OCR
        logger.info(
            f"PDF terdeteksi sebagai scan ({good_pages}/{len(pages)} halaman digital). "
            "Beralih ke OCR..."
        )
        pages = _extract_ocr(str(path))

    total_chars = sum(len(p["text"]) for p in pages)
    logger.info(
        f"Berhasil ekstrak {len(pages)} halaman dari '{path.name}' "
        f"(total {total_chars:,} karakter)."
    )
    return pages


def _extract_digital(pdf_path: str) -> List[dict]:
    """Ekstrak teks dari PDF digital via pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber belum terinstall. Jalankan: pip install pdfplumber")

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Gagal ekstrak halaman {i}: {e}")
                text = ""
            pages.append({"text": text, "page": i})
    return pages


def _extract_pypdf(pdf_path: str) -> List[dict]:
    """Fallback ekstraksi digital via pypdf (lebih toleran terhadap PDF yang
    ditolak pdfplumber, mis. struktur xref object stream yang ketat)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf belum terinstall. Jalankan: pip install pypdf")

    pages = []
    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Gagal ekstrak halaman {i} (pypdf): {e}")
            text = ""
        pages.append({"text": text, "page": i})
    return pages


def _extract_ocr(pdf_path: str) -> List[dict]:
    """Ekstrak teks dari PDF scan via Tesseract OCR."""
    try:
        import pytesseract
    except ImportError:
        raise ImportError(
            "pytesseract atau Pillow belum terinstall. "
            "Jalankan: pip install pytesseract Pillow"
        )

    # Set path Tesseract jika di Windows
    tesseract_cmd = os.getenv("TESSERACT_CMD", "")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # Render PDF ke gambar via pdf2image
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image belum terinstall. Jalankan: pip install pdf2image\n"
            "Juga perlu poppler: https://github.com/oschwartz10612/poppler-windows/releases"
        )

    pages = []
    try:
        images = convert_from_path(pdf_path, dpi=200)
    except Exception as e:
        logger.error(f"Gagal render PDF ke gambar: {e}")
        return []

    for i, image in enumerate(images, start=1):
        try:
            # OCR dengan bahasa Indonesia + Inggris
            text = pytesseract.image_to_string(image, lang="ind+eng")
        except Exception as e:
            logger.warning(f"Gagal OCR halaman {i}: {e}")
            text = ""
        pages.append({"text": text, "page": i})
        logger.debug(f"OCR halaman {i}/{len(images)} selesai ({len(text)} karakter)")

    return pages
