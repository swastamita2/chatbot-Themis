r"""
src/ingest/chunker.py
Pemecah dokumen berbasis struktur Pasal untuk regulasi Indonesia
(pasar modal: POJK OJK & Peraturan Bursa Efek Indonesia).

Strategi chunking (sesuai PRD F3):
- Unit utama: Per Pasal (regex Pasal \d+ atau Pasal [IVXLC]+)
- Sub-chunk: Per ayat (1), (2), ... jika Pasal > max_tokens
- Fallback: Fixed-size (512 token, overlap 50) jika tidak ada struktur Pasal
- Metadata wajib per chunk: nomor_peraturan, judul_peraturan, nomor_pasal,
  nomor_ayat, url_sumber, halaman_pdf
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

# Konfigurasi
MAX_TOKENS = 512
OVERLAP_TOKENS = 50
CHARS_PER_TOKEN = 4   # estimasi rata-rata karakter per token (BPE)
MAX_CHARS = MAX_TOKENS * CHARS_PER_TOKEN          # ~2048 karakter
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN  # ~200 karakter

# Regex patterns
_RE_PASAL = re.compile(
    r"(?:^|\n)\s*(Pasal\s+(?:\d+|[IVXLCDM]+))\b",
    re.IGNORECASE | re.MULTILINE,
)
_RE_AYAT = re.compile(
    r"(?:^|\n)\s*\((\d+)\)\s",
    re.MULTILINE,
)
_RE_NOMOR_PASAL = re.compile(
    r"Pasal\s+((?:\d+|[IVXLCDM]+))",
    re.IGNORECASE,
)


@dataclass
class Chunk:
    """Representasi satu potongan teks dengan metadata lengkap."""
    text: str
    nomor_peraturan: str
    judul_peraturan: str
    nomor_pasal: str
    nomor_ayat: str
    url_sumber: str
    halaman_pdf: int = 0

    def to_dict(self) -> dict:
        return {
            "text": self.text.strip(),
            "nomor_peraturan": self.nomor_peraturan,
            "judul_peraturan": self.judul_peraturan,
            "nomor_pasal": self.nomor_pasal,
            "nomor_ayat": self.nomor_ayat,
            "url_sumber": self.url_sumber,
            "halaman_pdf": self.halaman_pdf,
        }


def chunk_document(
    pages: List[dict],
    nomor_peraturan: str,
    judul_peraturan: str,
    url_sumber: str,
) -> List[Chunk]:
    """
    Memecah dokumen menjadi chunk berbasis struktur Pasal.

    Args:
        pages: List dict {'text': str, 'page': int} dari parser.py
        nomor_peraturan: Misal "Permen ESDM No. 7 Tahun 2021"
        judul_peraturan: Judul lengkap peraturan
        url_sumber: URL PDF sumber di JDIH ESDM

    Returns:
        List Chunk siap untuk di-embed dan disimpan ke Milvus
    """
    # Gabungkan semua halaman menjadi satu teks, dengan penanda halaman
    full_text = ""
    page_map: List[tuple[int, int]] = []  # (char_start, page_number)
    for page_data in pages:
        start = len(full_text)
        full_text += page_data["text"] + "\n"
        page_map.append((start, page_data.get("page", 0)))

    def _get_page(char_pos: int) -> int:
        """Mengembalikan nomor halaman berdasarkan posisi karakter."""
        page = 0
        for start, pg in page_map:
            if char_pos >= start:
                page = pg
        return page

    # Cari semua posisi "Pasal X"
    pasal_matches = list(_RE_PASAL.finditer(full_text))

    if len(pasal_matches) < 2:
        # Fallback: tidak ada struktur Pasal yang cukup → fixed-size chunking
        logger.info(
            f"Tidak ditemukan struktur Pasal pada '{nomor_peraturan}'. "
            "Menggunakan fixed-size chunking."
        )
        return _fixed_size_chunks(
            full_text, nomor_peraturan, judul_peraturan, url_sumber, _get_page
        )

    chunks: List[Chunk] = []

    for i, match in enumerate(pasal_matches):
        pasal_label = match.group(1).strip()
        nomor_match = _RE_NOMOR_PASAL.search(pasal_label)
        nomor_pasal = nomor_match.group(1) if nomor_match else pasal_label

        # Ambil teks dari awal Pasal ini sampai awal Pasal berikutnya
        start = match.start()
        end = pasal_matches[i + 1].start() if i + 1 < len(pasal_matches) else len(full_text)
        pasal_text = full_text[start:end].strip()
        page_num = _get_page(start)

        if len(pasal_text) <= MAX_CHARS:
            # Pasal cukup pendek → satu chunk
            chunks.append(Chunk(
                text=pasal_text,
                nomor_peraturan=nomor_peraturan,
                judul_peraturan=judul_peraturan,
                nomor_pasal=nomor_pasal,
                nomor_ayat="",
                url_sumber=url_sumber,
                halaman_pdf=page_num,
            ))
        else:
            # Pasal panjang → sub-chunk per ayat
            ayat_chunks = _split_by_ayat(
                pasal_text, nomor_peraturan, judul_peraturan,
                nomor_pasal, url_sumber, page_num
            )
            chunks.extend(ayat_chunks)

    logger.info(
        f"'{nomor_peraturan}': {len(pasal_matches)} Pasal → {len(chunks)} chunk"
    )
    return chunks


def _split_by_ayat(
    pasal_text: str,
    nomor_peraturan: str,
    judul_peraturan: str,
    nomor_pasal: str,
    url_sumber: str,
    halaman_pdf: int,
) -> List[Chunk]:
    """Sub-chunk satu Pasal panjang berdasarkan ayat (1), (2), ..."""
    ayat_matches = list(_RE_AYAT.finditer(pasal_text))

    if not ayat_matches:
        # Tidak ada ayat → potong fixed-size
        return _fixed_size_text_chunks(
            pasal_text, nomor_peraturan, judul_peraturan,
            nomor_pasal, url_sumber, halaman_pdf
        )

    chunks: List[Chunk] = []
    # Header Pasal (sebelum ayat pertama)
    header = pasal_text[:ayat_matches[0].start()].strip()

    for i, m in enumerate(ayat_matches):
        nomor_ayat = m.group(1)
        start = m.start()
        end = ayat_matches[i + 1].start() if i + 1 < len(ayat_matches) else len(pasal_text)
        ayat_text = (header + "\n" + pasal_text[start:end]).strip()

        chunks.append(Chunk(
            text=ayat_text,
            nomor_peraturan=nomor_peraturan,
            judul_peraturan=judul_peraturan,
            nomor_pasal=nomor_pasal,
            nomor_ayat=nomor_ayat,
            url_sumber=url_sumber,
            halaman_pdf=halaman_pdf,
        ))

    return chunks


def _fixed_size_text_chunks(
    text: str,
    nomor_peraturan: str,
    judul_peraturan: str,
    nomor_pasal: str,
    url_sumber: str,
    halaman_pdf: int,
) -> List[Chunk]:
    """Potong teks menjadi chunk fixed-size dengan overlap."""
    chunks: List[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                nomor_peraturan=nomor_peraturan,
                judul_peraturan=judul_peraturan,
                nomor_pasal=nomor_pasal,
                nomor_ayat="",
                url_sumber=url_sumber,
                halaman_pdf=halaman_pdf,
            ))
        start = end - OVERLAP_CHARS if end < len(text) else end
    return chunks


def _fixed_size_chunks(
    full_text: str,
    nomor_peraturan: str,
    judul_peraturan: str,
    url_sumber: str,
    get_page_fn,
) -> List[Chunk]:
    """Fixed-size chunking untuk dokumen tanpa struktur Pasal."""
    chunks: List[Chunk] = []
    start = 0
    while start < len(full_text):
        end = min(start + MAX_CHARS, len(full_text))
        chunk_text = full_text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                nomor_peraturan=nomor_peraturan,
                judul_peraturan=judul_peraturan,
                nomor_pasal="",
                nomor_ayat="",
                url_sumber=url_sumber,
                halaman_pdf=get_page_fn(start),
            ))
        start = end - OVERLAP_CHARS if end < len(full_text) else end
    return chunks
