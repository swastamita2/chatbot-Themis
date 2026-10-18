"""
src/ingest/scraper.py
Scraper modular untuk mengunduh dokumen PDF dari JDIH Kementerian ESDM.
URL sumber: https://jdih.esdm.go.id/

Fitur:
- Rate limiting antar-request (sesuai PRD N4)
- Simpan PDF ke data/raw/ dengan nama file yang aman
- Metadata parsing dari halaman detail JDIH (judul, nomor peraturan)
- Desain modular — mudah disesuaikan jika struktur situs berubah (Risiko 5)
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Konfigurasi
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
REQUEST_DELAY_SECONDS = 2.0   # Jeda antar-request (PRD N4)
REQUEST_TIMEOUT = 30
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ThemisRAG/1.0; "
        "+https://github.com/your-username/themis-rag)"
    ),
    "Accept": "text/html,application/pdf,*/*",
}


class DocumentMetadata:
    """Metadata yang diekstrak dari halaman detail JDIH ESDM."""
    def __init__(
        self,
        url: str,
        judul: str = "",
        nomor_peraturan: str = "",
        pdf_url: str = "",
    ):
        self.url = url
        self.judul = judul
        self.nomor_peraturan = nomor_peraturan
        self.pdf_url = pdf_url

    def __repr__(self) -> str:
        return f"<DocumentMetadata nomor='{self.nomor_peraturan}' judul='{self.judul[:50]}...'>"


class JdihScraper:
    """
    Scraper untuk situs JDIH Kementerian ESDM (jdih.esdm.go.id).

    Penggunaan:
        scraper = JdihScraper()
        metadata = scraper.get_metadata("https://jdih.esdm.go.id/storage/...")
        pdf_path = scraper.download_pdf(metadata)
    """

    BASE_URL = "https://jdih.esdm.go.id"

    def __init__(self, delay: float = REQUEST_DELAY_SECONDS):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _get(self, url: str) -> requests.Response:
        """HTTP GET dengan rate limiting dan error handling."""
        time.sleep(self.delay)
        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            logger.error(f"Gagal mengakses {url}: {e}")
            raise

    def get_metadata(self, url: str) -> DocumentMetadata:
        """
        Ekstrak metadata dari halaman detail dokumen JDIH ESDM.

        Mendukung dua jenis URL:
        - Halaman detail HTML: https://jdih.esdm.go.id/peraturan/detail/...
        - URL langsung ke PDF: https://jdih.esdm.go.id/storage/.../file.pdf

        Args:
            url: URL halaman detail atau PDF langsung

        Returns:
            DocumentMetadata dengan judul, nomor_peraturan, dan pdf_url
        """
        meta = DocumentMetadata(url=url)

        # Jika langsung PDF → tidak ada halaman HTML untuk di-parse
        if url.lower().endswith(".pdf") or "/storage/" in url.lower():
            meta.pdf_url = url
            # Ekstrak nomor peraturan dari URL/nama file
            meta.nomor_peraturan = _extract_nomor_from_url(url)
            meta.judul = meta.nomor_peraturan
            logger.info(f"URL langsung ke PDF: {url}")
            return meta

        # Parse halaman HTML detail
        try:
            resp = self._get(url)
            soup = BeautifulSoup(resp.text, "html.parser")

            # Kumpulkan semua link download di halaman
            pdf_links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "download" in href.lower() or ".pdf" in href.lower():
                    pdf_links.append({
                        "url": urljoin(self.BASE_URL, href),
                        "label": a.get_text(" ", strip=True),
                    })

            # Pilih dokumen LENGKAP, bukan ABSTRAK/ringkasan:
            # halaman detail JDIH biasanya menampilkan 2 link — abstrak (56 KB)
            # dan dokumen penuh. Abstrak dihindari agar chunk berasal dari
            # teks peraturan yang sesungguhnya.
            full_links = [l for l in pdf_links if not re.search(r"abstrak|ringkasan|summary", l["label"], re.IGNORECASE)]
            chosen = (full_links or pdf_links or [None])[0]

            if chosen:
                meta.pdf_url = chosen["url"]
                meta.judul = chosen["label"] or meta.judul
                # Ekstrak nomor peraturan dari label/nama file dokumen (sumber
                # metadata paling andal — halaman detail JDIH di-render JS).
                meta.nomor_peraturan = _extract_nomor_from_url(chosen["label"] or chosen["url"])
            else:
                meta.pdf_url = url

            if not meta.nomor_peraturan:
                meta.nomor_peraturan = _extract_nomor_from_url(url)

        except Exception as e:
            logger.warning(f"Gagal parse metadata dari {url}: {e}. Menggunakan fallback.")
            meta.nomor_peraturan = _extract_nomor_from_url(url)
            meta.judul = meta.nomor_peraturan
            meta.pdf_url = url

        return meta

    def download_pdf(self, meta: DocumentMetadata) -> Optional[Path]:
        """
        Unduh file PDF dan simpan ke data/raw/.

        Args:
            meta: DocumentMetadata yang berisi pdf_url

        Returns:
            Path ke file PDF yang disimpan, atau None jika gagal
        """
        if not meta.pdf_url:
            logger.error(f"Tidak ada pdf_url untuk: {meta.url}")
            return None

        # Buat nama file yang aman dari nomor peraturan
        safe_name = _safe_filename(meta.nomor_peraturan or meta.pdf_url) + ".pdf"
        dest = DATA_DIR / safe_name

        if dest.exists():
            logger.info(f"File sudah ada, skip download: {safe_name}")
            return dest

        logger.info(f"Mengunduh: {meta.pdf_url}")
        try:
            resp = self._get(meta.pdf_url)

            content_type = resp.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and len(resp.content) < 1024:
                logger.warning(
                    f"Response bukan PDF (Content-Type: {content_type}). Mungkin redirect/error."
                )

            dest.write_bytes(resp.content)
            size_kb = len(resp.content) / 1024
            logger.info(f"Tersimpan: {safe_name} ({size_kb:.1f} KB)")
            return dest

        except Exception as e:
            logger.error(f"Gagal mengunduh {meta.pdf_url}: {e}")
            return None

    def process_url(self, url: str) -> tuple[Optional[Path], DocumentMetadata]:
        """
        Pipeline lengkap: get_metadata + download_pdf untuk satu URL.

        Returns:
            Tuple (path_pdf_atau_none, metadata)
        """
        meta = self.get_metadata(url)
        path = self.download_pdf(meta)
        return path, meta


# ─── Helper functions ─────────────────────────────────────────────────────────

def _safe_filename(text: str) -> str:
    """Mengubah teks menjadi nama file yang aman untuk sistem file."""
    # Normalisasi unicode
    text = unicodedata.normalize("NFKD", text)
    # Hapus karakter non-ASCII
    text = text.encode("ascii", "ignore").decode("ascii")
    # Ganti karakter ilegal dengan underscore
    text = re.sub(r'[\\/*?:"<>|]', "_", text)
    # Ganti spasi dengan underscore
    text = re.sub(r"\s+", "_", text.strip())
    # Truncate jika terlalu panjang
    return text[:100] if text else "dokumen"


def _extract_nomor_from_url(url: str) -> str:
    """
    Ekstrak nomor peraturan dari URL, nama file, atau label dokumen.

    Sumber metadata paling andal di JDIH (halaman detail di-render JS,
    jadi judul/nomor diambil dari nama file PDF dokumen lengkap).
    Contoh hasil:
      'Permen+ESDM+Nomor+36+Tahun+2021+...pdf' -> 'Permen ESDM Nomor 36 Tahun 2021'
      'kepmen-esdm-23-thn-2018'                 -> 'Kepmen ESDM 23 Thn 2018'
    """
    from urllib.parse import unquote

    parsed = urlparse(url)
    text = unquote(Path(parsed.path).name if parsed.path else Path(url).name)
    # Buang ekstensi, normalisasi pemisah
    text = re.sub(r"\.pdf$", "", text, flags=re.IGNORECASE)
    text = text.replace("+", " ").replace("-", " ").replace("_", " ")

    # Pola utama: jenis peraturan + nomor + tahun (lazy, berhenti di tahun pertama)
    m = re.search(
        r"((?:permen|kepmen|kemen|perpres|peraturan|pp|uu)\b[^|]*?\b(?:tahun|thn)\s*\d{4})",
        text,
        re.IGNORECASE,
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip().title()

    # Fallback: pakai teks nama file apa adanya
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned.title() if cleaned else "Dokumen"
