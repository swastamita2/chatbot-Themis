"""
src/vector_store/milvus_client.py
Klien Milvus untuk Themis RAG (PyMilvus 3.x — API modern MilvusClient).
- Koneksi ke Milvus VPS via MILVUS_URI di .env (db target: MILVUS_DB, default "default")
- Schema koleksi: chunk_id, text, embedding (dim=1024), + 6 metadata fields
- Operasi: insert, similarity_search, delete, count
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# Konstanta skema
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "megan_rag")
EMBEDDING_DIM = 1024          # Dimensi dense vector BGE-M3
MAX_TEXT_LENGTH = 65535       # Batas karakter field text Milvus

# Field names
FIELD_ID = "chunk_id"
FIELD_TEXT = "text"
FIELD_EMBEDDING = "embedding"
FIELD_NOMOR_PERATURAN = "nomor_peraturan"
FIELD_JUDUL_PERATURAN = "judul_peraturan"
FIELD_NOMOR_PASAL = "nomor_pasal"
FIELD_NOMOR_AYAT = "nomor_ayat"
FIELD_URL_SUMBER = "url_sumber"
FIELD_HALAMAN_PDF = "halaman_pdf"

OUTPUT_FIELDS = [
    FIELD_TEXT, FIELD_NOMOR_PERATURAN, FIELD_JUDUL_PERATURAN,
    FIELD_NOMOR_PASAL, FIELD_NOMOR_AYAT, FIELD_URL_SUMBER, FIELD_HALAMAN_PDF,
]


class MilvusClient:
    """
    Client Milvus untuk operasi CRUD dan similarity search.
    Bungkus pymilvus.MilvusClient (SDK modern) agar tidak ada lagi
    PyMilvusDeprecationWarning dari API ORM style.
    """

    def __init__(self):
        self._connected = False
        self._client = None
        self._collection_loaded = False

    def connect(self) -> None:
        """Membuat koneksi ke Milvus dan memastikan koleksi siap & ter-load.

        Idempoten: pemanggilan berulang tidak membuat koneksi ganda
        (koneksi lama sebelumnya bocor karena connect() kedua
        membuat client baru tanpa menutup yang lama).
        """
        if self._connected:
            return
        from pymilvus import MilvusClient as _PymilvusClient

        uri = os.getenv("MILVUS_URI")
        if not uri:
            raise RuntimeError(
                "MILVUS_URI belum diset di .env. "
                "Contoh: http://ip-vps:19530"
            )

        user = os.getenv("MILVUS_USER") or os.getenv("MILVUS_USERNAME") or ""
        password = os.getenv("MILVUS_PASSWORD") or ""
        token = os.getenv("MILVUS_TOKEN") or ""
        db_name = os.getenv("MILVUS_DB", "").strip() or "default"

        kwargs: Dict[str, Any] = {"uri": uri}
        if user:
            kwargs["user"] = user
        if password:
            kwargs["password"] = password
        if token:
            kwargs["token"] = token
        kwargs["db_name"] = db_name

        self._client = _PymilvusClient(**kwargs)
        logger.info(f"Terhubung ke Milvus: {uri} (db: {db_name}, user: {user or 'none'})")
        self._connected = True

        if not self._client.has_collection(COLLECTION_NAME):
            self._create_collection()
        self._client.load_collection(COLLECTION_NAME)
        self._collection_loaded = True

    def _create_collection(self) -> None:
        """Membuat koleksi Milvus dengan schema Themis RAG + index HNSW."""
        from pymilvus import MilvusClient as _PymilvusClient
        from pymilvus.orm.collection import CollectionSchema
        from pymilvus.orm.schema import FieldSchema
        from pymilvus.orm.types import DataType

        fields = [
            FieldSchema(name=FIELD_ID, dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
            FieldSchema(name=FIELD_TEXT, dtype=DataType.VARCHAR, max_length=MAX_TEXT_LENGTH),
            FieldSchema(name=FIELD_EMBEDDING, dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name=FIELD_NOMOR_PERATURAN, dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name=FIELD_JUDUL_PERATURAN, dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name=FIELD_NOMOR_PASAL, dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name=FIELD_NOMOR_AYAT, dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name=FIELD_URL_SUMBER, dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name=FIELD_HALAMAN_PDF, dtype=DataType.INT64),
        ]
        schema = CollectionSchema(
            fields,
            description="Themis RAG — Regulasi Pasar Modal",
            enable_dynamic_field=False,
        )

        index_params = _PymilvusClient.prepare_index_params()
        index_params.add_index(
            field_name=FIELD_EMBEDDING,
            index_type="HNSW",
            metric_type="COSINE",
            M=16,
            efConstruction=200,
        )

        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Koleksi '{COLLECTION_NAME}' dengan index HNSW (COSINE) berhasil dibuat.")

    def _ensure_ready(self) -> None:
        if not self._connected or self._client is None:
            self.connect()
        if not self._collection_loaded:
            self._client.load_collection(COLLECTION_NAME)
            self._collection_loaded = True

    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Insert batch chunk ke Milvus.

        Args:
            chunks: List dict dengan keys: text, embedding, nomor_peraturan,
                    judul_peraturan, nomor_pasal, nomor_ayat, url_sumber, halaman_pdf

        Returns:
            Jumlah chunk yang berhasil diinsert.
        """
        self._ensure_ready()
        if not chunks:
            return 0

        rows = [
            {
                FIELD_ID: str(uuid.uuid4()),
                FIELD_TEXT: c.get("text", "")[:MAX_TEXT_LENGTH],
                FIELD_EMBEDDING: c["embedding"],
                FIELD_NOMOR_PERATURAN: c.get("nomor_peraturan", "")[:256],
                FIELD_JUDUL_PERATURAN: c.get("judul_peraturan", "")[:512],
                FIELD_NOMOR_PASAL: c.get("nomor_pasal", "")[:64],
                FIELD_NOMOR_AYAT: c.get("nomor_ayat", "")[:32],
                FIELD_URL_SUMBER: c.get("url_sumber", "")[:1024],
                FIELD_HALAMAN_PDF: int(c.get("halaman_pdf", 0)),
            }
            for c in chunks
        ]

        self._client.insert(collection_name=COLLECTION_NAME, data=rows)
        count = len(rows)
        logger.info(f"Berhasil insert {count} chunk ke Milvus.")
        return count

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Similarity search (COSINE) terhadap query embedding.

        Args:
            query_embedding: Dense vector hasil embed_query() dari BGE-M3
            top_k: Jumlah hasil teratas
            expr: Filter expression Milvus (e.g., 'nomor_peraturan == "..."')

        Returns:
            List dict berisi text, score, dan semua metadata.
        """
        self._ensure_ready()

        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = self._client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            filter=expr or "",
            limit=top_k,
            output_fields=OUTPUT_FIELDS,
            search_params=search_params,
        )

        hits: List[Dict[str, Any]] = []
        for row in results[0]:
            ent = row.get("entity", {})
            # API modern memakai `distance`; ORM lama memakai `score`.
            score = float(row.get("distance", row.get("score", 0.0)))
            hits.append({
                "chunk_id": row.get("id", ""),
                "text": ent.get(FIELD_TEXT, ""),
                "score": score,
                "nomor_peraturan": ent.get(FIELD_NOMOR_PERATURAN, ""),
                "judul_peraturan": ent.get(FIELD_JUDUL_PERATURAN, ""),
                "nomor_pasal": ent.get(FIELD_NOMOR_PASAL, ""),
                "nomor_ayat": ent.get(FIELD_NOMOR_AYAT, ""),
                "url_sumber": ent.get(FIELD_URL_SUMBER, ""),
                "halaman_pdf": ent.get(FIELD_HALAMAN_PDF, 0),
            })
        return hits

    def query(
        self,
        filter_expr: str = 'nomor_peraturan != ""',
        output_fields: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Kueri chunk berdasarkan filter expression (untuk inspeksi/diagnostik)."""
        self._ensure_ready()
        results = self._client.query(
            collection_name=COLLECTION_NAME,
            filter=filter_expr,
            output_fields=output_fields or OUTPUT_FIELDS,
            limit=limit,
        )
        return results

    def count(self) -> int:
        """Jumlah chunk tersimpan di koleksi (statistik row_count)."""
        self._ensure_ready()
        stats = self._client.get_collection_stats(COLLECTION_NAME)
        return int(stats.get("row_count", 0))

    @property
    def collection_name(self) -> str:
        """Nama koleksi aktif (dipakai frontend & API)."""
        return COLLECTION_NAME

    def delete_by_peraturan(self, nomor_peraturan: str) -> None:
        """Menghapus semua chunk dari satu peraturan (untuk re-ingest)."""
        self._ensure_ready()
        self._client.delete(
            collection_name=COLLECTION_NAME,
            filter=f'{FIELD_NOMOR_PERATURAN} == "{nomor_peraturan}"',
        )
        logger.info(f"Chunk dari '{nomor_peraturan}' dihapus dari Milvus.")

    def disconnect(self) -> None:
        """Menutup koneksi Milvus."""
        try:
            self._client.close()
        except Exception:
            pass
        self._connected = False
        self._collection_loaded = False
        logger.info("Koneksi Milvus ditutup.")


# Singleton instance
_client: Optional[MilvusClient] = None


def get_client() -> MilvusClient:
    """Mengembalikan singleton MilvusClient yang sudah terhubung."""
    global _client
    if _client is None:
        _client = MilvusClient()
        _client.connect()
    return _client


def warmup() -> None:
    """
    Prakoneksi ke Milvus (dipanggil saat startup aplikasi) agar koneksi
    dan load koleksi tidak menambah latensi query pertama.
    """
    get_client()
    logger.info("[WARMUP] Koneksi ke Milvus siap.")
