"""
src/api/main.py
FastAPI backend server untuk Themis RAG.

Endpoints:
- POST /api/chat     : Mengirim pertanyaan ke RAG agent (mengembalikan jawaban, sitasi, skor)
- GET  /api/health   : Health check sistem (Milvus, LLM, Langfuse, Alerting)
- POST /api/ingest   : Trigger pipeline ingest dokumen (sync atau background task)
- GET  /api/stats    : Statistik jumlah chunk dan koleksi di Milvus
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Tambahkan root repo ke sys.path jika belum ada
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("themis_api")

app = FastAPI(
    title="Themis RAG API",
    description="API Chatbot RAG untuk Regulasi Pasar Modal Indonesia (POJK OJK & Peraturan Bursa Efek Indonesia)",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend statis (HTML/CSS/JS) — served oleh FastAPI.
FRONTEND_STATIC = ROOT_DIR / "frontend" / "static"
INDEX_HTML = FRONTEND_STATIC / "index.html"
if FRONTEND_STATIC.is_dir():
    # Aset statis (styles.css, app.js, ikon) — mounted sebelum catch-all agar menang.
    app.mount("/static", StaticFiles(directory=FRONTEND_STATIC), name="static")


# ── Global Exception Handler & Alerting ───────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    try:
        from alerting import alert_manager
        alert_manager.send_alert(
            subject=f"[Themis API Error] 500 on {request.url.path}",
            message=f"Request: {request.method} {request.url}\nError: {exc}",
        )
    except Exception as e:
        logger.warning(f"Failed to send alert on unhandled error: {e}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Terjadi kesalahan internal server.", "error": str(exc)},
    )


# ── Request / Response Schemas ───────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="Pertanyaan regulasi")
    top_k: int = Field(default=5, ge=1, le=20, description="Jumlah chunk sitasi yang digunakan")
    provider: str | None = Field(default=None, description="Provider/model LLM (additive; None = default .env)")


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence_passed: bool
    latency_seconds: float


class IngestRequest(BaseModel):
    url: str = Field(..., description="URL halaman detail peraturan atau PDF (JDIH OJK / IDX)")
    run_in_background: bool = Field(default=False, description="Jalankan sebagai background task")


# ── API Endpoints ────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    """Sajikan frontend statis (index.html); fallback JSON bila file tidak ada."""
    if INDEX_HTML.is_file():
        return FileResponse(INDEX_HTML)
    return {
        "app": "Themis RAG API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
    }


@app.get("/api/health")
def health_check():
    """Health check semua komponen pendukung (Milvus, LLM, Langfuse)."""
    from src.reliability.defense import get_system_health
    return get_system_health()


@app.get("/api/stats")
def get_stats():
    """Mengambil statistik jumlah dokumen dan chunk yang tersimpan di Milvus."""
    try:
        from src.vector_store.milvus_client import get_client
        client = get_client()
        count = client.count()
        collection_name = client.collection_name
        return {
            "status": "success",
            "milvus_uri": os.getenv("MILVUS_URI", ""),
            "collection_name": collection_name,
            "total_chunks": count,
        }
    except Exception as e:
        logger.error(f"Gagal mengambil statistik Milvus: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Gagal menghubungi Milvus: {str(e)}",
        )


@app.get("/api/models")
def list_models():
    """Daftar model LLM yang benar-benar tersedia (dari .env), untuk dropdown UI."""
    from llm_config import list_available_models
    return list_available_models()


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Endpoint interaksi tanya-jawab regulasi.
    Memproses query melalui LangGraph Agent (Retrieve → Rerank → Confidence Check → Qwen LLM).
    """
    from src.reliability.defense import validate_query
    from src.query.agent import run_rag_agent

    # 1. Defense / Input Validation
    is_valid, validation_msg = validate_query(req.query)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation_msg,
        )

    start_time = time.time()
    try:
        # 2. Eksekusi Agent (provider opsional; None = default .env)
        result = run_rag_agent(query=req.query, top_k=req.top_k, provider=req.provider)
        latency = round(time.time() - start_time, 3)

        return ChatResponse(
            query=req.query,
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            confidence_passed=result.get("confidence_passed", True),
            latency_seconds=latency,
        )
    except Exception as e:
        logger.error(f"Error pada pemrosesan chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memproses pertanyaan: {str(e)}",
        )


def _background_ingest(url: str):
    from src.ingest.pipeline import ingest_document
    logger.info(f"[BG_INGEST] Memulai ingest di background: {url}")
    res = ingest_document(url)
    if res.success:
        logger.info(f"[BG_INGEST] Sukses ingest {url} ({res.num_chunks} chunks)")
    else:
        logger.error(f"[BG_INGEST] Gagal ingest {url}: {res.error}")


@app.post("/api/ingest")
def ingest_endpoint(req: IngestRequest, bg_tasks: BackgroundTasks):
    """
    Trigger ingest satu URL dokumen regulasi ke Milvus.
    Bisa dijalankan synchronously atau di background via FastAPI BackgroundTasks.
    """
    from src.ingest.pipeline import ingest_document

    if req.run_in_background:
        bg_tasks.add_task(_background_ingest, req.url)
        return {
            "status": "accepted",
            "message": f"Ingest URL {req.url} dijadwalkan di background task.",
        }

    res = ingest_document(req.url)
    if not res.success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Gagal ingest dokumen: {res.error}",
        )

    return {
        "status": "success",
        "url": res.url,
        "nomor_peraturan": res.nomor_peraturan,
        "judul_peraturan": res.judul_peraturan,
        "num_pages": res.num_pages,
        "num_chunks": res.num_chunks,
        "num_inserted": res.num_inserted,
    }


# Catch-all: sajikan frontend untuk rute lain selain /api/* (SPA deep-linking).
@app.get("/{full_path:path}", include_in_schema=False)
def frontend_catch_all(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    if INDEX_HTML.is_file():
        return FileResponse(INDEX_HTML)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Frontend belum tersedia")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
