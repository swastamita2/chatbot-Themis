"""
frontend/app.py
Streamlit Web UI untuk Themis RAG: Chatbot Regulasi Pasar Modal Indonesia.

Fitur:
- Antarmuka chat interaktif dengan riwayat percakapan.
- Panel sidebar untuk status sistem (Milvus, LLM) & tool ingest dokumen.
- Penampil sitasi/sumber pasal secara transparan (Grounding & Anti-halusinasi).

Desain: design system monokrom (gaya Vercel) — ink-on-canvas dengan satu aksen,
tipografi Geist via Google Fonts, status sebagai chip, sitasi sebagai panel
datar tanpa ornamen. Token warna & font didefinisikan di .streamlit/config.toml.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# Tambahkan root path ke sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

# ── Konfigurasi Halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Themis RAG — Regulasi Pasar Modal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Font: Geist (substitusi Google Fonts untuk Vercel Geist Sans / Mono).
# Fallback ke font theme config.toml bila CDN tidak terjangkau.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap');
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Token & komponen CSS (gaya Vercel: monokrom + satu aksen) ─────────────────
# Token warna mengikuti .streamlit/config.toml (base=light, primary=ink #0A0A0A).
# Kelas di bawah hanya menangani struktur komponen khusus chat.
st.markdown(
    """
    <style>
    /* Header chat: menggantikan judul marketing — wordmark + deskripsi rapat */
    .app-wordmark {
        font-family: 'Geist', ui-sans-serif, system-ui, -apple-system, sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.022em;
        line-height: 1.2;
        color: var(--text-color, #0a0a0a);
    }
    .app-tagline {
        font-family: 'Geist', ui-sans-serif, system-ui, sans-serif;
        font-size: 0.95rem;
        line-height: 1.5;
        color: var(--secondary-color, #666666);
        max-width: 60ch;
        margin-top: 0.35rem;
    }
    .app-header-rule {
        border: none;
        border-top: 1px solid var(--border-color, #e5e5e5);
        margin: 1.1rem 0 1.6rem;
    }

    /* Chip status: pengganti kotak st.success/st.info besar di sidebar */
    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-family: 'Geist', ui-sans-serif, system-ui, sans-serif;
        font-size: 0.82rem;
        font-weight: 500;
        padding: 0.3rem 0.65rem;
        border: 1px solid var(--border-color, #e5e5e5);
        border-radius: 999px;
        color: var(--secondary-color, #666666);
        margin: 0 0.4rem 0.4rem 0;
        background: transparent;
    }
    .status-chip .dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #9ca3af;
        flex-shrink: 0;
    }
    .status-chip.ok { color: var(--text-color, #0a0a0a); }
    .status-chip.ok .dot { background: #16a34a; }
    .status-chip.err { color: #b91c1c; border-color: #fecaca; }
    .status-chip.err .dot { background: #dc2626; }
    .status-chip code {
        font-family: 'Geist Mono', ui-monospace, monospace;
        font-size: 0.78rem;
        color: inherit;
    }

    /* Badge kepercayaan jawaban: chip datar, tanpa emoji wajib */
    .conf-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-family: 'Geist', ui-sans-serif, system-ui, sans-serif;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        margin-bottom: 0.55rem;
        border: 1px solid;
    }
    .conf-badge.ok {
        color: #15803d;
        background: #f0fdf4;
        border-color: #bbf7d0;
    }
    .conf-badge.low {
        color: #b45309;
        background: #fffbeb;
        border-color: #fde68a;
    }

    /* Sitasi sumber: panel datar — tanpa "accent rail" border-kiri tebal */
    .citation-box {
        border: 1px solid var(--border-color, #e5e5e5);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin-bottom: 0.5rem;
        background: rgba(0, 0, 0, 0.015);
    }
    .citation-box .cite-head {
        font-weight: 600;
        font-size: 0.85rem;
        color: var(--text-color, #0a0a0a);
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .citation-box .cite-meta {
        font-family: 'Geist Mono', ui-monospace, monospace;
        font-size: 0.75rem;
        font-weight: 400;
        color: var(--secondary-color, #666666);
        white-space: nowrap;
    }
    .citation-box .cite-snippet {
        margin-top: 0.5rem;
        color: var(--secondary-color, #4b5563);
        line-height: 1.55;
        font-size: 0.83rem;
    }

    /* Tombol pertanyaan cepat: chip netral, radius konsisten */
    div[data-testid="stButton"] > button {
        font-family: 'Geist', ui-sans-serif, system-ui, sans-serif !important;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Warmup komponen berat (1x per proses, bukan per query) ───────────────────
# Model BGE-M3 & BGE-Reranker dimuat sekali saat startup. Tanpa ini, query
# pertama menanggung ~90 dtk biaya loading model.
if "warmup_done" not in st.session_state:
    with st.spinner(
        "Memuat model BGE-M3, BGE-Reranker & koneksi Milvus "
        "(hanya terjadi sekali per sesi)..."
    ):
        try:
            from src.warmup import warmup_all
            st.session_state.warmup_timings = warmup_all()
        except Exception as e:
            st.session_state.warmup_timings = {}
            st.session_state.warmup_error = str(e)
    st.session_state.warmup_done = True


# ── Helper: agen & data ───────────────────────────────────────────────────────
def query_agent(prompt: str, top_k: int = 5) -> Dict[str, Any]:
    """Panggil LangGraph agent RAG secara langsung."""
    try:
        from src.query.agent import run_rag_agent
        from src.reliability.defense import validate_query

        is_valid, msg = validate_query(prompt)
        if not is_valid:
            return {
                "answer": f"Query tidak valid: {msg}",
                "sources": [],
                "confidence_passed": False,
            }
        return run_rag_agent(query=prompt, top_k=top_k)
    except Exception as e:  # surface error apa adanya ke UI
        return {
            "answer": f"Terjadi kesalahan saat memproses jawaban: {e}",
            "sources": [],
            "confidence_passed": False,
        }


def get_stats_data() -> Dict[str, Any]:
    """Ambil data statistik koleksi Milvus."""
    try:
        from src.vector_store.milvus_client import get_client
        client = get_client()
        return {
            "status": "ok",
            "total_chunks": client.count(),
            "collection": client.collection_name,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "total_chunks": 0}


def render_citations(sources: List[Dict[str, Any]]) -> None:
    """Render blok sitasi sumber (dipakai riwayat & jawaban baru — 1 implementasi)."""
    if not sources:
        return
    with st.expander(f"Rujukan & Sitasi Pasal ({len(sources)})"):
        for i, src in enumerate(sources, 1):
            pasal = src.get("nomor_pasal", "—")
            ayat = src.get("nomor_ayat")
            peraturan = src.get("nomor_peraturan", "Peraturan")
            skor = src.get("score", 0.0)
            hal = src.get("halaman_pdf")
            text = src.get("text", "")

            meta_parts = [f"Pasal {pasal}"]
            if ayat:
                meta_parts[0] = f"Pasal {pasal} ayat ({ayat})"
            if hal not in (None, ""):
                meta_parts.append(f"hal. {hal}")
            meta_parts.append(f"skor {skor:.3f}")

            snippet = text[:350] + ("…" if len(text) > 350 else "")
            st.markdown(
                f"""
                <div class="citation-box">
                    <div class="cite-head">
                        <span>[{i}] {peraturan}</span>
                        <span class="cite-meta">{' · '.join(meta_parts)}</span>
                    </div>
                    <div class="cite-snippet">“{snippet}”</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_confidence_badge(confidence_passed: bool, detail: str | None = None) -> None:
    """Render chip kepercayaan jawaban (dipakai riwayat & jawaban baru)."""
    if confidence_passed:
        label = "Terdokumentasi" + (f" · {detail}" if detail else "")
        st.markdown(f'<span class="conf-badge ok">{label}</span>', unsafe_allow_html=True)
    else:
        label = "Confidence rendah — data terbatas" + (f" · {detail}" if detail else "")
        st.markdown(f'<span class="conf-badge low">{label}</span>', unsafe_allow_html=True)


def run_ingest(url: str) -> None:
    """Jalankan pipeline ingest dari sidebar; tampilkan hasilnya di sana juga."""
    with st.spinner("Mengunduh, mem-parsing, chunking, dan embedding dokumen..."):
        try:
            from src.ingest.pipeline import ingest_document
            res = ingest_document(url)
            if res.success:
                st.success(f"Berhasil di-ingest — {res.nomor_peraturan} · {res.num_chunks} chunks")
                st.rerun()
            else:
                st.error(f"Gagal ingest: {res.error}")
        except Exception as ex:
            st.error(f"Terjadi error: {ex}")


# ── Sidebar: status sistem + tool ingest ─────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom: 0.2rem;">
            <span style="font-family: 'Geist', sans-serif; font-size: 1.15rem; font-weight: 700;
                         letter-spacing: -0.02em; color: var(--text-color, #0a0a0a);">⚖ Themis RAG</span>
            <div style="font-size: 0.8rem; color: var(--secondary-color, #666666); margin-top: 0.15rem;">
                Tanya-jawab regulasi pasar modal
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("#### Status sistem")
    stats = get_stats_data()
    if stats.get("status") == "ok":
        st.markdown(
            f"""
            <div class="status-chip ok"><span class="dot"></span>Milvus online · <code>{stats['total_chunks']:,} chunks</code></div>
            <div class="status-chip"><span class="dot"></span>Koleksi: <code>{stats['collection']}</code></div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-chip err"><span class="dot"></span>Milvus tidak terhubung</div>',
            unsafe_allow_html=True,
        )
        st.caption(stats.get("error", ""))

    model_name = os.getenv("QWEN_MODEL", "Qwen/Qwen3.8-27B")
    st.markdown(
        f'<div class="status-chip"><span class="dot"></span>LLM: <code>{model_name}</code></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("#### Ingest regulasi baru")
    url_input = st.text_input(
        "URL JDIH OJK / IDX",
        placeholder="https://www.ojk.go.id/...  atau  https://www.idx.co.id/...",
        help="Masukkan URL halaman detail peraturan atau URL PDF langsung.",
        label_visibility="collapsed",
    )
    if st.button("Mulai Ingest Dokumen", width="stretch"):
        if not url_input.strip():
            st.warning("Masukkan URL dokumen terlebih dahulu.")
        else:
            run_ingest(url_input.strip())

    st.markdown("---")
    st.caption("Riset & portofolio RAG — Regulasi Pasar Modal (Themis RAG)")


# ── Main content: header chat ────────────────────────────────────────────────
st.markdown(
    """
    <div class="app-wordmark">Themis RAG Assistant</div>
    <div class="app-tagline">
        Tanyakan POJK OJK, peraturan Bursa Efek Indonesia (IDX), penjaminan emisi,
        perdagangan efek, dan keanggotaan bursa — setiap jawaban disertai
        rujukan pasal dari dokumen resmi.
    </div>
    <hr class="app-header-rule"/>
    """,
    unsafe_allow_html=True,
)

# Inisialisasi state percakapan
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Halo — saya Themis, asisten regulasi pasar modal Indonesia. Saya menjawab pertanyaan "
                "(POJK OJK & Peraturan Bursa Efek Indonesia) "
                "dengan rujukan pasal dari dokumen resmi. "
                "Apa yang ingin Anda tanyakan?"
            ),
            "sources": [],
            "confidence_passed": True,
        }
    ]

# Contoh pertanyaan (hanya saat chat masih kosong)
if len(st.session_state.messages) <= 1:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Syarat menjadi penjamin emisi efek"):
            st.session_state.preset_prompt = "Apa syarat agar perusahaan efek dapat menjadi penjamin emisi efek?"
            st.rerun()
    with c2:
        if st.button("Ketentuan perdagangan efek di bursa"):
            st.session_state.preset_prompt = "Bagaimana ketentuan perdagangan efek pada bursa efek?"
            st.rerun()
    st.write("")

# Render riwayat chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_confidence_badge(msg.get("confidence_passed", True))
        st.markdown(msg["content"])
        render_citations(msg.get("sources", []))

# Input chat: preset prompt (dari tombol contoh) atau ketik manual
prompt_to_send = st.session_state.pop("preset_prompt", None) or st.chat_input(
    "Ketik pertanyaan Anda tentang regulasi pasar modal..."
)

if prompt_to_send:
    st.session_state.messages.append({"role": "user", "content": prompt_to_send})
    with st.chat_message("user"):
        st.markdown(prompt_to_send)

    with st.chat_message("assistant"):
        with st.spinner("Mencari pasal relevan (BGE-M3 + reranker)..."):
            start_t = time.time()
            res = query_agent(prompt_to_send)
            latency = round(time.time() - start_t, 2)

        answer = res.get("answer", "Maaf, tidak ada respon.")
        sources = res.get("sources", [])
        conf = res.get("confidence_passed", True)

        render_confidence_badge(conf, detail=f"{latency}s")
        st.markdown(answer)
        render_citations(sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "confidence_passed": conf,
            }
        )
