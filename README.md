# ⚡ Themis RAG

*Chatbot RAG untuk regulasi pasar modal Indonesia (POJK OJK & peraturan Bursa Efek Indonesia) — setiap jawaban menyertakan sitasi pasal dari dokumen aslinya.*

Dulu bernama **Elektra RAG**.

![License](https://img.shields.io/badge/license-MIT-green) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Status](https://img.shields.io/badge/status-aktif-green)

Themis RAG menjawab pertanyaan tentang regulasi pasar modal dengan alur retrieve → rerank → generate. Sistem mencari chunk regulasi yang relevan di Milvus, mererangknya dengan cross-encoder, lalu meminta LLM menjawab **hanya berdasarkan chunk yang ditemukan**, dengan sitasi `[Nomor Peraturan, Pasal, Ayat]` yang wajib tercantum di setiap jawaban. Proyek ini dibangun sebagai replika skala kecil arsitektur RAG produksi yang dipelajari selama magang, sekaligus portofolio pribadi.

---

## 📑 Daftar Isi

- [✨ Key Features](#-key-features)
- [🏛️ Architecture & Tech Stack](#-architecture--tech-stack)
- [📁 Project Structure](#-project-structure)
- [📦 Prerequisites](#-prerequisites)
- [🚀 Installation](#-installation)
- [⚙️ Configuration / Environment Variables](#️-configuration--environment-variables)
- [💻 Usage Guide](#-usage-guide)
- [🛠️ Troubleshooting / FAQ](#️-troubleshooting--faq)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Key Features

- **Sitasi berbasis pasal.** Prompt secara tegas mewajibkan LLM menyertakan sitasi `[Nomor Peraturan, Pasal X, Ayat Y]` yang diambil dari metadata asli potongan dokumen — bukan dari ingatan model.
- **Retrieval dua tahap.** Pencarian dense BGE-M3 (top-20 kandidat) dilanjutkan re-ranking cross-encoder BGE-Reranker-v2-M3 (top-5), sehingga konteks yang dikirim ke LLM lebih tepat dan hemat token.
- **Confidence gate.** Jika skor reranker tertinggi di bawah ambang (0.45), agen menolak menjawab dan memberi tahu bahwa rujukan belum mencukupi. Mencegah jawaban percaya diri yang ternyata halusinasi.
- **LLM multi-provider.** Ganti model cukup dengan satu variabel di `.env` (Qwen sebagai default, plus Gemini, OpenRouter, dan Ollama). UI hanya menampilkan model yang benar-benar terkonfigurasi, bukan daftar statis.
- **Ingest idempoten.** Re-ingest peraturan yang sama menghapus chunk lamanya terlebih dahulu (kunci: `nomor_peraturan`), jadi tidak ada duplikasi di Milvus.
- **Dua jalur ingest.** PDF lokal (rekomendasi untuk korpus pasar modal) maupun scraping langsung dari JDIH, dengan parser PDF dan fallback OCR untuk dokumen scan.
- **Alerting multi-channel.** Telegram, Discord, Resend, Brevo, Google Apps Script, SMTP, dan fallback console. Jika satu kanal sedang bermasalah, query tetap diproses normal tanpa terganggu.
- **Observability.** Tracing Langfuse di setiap tahap pipeline query, plus health check API yang memeriksa Milvus, LLM, dan Langfuse.
- **Satu server, dua antarmuka.** FastAPI menyajikan UI web statis (dark theme) di `/` sekaligus REST API lengkap dengan Swagger di `/docs`. Ada juga UI Streamlit lama yang masih bisa dijalankan.

---

## 🏛️ Architecture & Tech Stack

```
                    [ Pengguna / Web UI ]
                             │
                             ▼
                 [ FastAPI Backend / Streamlit ]
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
     [ Ingest Pipeline ]            [ Query Pipeline ]
             │                               │
      JDIH Scraper / PDF Lokal      LangGraph Agent
             │                               │
      PDF Parser (+ OCR fallback)   BGE-M3 Query Embed
             │                               │
      Pasal-based Chunker           Milvus Semantic Search
             │                               │
       BGE-M3 Embedder              BGE-Reranker-v2-M3
             │                               │
             ▼                               ▼
   [ Milvus Vector Store ]      Confidence Gate (threshold 0.45)
   (HNSW, dim=1024)                        │
                                           ▼
                                LLM: Qwen/Qwen3.8-27B
                                (via LiteLLM Proxy, default)
                                           │
                                           ▼
                                  Jawaban + Sitasi Pasal
```

Alur query di agen LangGraph: `retrieve → rerank → confidence_check → generate`, dengan cabang `no_answer` jika confidence rendah.

| Komponen | Teknologi | Catatan |
|---|---|---|
| LLM utama | `Qwen/Qwen3.8-27B` via LiteLLM Proxy | Alternatif: Gemini, OpenRouter, Ollama |
| Embedding | `BAAI/bge-m3` | Dense 1024-dim, jalan in-process |
| Reranker | `BAAI/bge-reranker-v2-m3` | Cross-encoder, top-20 → top-5 |
| Vector DB | Milvus (SDK PyMilvus) | Index HNSW, koleksi `megan_rag` |
| Orkestrasi agen | LangGraph + LangChain | Workflow 4 node + cabang no_answer |
| API | FastAPI + Uvicorn | REST + Swagger + CORS |
| Frontend | HTML/CSS/JS statis (dark theme) | Di-serve FastAPI; UI Streamlit sebagai legacy |
| Observability | Langfuse | Tracing per tahap pipeline |
| Alerting | `alerting.py` (kustom) | 7 kanal notifikasi + fallback console |
| Parser dokumen | pdfplumber, pypdf, Tesseract | OCR fallback untuk PDF hasil scan |
---

## 📁 Project Structure

```
benchmark-agent-mcp/
├── .env                     # Variabel environment & API keys (JANGAN di-commit)
├── .env.example             # Template konfigurasi (salin ke .env)
├── PRD.md                   # Dokumen Product Requirements
├── requirements.txt         # Dependensi Python
├── README.md                # Dokumentasi proyek
├── alerting.py              # Sistem notifikasi & alerting multi-channel
├── llm_config.py            # Konfigurasi LLM multi-provider (registry builder)
│
├── src/
│   ├── warmup.py            # Prakisi model embedding/reranker + Milvus saat startup
│   ├── embeddings/
│   │   └── bge_m3.py        # Wrapper BGE-M3 (singleton, dim=1024)
│   ├── vector_store/
│   │   └── milvus_client.py # Client Milvus: schema, HNSW index, CRUD, search
│   ├── ingest/
│   │   ├── scraper.py       # Scraper JDIH (rate-limited)
│   │   ├── parser.py        # Ekstraksi teks PDF + fallback OCR Tesseract
│   │   ├── chunker.py       # Chunker berbasis struktur Pasal & Ayat
│   │   ├── embed_store.py   # Embed + insert ke Milvus (dipakai kedua jalur ingest)
│   │   └── pipeline.py      # Orkestrator ingest end-to-end
│   ├── query/
│   │   ├── prompts.py       # Prompt template + konstanta (threshold, top_k)
│   │   ├── retriever.py     # Milvus similarity search
│   │   ├── reranker.py      # Cross-encoder re-ranking
│   │   └── agent.py         # Workflow agen LangGraph (retrieve→rerank→gate→generate)
│   ├── reliability/
│   │   ├── defense.py       # Validasi input & health check sistem
│   │   └── observability.py # Integrasi callback & tracing Langfuse
│   └── api/
│       └── main.py          # Backend FastAPI (REST + serve frontend statis)
│
├── frontend/
│   ├── app.py               # Web app Streamlit (legacy)
│   ├── .streamlit/          # Konfigurasi Streamlit
│   └── static/              # UI web statis (dark theme), di-serve FastAPI
│       ├── index.html
│       ├── styles.css
│       └── app.js
│
├── scripts/
│   ├── run_ingest.py        # CLI: ingest via URL JDIH (satu URL atau file daftar)
│   ├── ingest_local_pdf.py  # CLI: ingest file PDF lokal (korpus pasar modal)
│   ├── e2e_test.py          # Uji end-to-end: warmup + beberapa pertanyaan
│   ├── diagnose_query.py    # Diagnostik: inspect koleksi + ukur latensi tiap tahap
│   └── check_counts.py      # Diagnostik: jumlah chunk per peraturan di Milvus
│
└── data/
    ├── raw/                 # Penyimpanan PDF hasil unduhan
    └── urls.txt             # Daftar URL regulasi (satu per baris)
```

---

## 📦 Prerequisites

| Ketergantungan | Versi | Keterangan |
|---|---|---|
| Python | 3.10+ | Proyek dikembangkan dengan 3.14 |
| Git | terbaru | Untuk clone repo |
| Tesseract OCR | opsional | Hanya jika ada PDF hasil scan |
| Poppler | opsional | Dibutuhkan `pdf2image` untuk jalur OCR |

Model embedding (BGE-M3) dan reranker (BGE-Reranker-v2-M3) diunduh otomatis oleh `sentence-transformers` saat pertama kali dijalankan, lalu di-cache di `~/.cache/huggingface`. Setelah itu bisa dijalankan offline.

---

## 🚀 Installation

1. **Clone repo**
   ```bash
   git clone https://github.com/swastamita2/chatbot-Themis.git
   cd chatbot-Themis
   ```

2. **Buat & aktifkan virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Pasang dependensi**
   ```bash
   pip install -r requirements.txt
   ```

4. **Siapkan konfigurasi** (lihat bagian Configuration di bawah)
   ```bash
   cp .env.example .env   # Windows: copy .env.example .env
   ```

> [!TIP]
> Jalankan `python -c "import torch; print(torch.__version__)"` untuk memastikan PyTorch terpasang benar. Instalasi CPU sudah cukup; GPU opsional untuk kecepatan.

---

## ⚙️ Configuration / Environment Variables

Semua konfigurasi dibaca dari file `.env` di root proyek (di-load oleh `python-dotenv`). Template lengkap ada di [`.env.example`](.env.example). Salin lalu isi nilainya.

**Wajib (untuk menjalankan query):**

| Variabel | Contoh | Fungsi |
|---|---|---|
| `LLM_PROVIDER` | `qwen` | Provider LLM aktif: `qwen`/`gemini`/`openrouter`/`ollama` |
| `QWEN_BASE_URL` | `https://aicon.air.id/` | Endpoint LiteLLM Proxy (untuk provider `qwen`) |
| `QWEN_API_KEY` | `***` | Kunci API LLM |
| `QWEN_MODEL` | `Qwen/Qwen3.8-27B` | Nama model LLM |
| `MILVUS_URI` | `http://IP:19530` | Alamat server Milvus |
| `MILVUS_DB` | `megan_rag` | Database Milvus tujuan |
| `MILVUS_COLLECTION` | `megan_rag` | Nama koleksi |

> [!IMPORTANT]
> `MILVUS_USER` / `MILVUS_PASSWORD` / `MILVUS_TOKEN` hanya diisi jika server Milvus Anda memakai autentikasi.

**Opsional (observability & alerting):** isi `LANGFUSE_*` untuk tracing, dan minimal satu kanal di kelompok `TELEGRAM_*`, `DISCORD_WEBHOOK_URL`, `RESEND_*`, `BREVO_*`, `GMAIL_WEBHOOK_URL`, atau `EMAIL_*` untuk notifikasi. Jika semuanya kosong, alert hanya dicetak ke console dan sistem tetap berjalan normal.
---

## 💻 Usage Guide

### 1. Ingest dokumen ke Milvus

Dua jalur tersedia. Keduanya idempoten per `nomor_peraturan`.

**Jalur A — PDF lokal** (rekomendasi untuk korpus pasar modal):
```bash
# Ingest seluruh korpus yang sudah terdaftar di skrip
python scripts/ingest_local_pdf.py --all

# Ingest satu PDF dengan metadata eksplisit
python scripts/ingest_local_pdf.py --file "corpus/POJK-33-2024.pdf" \
    --nomor "POJK No. 33 Tahun 2024" \
    --judul "Pengelolaan Investasi di Pasar Modal"
```

**Jalur B — scraping JDIH** (untuk regulasi ketenagalistrikan / dokumen yang belum diunduh):
```bash
# Satu URL
python scripts/run_ingest.py --url "https://jdih.esdm.go.id/dokumen/view?id=2428"

# Daftar URL dari file
python scripts/run_ingest.py --file data/urls.txt --batch-size 32
```

> [!NOTE]
> Ingest bersifat idempoten: chunk lama milik peraturan yang sama dihapus dulu sebelum insert. Tambahkan `--keep-stale` jika memang ingin menyimpan versi lama.

### 2. Menjalankan Web UI

**UI statis (default, dark theme)** — cukup jalankan backend FastAPI:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Lalu buka `http://localhost:8000`.

**UI Streamlit (legacy):**
```bash
streamlit run frontend/app.py
```
Buka `http://localhost:8501`.

### 3. API Endpoints (REST)

Dokumentasi interaktif: `http://localhost:8000/docs`.

| Method | Path | Fungsi |
|---|---|---|
| `POST` | `/api/chat` | Kirim pertanyaan, dapat jawaban + sitasi + skor |
| `GET` | `/api/health` | Health check (Milvus, LLM, Langfuse, Alerting) |
| `GET` | `/api/stats` | Jumlah chunk & koleksi di Milvus |
| `GET` | `/api/models` | Daftar model LLM yang terkonfigurasi |
| `POST` | `/api/ingest` | Trigger ingest satu URL (sync/background) |

**Contoh `POST /api/chat`:**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apa syarat agar perusahaan efek dapat menjadi penjamin emisi efek?",
    "top_k": 5
  }'
```

Bentuk responsnya (nilai di bawah ini contoh ilustratif, bukan data asli):
```json
{
  "query": "Apa syarat agar perusahaan efek dapat menjadi penjamin emisi efek?",
  "answer": "Berdasarkan POJK No. 3 Tahun 2026, Pasal 23, ayat (1), syaratnya adalah …",
  "sources": [
    {
      "nomor_peraturan": "POJK No. 3 Tahun 2026",
      "nomor_pasal": "23",
      "nomor_ayat": "1",
      "score": 0.87
    }
  ],
  "confidence_passed": true,
  "latency_seconds": 12.4
}
```

### 4. Skrip diagnostik
```bash
python scripts/check_counts.py        # Jumlah chunk per peraturan
python scripts/diagnose_query.py --query "..."   # Ukur latensi tiap tahap
python scripts/e2e_test.py            # Uji end-to-end beberapa pertanyaan
```

---

## 🛠️ Troubleshooting / FAQ

**Query sangat lambat atau pertama kali terasa "hang".**
Model BGE-M3 dan reranker butuh waktu ~40–45 detik untuk di-load pertama kali. Skrip `e2e_test.py` dan warmup startup sudah memanggil `warmup_all()`. Untuk proses terpisah, biarkan warmup selesai sebelum mengukur performa.

**`ModuleNotFoundError: No module named 'llm_config'` saat menjalankan skrip.**
Jalankan skrip dari root proyek (`python scripts/...`), bukan dari dalam folder `scripts/`. Skrip sudah men-`insert` root ke `sys.path`, tetapi itu hanya berlaku jika dipanggil dengan path relatif dari root.

**Koneksi ke Milvus gagal / timeout.**
Cek `MILVUS_URI` di `.env`, pastikan VPS Milvus hidup dan port 19530 terbuka. Tes dengan `python scripts/check_counts.py` — jika berhasil menampilkan jumlah chunk, koneksi sehat.

**Jawaban kosong padahal `confidence_passed: true`.**
Ini umumnya terjadi pada provider Qwen karena fase "thinking" model mengisi kuota token sehingga konten final kosong. `llm_config.py` sudah mengatasi ini dengan `enable_thinking: false` dan `max_tokens=8192`. Jika masih terjadi, coba provider lain di `.env`.

**Langfuse / email notifier error tapi query tetap jalan.**
Sengaja dibuat demikian: kegagalan kanal pihak ketiga tidak memotong alur query utama, jadi pengguna akhir tidak merasakan gangguan.

**Ada dua instance BGE-M3 lalu proses CPU-nya deadlock.**
Jangan muat dua instance model BGE-M3 dalam satu proses. Jika server uvicorn sedang jalan dan Anda mau menjalankan harness/skrip terpisah, hentikan servernya dulu.

**PDF hasil scan tidak ter-ekstrak (teks kosong).**
Pasang Tesseract OCR dan set `TESSERACT_CMD` di `.env` bila binary tidak ada di `PATH`. Jalur OCR memakai `pdf2image`, yang membutuhkan Poppler.

---

## 🤝 Contributing

Repo ini masih bersifat pribadi/portofolio dan belum diinisialisasi sebagai git repository. Alur kontribusi yang direncanakan:

1. Clone repo dan buat branch baru: `git checkout -b fitur/nama-perubahan`.
2. Jalankan `python scripts/e2e_test.py` untuk memastikan tidak ada regresi sebelum mengirim.
3. Buka *pull request* dengan deskripsi singkat tentang apa yang berubah dan mengapa.

Kontribusi yang dicari: penambahan kanal alerting, dukungan model LLM baru (tambah satu builder di `llm_config.py`), perbaikan parser untuk format PDF yang belum tertangani, dan penguatan pengujian.

---

## 📄 License

Lisensi: [MIT](LICENSE).

> [!WARNING]
> File `.env` tidak disertakan di repo (sudah di-gitignore), karena isinya kredensial aktif: API key LLM, password Milvus, token bot, dan sebagainya. Jangan pernah commit, attach, atau membagikan file `.env` ke mana pun. Salin `.env.example` ke `.env` lalu isi nilainya sendiri.

Proyek ini dikembangkan sebagai portofolio pribadi dan independen dari PT PLN Icon Plus.
