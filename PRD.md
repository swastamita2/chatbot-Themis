# PRD: Themis RAG
## Chatbot RAG untuk Regulasi & Standar Ketenagalistrikan Indonesia

| | |
|---|---|
| **Status** | Draft |
| **Pemilik Proyek** | Sapta |
| **Jenis Proyek** | Proyek pribadi / portofolio (independen dari PT PLN Icon Plus) |
| **Versi Dokumen** | 1.0 |

---

## 1. Ringkasan Eksekutif

Elektra RAG (kini Themis RAG) adalah chatbot berbasis **RAG (Retrieval-Augmented Generation)** yang menjawab
pertanyaan seputar regulasi dan standar ketenagalistrikan Indonesia, dengan jawaban yang
**selalu merujuk ke pasal dan nomor peraturan asli** — bukan sekadar mengarang dari ingatan
model. Sumber data diambil dari dokumen resmi yang dipublikasikan oleh JDIH Kementerian ESDM.

Proyek ini dibangun sebagai **replikasi skala-kecil** dari arsitektur RAG produksi (Elarch)
yang dipelajari selama magang, sekaligus jadi portofolio pribadi yang relevan dengan bidang
kerja (sub-bidang Standarisasi, squad riset AI).

---

## 2. Latar Belakang & Masalah

Regulasi ketenagalistrikan Indonesia tersebar di banyak dokumen (UU, PP, Permen ESDM) yang
saling mengubah dan merujuk satu sama lain. Contoh nyata: Permen ESDM No. 10 Tahun 2016
mengubah sebagian isi Permen ESDM No. 05 Tahun 2014 — orang yang mencari satu ketentuan
harus tahu riwayat perubahan ini secara manual. Dokumen juga bervariasi bentuknya: sebagian
sudah digital, sebagian lain (terutama dokumen lama) hanya tersedia sebagai hasil pindai/scan.

**Masalah yang ingin diselesaikan**: sulit dan lambat mencari jawaban atas pertanyaan spesifik
soal regulasi ketenagalistrikan tanpa membaca banyak dokumen panjang secara manual.

---

## 3. Tujuan

### Tujuan Produk
- Menjawab pertanyaan seputar regulasi ketenagalistrikan Indonesia secara akurat, dengan
  sitasi pasal/nomor peraturan yang jelas.
- Mendukung dokumen campuran (PDF digital dan hasil scan/OCR).

### Tujuan Pembelajaran (tidak kalah penting untuk proyek ini)
- Mempraktikkan arsitektur RAG produksi (ingest pipeline, query pipeline, reranking) di luar
  konteks lab/eksperimen kecil.
- Melatih *clean architecture* dan kebiasaan observability/alerting yang sudah dibangun
  selama magang, diterapkan ke proyek nyata.

---

## 4. Target Pengguna

| Persona | Kebutuhan |
|---|---|
| Diri sendiri (utama) | Alat bantu belajar & referensi cepat regulasi ketenagalistrikan |
| Rekan magang/kolega di bidang energi | Referensi cepat tanpa harus baca dokumen penuh |
| Perekrut/pembimbing (tidak langsung) | Bukti kemampuan membangun sistem RAG end-to-end |

---

## 5. Ruang Lingkup

### In Scope (V1)
- Ingest dokumen dari `jdih.esdm.go.id` (mulai dari subset kecil: 15-20 dokumen)
- OCR untuk dokumen hasil scan
- Chunking berbasis struktur Pasal
- Pencarian semantik (Milvus) + reranking
- Jawaban dari LLM dengan sitasi sumber
- Observability (Langfuse) dan alert gangguan sistem (Telegram/Email)
- Frontend sederhana (Streamlit) untuk demo

### Out of Scope (V1)
- Update otomatis/berkala saat ada regulasi baru terbit (ingest manual dulu)
- Multi-user dengan autentikasi/akun
- Dukungan dokumen di luar ranah ketenagalistrikan
- Nasihat hukum yang mengikat (produk ini alat bantu referensi, bukan pengganti konsultasi hukum resmi)

---

## 6. User Stories

- Sebagai pengguna, saya ingin bertanya dalam bahasa natural ("apa syarat akreditasi
  ketenagalistrikan?") dan mendapat jawaban yang menyebutkan pasal/peraturan sumbernya.
- Sebagai pengguna, saya ingin tahu kalau chatbot **tidak yakin** atau tidak menemukan
  jawaban di dokumen yang tersedia — bukan mengarang jawaban (mencegah halusinasi, pelajaran
  dari kasus Air Canada yang pernah dibahas).
- Sebagai pemilik sistem, saya ingin mendapat notifikasi otomatis kalau ingest pipeline gagal
  atau kuota API model habis, tanpa harus cek manual.

---

## 7. Requirements Fungsional

| ID | Requirement |
|---|---|
| F1 | Sistem dapat mengunduh dan menyimpan dokumen PDF dari sumber JDIH ESDM |
| F2 | Sistem dapat mengekstrak teks dari PDF digital maupun hasil scan (OCR) |
| F3 | Sistem memecah dokumen menjadi potongan (chunk) berbasis struktur Pasal; spesifikasi detail lihat *Catatan Implementasi: Strategi Chunking* di bawah |
| F4 | Sistem menyimpan embedding tiap chunk ke Milvus, beserta metadata (nomor peraturan, nomor pasal, URL sumber) |
| F5 | Sistem dapat mencari chunk paling relevan terhadap pertanyaan pengguna (similarity search + reranking) |
| F6 | Sistem menghasilkan jawaban yang menyertakan sitasi sumber (nomor peraturan & pasal) |
| F7 | Sistem menolak menjawab atau menyatakan tidak yakin bila tidak ada chunk relevan yang cukup mirip |
| F8 | Sistem mencatat setiap request (prompt, token, latency) ke Langfuse |
| F9 | Sistem mengirim alert ke Telegram/Email saat terjadi gangguan (error koneksi, token/kredit API habis, ingest pipeline gagal) |

### Catatan Implementasi: Strategi Chunking (F3)

| Parameter | Nilai / Pendekatan |
|---|---|
| Unit chunk utama | Per Pasal (dideteksi via regex pola `Pasal [IVXLC0-9]+` atau `Pasal \d+`) |
| Granularitas sub-chunk | Per ayat `(1)`, `(2)`, dst. jika satu Pasal melebihi batas token |
| Ukuran maksimum | ~512 token per chunk |
| Overlap antar-chunk | ~50 token (menjaga konteks di batas antar-Pasal) |
| Fallback | Fixed-size chunking (512 token, overlap 50) jika struktur Pasal tidak terdeteksi (e.g., lampiran, tabel, dokumen tanpa penomoran Pasal) |
| Metadata wajib per chunk | `nomor_peraturan`, `judul_peraturan`, `nomor_pasal`, `nomor_ayat` (bila ada), `url_sumber`, `halaman_pdf` |

## 8. Requirements Non-Fungsional

| ID | Requirement |
|---|---|
| N1 | Latency jawaban untuk pertanyaan umum idealnya di bawah ~15 detik |
| N2 | Kredensial (API key, dsb) tidak pernah ditulis langsung di kode — selalu lewat `.env` |
| N3 | Sistem tetap berjalan (graceful degradation) walau salah satu layanan eksternal (Langfuse/Telegram/Email) sedang tidak dikonfigurasi atau down |
| N4 | Proses scraping ke situs JDIH ESDM diberi jeda antar-request (tidak membebani server sumber) |
| N5 | Komunikasi antara Streamlit Frontend dan FastAPI Backend menggunakan REST API; respons LLM dikirim secara streaming (Server-Sent Events / SSE) agar pengguna tidak harus menunggu jawaban lengkap sebelum mulai membaca |

---

## 9. Arsitektur & Tech Stack

| Layer | Tools | Lokasi Hosting |
|---|---|---|
| LLM | Qwen/Qwen3.8-27B via LiteLLM Proxy (`aicon.air.id`) | Eksternal (sudah tersedia) |
| Vector DB | Milvus | **VPS pribadi (sudah tersedia)** |
| Embedding | BGE-M3 (self-hosted, `sentence-transformers`) | Lihat catatan di Bagian 10 |
| Reranker | BGE-Reranker-v2-M3 | Lihat catatan di Bagian 10 |
| OCR | Tesseract (+ `tesseract-ocr-ind`) | Lihat catatan di Bagian 10 |
| Orkestrasi Agent | LangGraph + LangChain | Lihat catatan di Bagian 10 |
| Task Queue (ingest) | Redis + RQ | Lihat catatan di Bagian 10 |
| Backend | FastAPI | Lihat catatan di Bagian 10 |
| Frontend | Streamlit | Lihat catatan di Bagian 10 |
| Observability | Langfuse Cloud | Eksternal (sudah tersedia) |
| Alerting | Telegram Bot API, SMTP Email | Eksternal (sudah tersedia) |

---

## 10. Asumsi & Batasan (Constraints)

> **Perubahan penting dari rencana sebelumnya**: akses VPS yang dimiliki **hanya mencakup
> Milvus**, bukan akses penuh untuk menjalankan container/layanan lain di VPS tersebut.

Implikasinya terhadap arsitektur:
- **Milvus** tetap dihubungi dari jarak jauh (remote) lewat endpoint yang diekspos VPS.
- **Semua layanan lain** (Backend FastAPI, Embedding Service, Reranker, Redis, Frontend
  Streamlit, OCR) untuk tahap pengembangan **dijalankan secara lokal di laptop/PC pribadi**,
  bukan di VPS.
- Ini berarti label "mendekati produksi" di tahap V1 lebih tepat dibaca sebagai
  **"arsitektur dan praktik kerja mendekati produksi"** (observability, alerting, clean
  architecture), bukan **"infrastruktur deployment yang sepenuhnya production-grade"** —
  karena topologi hosting-nya masih tersebar (VPS + laptop), bukan satu tempat terpusat.
- **Open question yang perlu dikonfirmasi** ke penyedia VPS: apakah memungkinkan menambah
  akses (misal Docker) untuk deploy backend juga di VPS yang sama nantinya, atau perlu VPS
  terpisah untuk itu. Ini memengaruhi apakah V2 proyek bisa full production-grade atau tetap
  hybrid (Milvus remote + backend lokal).

Asumsi lain:
- Akses API Qwen (`aicon.air.id`) tersedia stabil selama pengembangan; API key diisi sendiri
  di `.env` (tidak tercantum di dokumentasi API yang diterima).
- Kuota/rate limit API Qwen belum diketahui pasti — perlu dites di awal pengembangan, dan
  sistem alert (F9) akan membantu mendeteksi kalau kuota habis di tengah jalan.
- **BGE-M3 (Embedding)** dijalankan **in-process** (langsung dipanggil via `sentence-transformers`
  dalam proses Python yang sama dengan ingest pipeline), bukan sebagai microservice terpisah.
  Ini cukup untuk skala dokumen yang direncanakan dan mengurangi kompleksitas setup lokal.
  Evaluasi ulang ke mode microservice jika dibutuhkan di versi mendatang.
- **URL dokumen JDIH ESDM** diasumsikan stabil selama pengembangan V1. Scraper menyimpan URL
  asli sebagai bagian dari metadata chunk (sudah ada di F4), sehingga jika URL berubah,
  referensi ke sumber tetap bisa ditelusuri. Pembaruan URL ditangani secara manual jika terjadi.

---

## 11. Metrik Keberhasilan

| Metrik | Target Awal |
|---|---|
| Akurasi jawaban tersitasi benar (manual review) | ≥ 80% dari sampel uji |
| Tingkat halusinasi (jawaban tanpa dasar dokumen) | Mendekati 0% — lebih baik sistem bilang "tidak tahu" |
| Cache/latency (kalau semantic cache diaktifkan) | Sesuai baseline dari eksperimen Use Case 3 |
| Alert delivery rate saat sengaja disimulasikan gagal | 100% (alert harus selalu terkirim saat dites) |

---

## 12. Roadmap & Milestone

| Fase | Fokus | Output | Estimasi Durasi | Definition of Done |
|---|---|---|---|---|
| Fase 1 | Ingest Pipeline | Scraper + OCR + chunking per Pasal + Milvus terisi (subset 15-20 dokumen untuk validasi awal) | ~2 minggu | ≥15 dokumen berhasil di-ingest ke Milvus; chunk tervalidasi manual pada ≥3 jenis dokumen berbeda (PDF digital, scan, lampiran) |
| Fase 2 | Query Pipeline | Similarity search + reranker + LLM dengan sitasi sumber berfungsi end-to-end | ~2 minggu | Query end-to-end berjalan; akurasi sitasi pasal ≥80% pada set 10 pertanyaan uji yang disiapkan sebelumnya |
| Fase 3 | Reliability | Langfuse + alerting Telegram/Email + defense layer dasar terpasang | ~1 minggu | Setiap request tercatat di Langfuse; alert terkirim 100% saat skenario gangguan disimulasikan (koneksi gagal, kredit habis) |
| Fase 4 | Deployment & Demo | Frontend Streamlit berjalan, siap didemokan sebagai portofolio | ~1 minggu | Demo end-to-end dapat dijalankan tanpa error; README lengkap dan repo siap dipublikasi |

---

## 13. Risiko & Mitigasi

| Risiko | Mitigasi |
|---|---|
| Chunking salah bikin jawaban terputus/tidak lengkap | Strategi chunking sudah didefinisikan secara eksplisit (lihat *Catatan Implementasi: Strategi Chunking*); validasi manual pada ≥3 jenis dokumen berbeda; acceptance criteria masuk Definition of Done Fase 1 |
| Model salah kutip pasal (halusinasi sitasi) | Wajibkan LLM menyertakan metadata sumber dari chunk yang benar-benar diambil, bukan dibiarkan menulis bebas |
| Kuota API Qwen habis di tengah pengembangan | Sistem alert (F9) + siapkan provider cadangan (OpenRouter/Gemini) yang sudah pernah disiapkan sebelumnya |
| Topologi hybrid (VPS + laptop) bikin latency tambahan | Ukur latency saat Fase 2, evaluasi apakah perlu VPS tambahan kalau jadi masalah nyata |
| Data situs JDIH berubah struktur/URL | Scraper dibuat modular, mudah disesuaikan kalau struktur situs sumber berubah |

---

## 14. Referensi

- Sumber data: JDIH Kementerian ESDM — https://jdih.esdm.go.id/
- Contoh dokumen relevan: Permen ESDM No. 7 Tahun 2021 tentang Standardisasi di Bidang
  Ketenagalistrikan
- Arsitektur acuan: Elarch General Architecture (dokumen internal magang)
