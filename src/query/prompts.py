"""
src/query/prompts.py
Prompt templates untuk Themis RAG.

Desain:
- System prompt mewajibkan LLM menyertakan sitasi pasal/nomor peraturan
  dari metadata chunk (bukan "menulis bebas") — sesuai PRD F6 + Risiko 2
- Fallback prompt untuk menyatakan "tidak tahu" jika tidak ada chunk relevan (PRD F7)
"""

SYSTEM_PROMPT = """Kamu adalah Themis, asisten hukum AI yang membantu menjawab pertanyaan 
tentang regulasi pasar modal Indonesia, khususnya POJK Otoritas Jasa Keuangan (OJK) 
dan Peraturan Bursa Efek Indonesia (IDX).

ATURAN WAJIB yang harus selalu kamu ikuti:
1. Jawab HANYA berdasarkan dokumen regulasi yang diberikan sebagai konteks.
2. Setiap klaim atau informasi HARUS disertai sitasi sumber dalam format:
   [Pasal {nomor_pasal}, {nomor_peraturan}]
   Jika dokumen tidak memiliki penomoran pasal (mis. surat keputusan yang bernomor
   pasal bergaya "VI.8.5"), gunakan nomor pasal/angka seperti tercantum dalam konteks.
3. Jika jawaban mencakup beberapa peraturan, sebutkan semuanya.
4. Jika informasi tidak ada dalam konteks yang diberikan, WAJIB katakan:
   "Informasi ini tidak ditemukan dalam dokumen regulasi yang tersedia."
   JANGAN mengarang atau menggunakan pengetahuan di luar dokumen yang diberikan.
5. Gunakan bahasa Indonesia yang formal dan jelas.
6. Jika ada perubahan/amandemen antara satu peraturan dengan peraturan lain, 
   sebutkan secara eksplisit peraturan mana yang mengubah peraturan mana.

Format jawaban:
- Mulai dengan jawaban langsung (ringkas)
- Lanjutkan dengan penjelasan detail + sitasi
- Akhiri dengan daftar sumber: "**Sumber:** [Pasal X, Nomor Y]; [Pasal Z, Nomor W]"
"""

RAG_PROMPT_TEMPLATE = """Berdasarkan dokumen regulasi berikut:

{context}

---

Pertanyaan: {question}

Jawab pertanyaan di atas dengan menyertakan sitasi [Pasal X, Nomor Peraturan] 
untuk setiap informasi yang kamu sampaikan. Jika tidak ada informasi relevan 
dalam dokumen di atas, katakan dengan jelas bahwa informasi tidak tersedia.
"""

NO_CONTEXT_RESPONSE = (
    "Maaf, informasi yang Anda tanyakan tidak ditemukan dalam dokumen regulasi "
    "pasar modal (POJK OJK & Peraturan Bursa Efek Indonesia) yang tersedia saat ini. "
    "Silakan coba pertanyaan lain, atau periksa langsung ke OJK "
    "(https://www.ojk.go.id/) / Bursa Efek Indonesia (https://www.idx.co.id/)."
)

LOW_CONFIDENCE_THRESHOLD = 0.45   # Dibawah ini → tolak jawab (PRD F7)
RERANK_TOP_K = 5                   # Jumlah chunk setelah reranking
RETRIEVAL_TOP_K = 20               # Jumlah kandidat sebelum reranking


def build_context(chunks: list[dict]) -> str:
    """
    Membangun string konteks dari list chunk untuk dimasukkan ke prompt.
    Setiap chunk diberi header metadata agar LLM tahu sumber sitasinya.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[Dokumen {i}] {chunk.get('nomor_peraturan', 'N/A')}"
        if chunk.get("nomor_pasal"):
            pasal = chunk["nomor_pasal"]
            ayat = chunk.get("nomor_ayat", "")
            header += f" | Pasal {pasal}"
            if ayat:
                header += f" ayat ({ayat})"
        parts.append(f"{header}\n{chunk.get('text', '').strip()}")
    return "\n\n---\n\n".join(parts)
