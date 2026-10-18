import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# Warmup model sekali (simulasi startup Streamlit)
from src.warmup import warmup_all

t0 = time.time()
warmup_all()
print(f"[WARMUP] model siap dalam {time.time()-t0:.1f}s (sekali saja)\n")

from src.query.agent import run_rag_agent

questions = [
    "Apa syarat agar perusahaan efek dapat menjadi penjamin emisi efek?",
    "Bagaimana ketentuan perdagangan efek pada bursa efek?",
    "Apa syarat keanggotaan bursa bagi perusahaan efek?",
]

for q in questions:
    t0 = time.time()
    res = run_rag_agent(q, top_k=3)
    lat = time.time() - t0
    conf = res.get("confidence_passed", False)
    srcs = res.get("sources", [])
    print("=" * 70)
    print("Q:", q)
    print(f"Latensi (model sudah warm): {lat:.1f}s | Confidence lolos: {conf}")
    print("-" * 70)
    print("JAWABAN:", res.get("answer", "")[:600])
    print("-" * 70)
    for s in srcs[:3]:
        print(
            f"  - {s.get('nomor_peraturan')} | Pasal {s.get('nomor_pasal')} "
            f"ayat {s.get('nomor_ayat')} | skor {s.get('score', 0):.4f}"
        )
    print()
