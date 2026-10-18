"""
scripts/diagnose_query.py
Skrip diagnostik: inspect koleksi Milvus + ukur latensi tiap tahap query pipeline.
Jalankan: python scripts/diagnose_query.py [--query "..."]
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def inspect_milvus():
    from src.vector_store.milvus_client import get_client

    c = get_client()
    print(f"[MILVUS] collection={c.collection_name}")
    print(f"[MILVUS] count chunk = {c.count()}")
    res = c.query(
        filter_expr='nomor_peraturan != ""',
        output_fields=["nomor_peraturan", "judul_peraturan", "nomor_pasal", "text"],
        limit=8,
    )
    for r in res:
        print("  -", r.get("nomor_peraturan"), "|", r.get("nomor_pasal"), "|", (r.get("text") or "")[:80].replace("\n", " "))


def tim_query(query: str):
    t0 = time.time()
    from src.embeddings.bge_m3 import embed_query

    emb = embed_query(query)
    t1 = time.time()
    print(f"[TIME] embed_query (load model + infer): {t1 - t0:.2f}s")

    from src.vector_store.milvus_client import get_client

    c = get_client()
    hits = c.search(emb, top_k=10)
    t2 = time.time()
    print(f"[TIME] retrieval Milvus top-10: {t2 - t1:.2f}s")
    for h in hits[:5]:
        print(f"   score={h['score']:.4f} | {h['nomor_peraturan']} Pasal {h['nomor_pasal']} | {h['text'][:70]!r}")

    if not hits:
        print("[TIME] (kosong, lewati rerank/LLM)")
        return

    from src.query.reranker import rerank

    top = rerank(query, hits, top_k=3)
    t3 = time.time()
    print(f"[TIME] reranker (load model + score {len(hits)} pairs -> {len(top)}): {t3 - t2:.2f}s")
    for h in top:
        print(f"   reranker_score={h['reranker_score']:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="Bagaimana aturan kapasitas untuk PLTS Atap?")
    args = p.parse_args()

    inspect_milvus()
    print("\n=== Timed query:", args.query, "===\n")
    tim_query(args.query)


if __name__ == "__main__":
    main()
