"""
scripts/check_counts.py
Cek jumlah chunk per peraturan di Milvus (diagnostik cepat).
Statistik row_count bisa stale; hitung ini memakai query (andal).
"""

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.vector_store.milvus_client import FIELD_NOMOR_PERATURAN, get_client

c = get_client()
res = c.query(
    filter_expr=f'{FIELD_NOMOR_PERATURAN} != ""',
    output_fields=[FIELD_NOMOR_PERATURAN],
    limit=10000,
)

cnt = Counter(r[FIELD_NOMOR_PERATURAN] for r in res)
total = 0
for nom, n in sorted(cnt.items(), key=lambda x: -x[1]):
    print(f"  {n:5d}  {nom}")
    total += n
print(f"TOTAL chunk (query-based, andal): {total}")
print("Stat API (bisa stale):", c.count())
