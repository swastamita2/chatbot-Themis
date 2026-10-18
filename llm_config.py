"""
llm_config.py - Modul Konfigurasi LLM Multi-Provider
Mendukung provider:
- qwen   : Qwen/Qwen3.8-27B via LiteLLM Proxy (aicon.air.id) — UTAMA untuk Themis RAG
- gemini : Google Gemini (via langchain-google-genai) — cadangan
- openrouter: OpenRouter (via langchain-openai)
- ollama : Ollama lokal (via langchain-ollama)

Switch provider cukup dengan mengubah LLM_PROVIDER di .env
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict

from dotenv import load_dotenv

# Muat environment variables dari .env di root atau folder saat ini
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


def get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "qwen").strip().lower()


def _require_env(var_name: str, hint: str) -> str:
    """Ambil env var wajib; raise pesan panduan jika belum di-set."""
    value = os.getenv(var_name, "").strip()
    if not value:
        raise RuntimeError(f"{var_name} belum diset di .env. {hint}")
    return value


def _build_qwen(temperature: float):
    from langchain_openai import ChatOpenAI

    # Default: endpoint LiteLLM Proxy PLN.
    base_url = os.getenv("QWEN_BASE_URL", "https://aicon.air.id/").strip()
    if not base_url:
        raise RuntimeError(
            "QWEN_BASE_URL kosong di .env. "
            "Isi dengan endpoint LiteLLM Proxy, contoh: https://aicon.air.id/"
        )
    api_key = os.getenv("QWEN_API_KEY", "")
    model_name = os.getenv("QWEN_MODEL", "Qwen/Qwen3.8-27B")

    return ChatOpenAI(
        base_url=base_url.rstrip("/") + "/",
        api_key=api_key if api_key else "no-key",
        model=model_name,
        temperature=temperature,
        # Qwen3 bersifat hybrid-thinking: model "berpikir" (reasoning_content
        # tersembunyi) SEBELUM menjawab. Selama fase berpikir, tidak ada token
        # keluar → socket proxy idle → "Timeout on reading data from socket".
        # Perbaikan berlapis:
        #  1) chat_template_kwargs=enable_thinking False — mematikan fase
        #     berpikir (mekanisme yang dihormati deployment Qwen3/vLLM ini;
        #     flag `enable_thinking` biasa diabaikan proxy).
        #  2) max_tokens=8192 — headroom agar meski sedikit reasoning lolos,
        #     content tetap muat (4096 sering habis untuk reasoning → content kosong).
        #  3) request_timeout=300 — toleransi generasi yang lambat.
        #  4) agent men-retry 1x jika content tetap kosong (lihat agent.py).
        max_tokens=8192,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        request_timeout=300,
        max_retries=1,
    )


def _build_gemini(temperature: float):
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = _require_env(
        "GOOGLE_API_KEY",
        "Silakan isi API key dari aistudio.google.com",
    )
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )


def _build_openrouter(temperature: float):
    from langchain_openai import ChatOpenAI

    api_key = _require_env(
        "OPENROUTER_API_KEY",
        "Silakan isi API key dari openrouter.ai",
    )
    model_name = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=model_name,
        temperature=temperature,
    )


def _build_ollama(temperature: float):
    from langchain_ollama import ChatOllama

    model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

    return ChatOllama(
        model=model_name,
        temperature=temperature,
    )


# Registry provider: nama (di .env) → builder.
# Menambah provider baru = tambah satu fungsi builder + satu entri di sini.
_BUILDERS: Dict[str, Callable[[float], object]] = {
    "qwen": _build_qwen,
    "gemini": _build_gemini,
    "openrouter": _build_openrouter,
    "ollama": _build_ollama,
}


def get_model(temperature: float = 0, provider: str | None = None):
    """
    Mengembalikan instance model chat LangChain berdasarkan LLM_PROVIDER di .env.

    `provider` bersifat opsional (additive): bila None/empty, memakai provider
    default dari .env sehingga jalur lama tetap identik. Dipakai UI untuk
    beralih antar model yang benar-benar terkonfigurasi.
    """
    provider = (provider or get_provider()).strip().lower()
    builder = _BUILDERS.get(provider)
    if builder is None:
        raise ValueError(
            f"Provider '{provider}' tidak dikenali. "
            f"Pilih: {', '.join(sorted(_BUILDERS))} di .env"
        )
    return builder(temperature)


# Peta provider → (variabel env nama model, label untuk UI, variabel env kunci API).
# Dipakai untuk enumerasi model yang benar-benar tersedia (bukan daftar statis).
_MODEL_ENV: Dict[str, tuple] = {
    "qwen": ("QWEN_MODEL", "Qwen via LiteLLM Proxy", "QWEN_API_KEY"),
    "gemini": ("GEMINI_MODEL", "Google Gemini", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_MODEL", "OpenRouter", "OPENROUTER_API_KEY"),
    "ollama": ("OLLAMA_MODEL", "Ollama (lokal)", None),
}


def list_available_models() -> Dict[str, Any]:
    """
    Kembalikan model LLM yang tersedia berdasarkan .env (bukan daftar palsu).
    Hanya provider yang nama modelnya terisi (dan, bila diperlukan, kunci API
    tersedia) ditandai `available`. Provider default = LLM_PROVIDER di .env.
    """
    default = get_provider()
    models = []
    for prov, (model_var, label, key_var) in _MODEL_ENV.items():
        model_name = os.getenv(model_var, "").strip()
        key_ok = key_var is None or bool(os.getenv(key_var, "").strip())
        models.append({
            "provider": prov,
            "model": model_name,
            "label": label,
            "available": bool(model_name) and key_ok,
            "is_default": prov == default,
        })
    return {"default_provider": default, "models": models}
