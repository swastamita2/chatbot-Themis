/* Themis RAG — frontend statis (vanilla JS, tanpa dependensi).
   Memakai API FastAPI yang sama yang dipakai backend: /api/chat, /api/health,
   /api/stats, /api/ingest, /api/models. Alur chat + sitasi tidak berubah. */
"use strict";

const $ = (id) => document.getElementById(id);

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  models: [],            // daftar model dari /api/models
  currentProvider: null, // provider yang dipilih (default = dari .env)
  topK: 5,
  messages: [],          // [{role, content, sources, conf, latency}]
  busy: false,
};

const WELCOME = (
  "Halo — saya Themis, asisten regulasi pasar modal Indonesia. Saya menjawab pertanyaan " +
  "(POJK OJK & Peraturan Bursa Efek Indonesia) " +
  "dengan rujukan pasal dari dokumen resmi. Apa yang ingin Anda tanyakan?"
);

// ── Util ─────────────────────────────────────────────────────────────────────
function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/* Renderer markdown ringkas & aman: escape dulu, lalu terapkan bold/code/list.
   Tidak menyisipkan HTML mentah dari jawaban LLM. */
function renderMarkdown(text) {
  const src = escapeHtml(text || "");
  let html = src
    // inline code
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    // bold
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  // Pecah paragraf; deteksi list sederhana (- / *) dan bernomor (1.)
  const blocks = html.split(/\n{2,}/);
  html = blocks
    .map((block) => {
      const lines = block.split("\n").map((l) => l.trim()).filter(Boolean);
      const isBullet = lines.length > 1 && lines.every((l) => /^[-*]\s/.test(l) || /^\d+\.\s/.test(l));
      if (isBullet) {
        return "<ul>" + lines.map((l) => "<li>" + l.replace(/^[-*]\s/, "") + "</li>").join("") + "</ul>";
      }
      if (lines.length > 1 && lines.every((l) => /^\d+\.\s/.test(l))) {
        return "<ol>" + lines.map((l) => "<li>" + l.replace(/^\d+\.\s/, "") + "</li>").join("") + "</ol>";
      }
      return "<p>" + lines.join("<br>") + "</p>";
    })
    .join("");
  return html;
}

/* Toast sederhana (status untuk elemen 'coming soon' & umpan balik aksi). */
let toastTimer = null;
function toast(msg, type) {
  const wrap = $("toastWrap");
  const t = document.createElement("div");
  t.className = "toast" + (type ? " " + type : "");
  t.textContent = msg;
  wrap.appendChild(t);
  requestAnimationFrame(() => t.classList.add("show"));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    t.classList.remove("show");
    setTimeout(() => t.remove(), 250);
  }, 2600);
}

// ── Model dropdown: model yang BENAR-BENAR tersedia (bukan "ChatGPT 4.0") ──
async function loadModels() {
  const sel = $("modelSelect");
  const dot = $("modelDot");
  try {
    const res = await fetch("/api/models");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    state.models = data.models || [];
    const def = data.default_provider;

    // Hanya tampilkan model yang available; default dipilih lebih dulu.
    const available = state.models.filter((m) => m.available);
    available.sort((a, b) => (b.is_default ? 1 : 0) - (a.is_default ? 1 : 0));

    sel.innerHTML = "";
    if (!available.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "Tidak ada model tersedia";
      sel.appendChild(opt);
      dot.style.background = "var(--err)";
      return;
    }
    for (const m of available) {
      const opt = document.createElement("option");
      opt.value = m.provider;
      const name = m.model || m.label;
      opt.textContent = name + (m.is_default ? "  (default)" : "");
      sel.appendChild(opt);
    }
    const defaultModel = available.find((m) => m.is_default);
    state.currentProvider = defaultModel ? defaultModel.provider : available[0].provider;
    sel.value = state.currentProvider;
    dot.style.background = "var(--ok)";

    sel.addEventListener("change", () => {
      state.currentProvider = sel.value;
      const m = state.models.find((x) => x.provider === sel.value);
      toast("Model: " + ((m && m.model) || sel.value));
    });
  } catch (e) {
    dot.style.background = "var(--err)";
    sel.innerHTML = '<option value="">Gagal memuat model</option>';
    console.error("loadModels:", e);
  }
}

// ── Status sistem (Milvus + LLM) untuk panel Configuration ───────────────────
// /api/health  → { status, components: { milvus:{ok,message}, llm:{ok,message}, langfuse:{...} } }
// /api/stats   → { status:"success", collection_name, total_chunks }  (atau 503)
async function loadStatus() {
  const rows = $("statusRows");
  const setRow = (name, ok, msg) => {
    const el = [...rows.children].find((c) => {
      const n = c.querySelector(".s-name");
      return n && n.textContent.trim() === name;
    });
    if (!el) return;
    el.classList.remove("ok", "err");
    el.classList.add(ok ? "ok" : "err");
    el.querySelector(".s-msg").textContent = msg || "—";
  };

  let health = null;
  let stats = null;
  try { health = await fetch("/api/health").then((r) => r.json()).catch(() => null); } catch (e) {}
  try { stats = await fetch("/api/stats").then((r) => r.json()).catch(() => null); } catch (e) {}

  // Milvus
  const m = health && health.components && health.components.milvus;
  const milvusOk = m ? !!m.ok : (stats && stats.status === "success");
  let milvusMsg = m ? m.message : (milvusOk ? "" : "tidak terhubung");
  if (stats && stats.status === "success") {
    milvusMsg = (stats.total_chunks ?? 0).toLocaleString() + " chunks · " + (stats.collection_name || "milvus");
  }
  setRow("Milvus", milvusOk, milvusMsg);

  // LLM — status dari health + nama model default yang benar-benar terkonfigurasi
  const l = health && health.components && health.components.llm;
  const llmOk = l ? !!l.ok : true;
  const def = state.models.find((x) => x.is_default && x.available);
  setRow("LLM", llmOk, def ? (def.model || def.label) : (l ? l.message : "memeriksa…"));
}

// ── Render sitasi (panel sumber pasal) — 1 implementasi untuk semua pesan ──
function citationHtml(src, i) {
  const pasal = src.nomor_pasal || "—";
  const ayat = src.nomor_ayat;
  const perat = src.nomor_peraturan || "Peraturan";
  const skor = (src.score ?? 0).toFixed(3);
  const hal = src.halaman_pdf;
  const parts = ["Pasal " + pasal];
  if (ayat) parts[0] = "Pasal " + pasal + " ayat (" + ayat + ")";
  if (hal) parts.push("hal. " + hal);
  const snippet = escapeHtml((src.text || "").slice(0, 350)) + ((src.text || "").length > 350 ? "…" : "");
  return (
    '<div class="cite">' +
      '<div class="head">' +
        '<span>[' + i + '] ' + escapeHtml(perat) + '</span>' +
        '<span class="meta">' + escapeHtml(parts.join(" · ")) + '</span>' +
        '<span class="score-tag">' + skor + '</span>' +
      '</div>' +
      (snippet ? '<div class="snippet">“' + snippet + '”</div>' : "") +
    '</div>'
  );
}

function messageHtml(msg) {
  const isUser = msg.role === "user";
  const roleLabel = isUser ? "Anda" : "Themis";
  let body = "";
  if (!isUser) {
    const ok = msg.conf;
    const label = ok
      ? ("Terdokumentasi" + (msg.latency ? " · " + msg.latency + "s" : ""))
      : ("Confidence rendah — data terbatas" + (msg.latency ? " · " + msg.latency + "s" : ""));
    body += '<span class="conf ' + (ok ? "ok" : "low") + '"><span class="dot"></span>' + label + "</span>";
  }
  body += '<div class="text">' + renderMarkdown(msg.content) + "</div>";
  if (!isUser && msg.sources && msg.sources.length) {
    body +=
      '<div class="cite-wrap">' +
        '<button class="cite-toggle" type="button" aria-expanded="false">' +
          '<svg class="ic"><use href="#i-book"/></svg> Rujukan &amp; Sitasi Pasal (' + msg.sources.length + ')' +
        '</button>' +
        '<div class="cite-list" hidden>' +
          msg.sources.map((s, i) => citationHtml(s, i + 1)).join("") +
        '</div>' +
      '</div>';
  }
  return (
    '<div class="msg ' + (isUser ? "user" : "assistant") + '">' +
      '<div class="avatar"><svg class="ic"><use href="#' + (isUser ? "i-plus" : "i-scale") + '" /></svg></div>' +
      '<div class="body"><div class="role">' + roleLabel + "</div>" + body + "</div>" +
    "</div>"
  );
}

function addMessage(msg) {
  state.messages.push(msg);
  const stage = $("stage");
  $("emptyWrap").hidden = true;
  const msgs = $("msgs");
  msgs.hidden = false;
  const div = document.createElement("div");
  div.innerHTML = messageHtml(msg);
  msgs.appendChild(div.firstElementChild);
  // Toggle sitasi
  const tog = div.querySelector(".cite-toggle");
  if (tog) {
    tog.addEventListener("click", () => {
      const list = div.querySelector(".cite-list");
      const open = list.hidden;
      list.hidden = !open;
      tog.setAttribute("aria-expanded", String(open));
    });
  }
  stage.scrollTop = stage.scrollHeight;
}

// ── Kirim pertanyaan (alur inti — panggil /api/chat) ─────────────────────────
async function sendPrompt(text) {
  text = (text || "").trim();
  if (!text || state.busy) return;
  state.busy = true;
  updateSendState();

  addMessage({ role: "user", content: text });
  const stage = $("stage");
  const loader = document.createElement("div");
  loader.className = "loading";
  loader.innerHTML = '<div class="spinner"></div> Mencari pasal relevan (BGE-M3 + reranker)…';
  $("msgs").appendChild(loader);
  stage.scrollTop = stage.scrollHeight;

  const start = performance.now();
  let latency = 0;
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: text,
        top_k: state.topK,
        provider: state.currentProvider || undefined,
      }),
    });
    latency = Math.round((performance.now() - start) / 100) / 10;
    const data = await res.json().catch(() => ({}));
    loader.remove();
    if (!res.ok) {
      const detail = (data && data.detail) || "Gagal memproses pertanyaan.";
      // 400 = pesan validasi (sudah ramah); 500/layar = tambah label kesalahan.
      const content = res.status === 400 ? detail : "Terjadi kesalahan: " + detail;
      addMessage({ role: "assistant", content, sources: [], conf: false, latency });
      return;
    }
    addMessage({
      role: "assistant",
      content: data.answer || "Maaf, tidak ada respon.",
      sources: data.sources || [],
      conf: data.confidence_passed,
      latency,
    });
  } catch (e) {
    loader.remove();
    addMessage({
      role: "assistant",
      content: "Tidak dapat terhubung ke server: " + e.message,
      sources: [], conf: false, latency,
    });
  } finally {
    state.busy = false;
    updateSendState();
  }
}

// ── Composer: textarea auto-resize + tombol kirim + Enter untuk kirim ────────
function updateSendState() {
  const btn = $("btnSend");
  const hasText = $("chatInput").value.trim().length >= 3;
  btn.disabled = state.busy || !hasText;
}

function setupComposer() {
  const ta = $("chatInput");
  ta.addEventListener("input", () => {
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 180) + "px";
    updateSendState();
  });
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const t = ta.value.trim();
      if (t && !state.busy) { ta.value = ""; ta.style.height = "auto"; sendPrompt(t); }
    }
  });
  $("btnSend").addEventListener("click", () => {
    const t = ta.value.trim();
    if (t && !state.busy) { ta.value = ""; ta.style.height = "auto"; sendPrompt(t); }
  });
}

// ── Capsule (quick start) → isi composer dengan prefix, fokus ke input ───────
function setupCapsules() {
  document.querySelectorAll(".capsule[data-fill]").forEach((b) => {
    b.addEventListener("click", () => {
      const ta = $("chatInput");
      ta.value = b.dataset.fill;
      ta.focus();
      ta.dispatchEvent(new Event("input"));
    });
  });
}

// ── New Chat: reset percakapan ───────────────────────────────────────────────
function newChat() {
  state.messages = [{ role: "assistant", content: WELCOME, sources: [], conf: true }];
  $("msgs").innerHTML = "";
  $("msgs").hidden = true;
  $("emptyWrap").hidden = false;
  const ta = $("chatInput");
  ta.value = ""; ta.style.height = "auto"; ta.focus();
  updateSendState();
  toast("Percakapan baru dimulai.");
}

// ── Elemen 'coming soon' (jangan dibuat berfungsi bohong) ─────────────────────
// Dipakai aria-disabled (bukan attr `disabled`) agar tombol tetap memberi
// umpan balik nyata (toast "coming soon") saat diklik, sekaligus terlihat
// jelas sebagai status "segera hadir". Bukan aksi palsu.
function setupSoon() {
  const soonMsg = "Fitur ini segera hadir (coming soon).";
  const markSoon = (el) => {
    if (!el || el.dataset.soonSet) return;
    el.dataset.soonSet = "1";
    el.setAttribute("aria-disabled", "true");
    el.classList.add("coming-soon");
    el.addEventListener("click", (e) => { e.preventDefault(); toast(soonMsg); });
  };
  ["btnUpgrade", "btnAttach", "btnOptions", "btnMic"].forEach((id) => markSoon($(id)));
  document.querySelectorAll(".card .mini-cta[data-soon]").forEach(markSoon);
  ["navArchived", "navLibrary"].forEach((id) => markSoon($(id)));
  document.querySelectorAll('[data-nav="ws"]').forEach(markSoon);
}

// ── Parameter top-k (dari panel Configuration) ───────────────────────────────
function setupTopK() {
  $("topk").addEventListener("change", () => { state.topK = parseInt($("topk").value, 10) || 5; });
}

// ── Ingest regulasi baru (panggil /api/ingest) ───────────────────────────────
async function runIngest() {
  const url = $("ingestUrl").value.trim();
  const msg = $("ingestMsg");
  const btn = $("btnIngest");
  if (!url) { msg.textContent = "Masukkan URL dokumen terlebih dahulu."; msg.style.color = "var(--warn)"; return; }
  btn.disabled = true; msg.style.color = "var(--text-muted)";
  msg.textContent = "Mengunduh, mem-parsing, chunking, dan embedding dokumen…";
  try {
    const res = await fetch("/api/ingest", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, run_in_background: false }),
    });
    const data = await res.json().catch(() => ({}));
    if (res.ok) {
      msg.style.color = "var(--ok)";
      msg.textContent = "Berhasil — " + (data.nomor_peraturan || url) + " · " + (data.num_chunks ?? 0) + " chunks";
      $("ingestUrl").value = "";
      loadStatus();
    } else {
      msg.style.color = "var(--err)";
      msg.textContent = "Gagal ingest: " + ((data && data.detail) || "error");
    }
  } catch (e) {
    msg.style.color = "var(--err)"; msg.textContent = "Terjadi error: " + e.message;
  } finally { btn.disabled = false; }
}

// ── Modal helpers ────────────────────────────────────────────────────────────
function openModal(id) { $(id).classList.add("open"); }
function closeModal(id) { $(id).classList.remove("open"); }

function setupModals() {
  // Buka
  $("btnConfig").addEventListener("click", () => { loadStatus(); openModal("overlayConfig"); });
  $("btnExport").addEventListener("click", () => openModal("overlayExport"));
  $("btnSettings").addEventListener("click", () => { loadStatus(); openModal("overlayConfig"); });
  // Tutup
  $("closeConfig").addEventListener("click", () => closeModal("overlayConfig"));
  $("closeExport").addEventListener("click", () => closeModal("overlayExport"));
  // Klik di luar panel = tutup
  ["overlayConfig", "overlayExport"].forEach((id) => {
    $(id).addEventListener("click", (e) => { if (e.target === $(id)) closeModal(id); });
  });
  // Esc = tutup
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { closeModal("overlayConfig"); closeModal("overlayExport"); }
  });
  $("btnIngest").addEventListener("click", runIngest);
}

// ── Export percakapan → Markdown ─────────────────────────────────────────────
function conversationMarkdown() {
  const lines = ["# Themis — Percakapan", "",
    "> Regulasi pasar modal Indonesia (POJK OJK & Peraturan Bursa Efek Indonesia)", ""];
  for (const m of state.messages) {
    lines.push("## " + (m.role === "user" ? "Anda" : "Themis"), "");
    lines.push(m.content, "");
    if (m.sources && m.sources.length) {
      lines.push("**Rujukan & Sitasi Pasal:**", "");
      m.sources.forEach((s, i) => {
        let meta = "Pasal " + (s.nomor_pasal || "—");
        if (s.nomor_ayat) meta += " ayat (" + s.nomor_ayat + ")";
        lines.push(
          i + 1 + ". " + (s.nomor_peraturan || "Peraturan") + " — " + meta +
          (s.halaman_pdf ? " (hal. " + s.halaman_pdf + ")" : "") + " — skor " + (s.score ?? 0).toFixed(3)
        );
      });
      lines.push("");
    }
  }
  return lines.join("\n");
}

function setupExport() {
  $("btnDownloadMd").addEventListener("click", () => {
    const blob = new Blob([conversationMarkdown()], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "themis-percakapan-" + new Date().toISOString().slice(0, 10) + ".md";
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    toast("Percakapan diunduh sebagai .md", "ok");
  });
  $("btnCopyMd").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(conversationMarkdown()); toast("Disalin ke clipboard", "ok"); }
    catch (e) { toast("Gagal menyalin: " + e.message, "err"); }
  });
}

// ── Nav: Chat aktif; yang lain coming soon ───────────────────────────────────
function setupNav() {
  $("navChat").addEventListener("click", (e) => {
    e.preventDefault();
    document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
    $("navChat").classList.add("active");
    newChat();
  });
}

// ── Init ─────────────────────────────────────────────────────────────────────
function init() {
  state.messages = [{ role: "assistant", content: WELCOME, sources: [], conf: true }];
  setupComposer();
  setupCapsules();
  setupSoon();
  setupTopK();
  setupModals();
  setupExport();
  setupNav();
  $("btnNewChat").addEventListener("click", newChat);
  updateSendState();
  loadModels();   // model yang benar-benar tersedia
  loadStatus();   // Milvus + LLM
}

document.addEventListener("DOMContentLoaded", init);
