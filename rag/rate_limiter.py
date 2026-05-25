"""
rate_limiter.py
---------------
Monkeypatches Gemini (LLM) and GeminiEmbedding to enforce free-tier
API rate limits separately for LLM calls and Embedding calls.

Key design:
  - SEPARATE rate limit delays for LLM (chat/complete) vs Embedding:
      LLM_RATE_LIMIT_DELAY  = 20s  → safe 3 RPM (LLM quota is tighter)
      EMBED_RATE_LIMIT_DELAY = 12s → safe 5 RPM (embedding quota)
  - Adaptive penalty: after a 429, the delay for that API type is
    temporarily raised and decays back to normal over time.
  - TRUE BATCH embedding: embeds a whole batch of texts in ONE API
    call via genai.embed_content(content=[...list...]).
  - Exponential backoff on retries (not just the API-reported delay).
  - Separate threading.Locks for LLM and Embedding to prevent them
    from blocking each other unnecessarily.
"""

import re
import time
import asyncio
import threading

import google.generativeai as genai
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# ── Per-API rate limit state ──────────────────────────────────────────────────
# LLM calls (complete / chat) — stricter quota on free tier
_llm_lock            = threading.Lock()
_llm_last_call       = 0.0
LLM_RATE_LIMIT_DELAY = 20.0   # 20 s gap → max 3 RPM (safe for free tier LLM)
_llm_penalty         = 0.0    # extra delay added after 429s, decays over time

# Embedding calls — slightly more generous quota
_emb_lock            = threading.Lock()
_emb_last_call       = 0.0
EMBED_RATE_LIMIT_DELAY = 12.0  # 12 s gap → max 5 RPM

def _wait(last_call: float, base_delay: float, penalty: float, label: str) -> float:
    """
    Wait until the rate-limit window + any penalty has passed.
    Returns the timestamp of this call (for updating last_call).
    """
    effective_delay = base_delay + penalty
    elapsed = time.time() - last_call
    wait = effective_delay - elapsed
    if wait > 0:
        print(f"[Rate Limiter] Waiting {wait:.1f}s before '{label}'...")
        time.sleep(wait)
    print(f"[Rate Limiter] → {label}")
    return time.time()

def _parse_retry_delay(exc: Exception) -> float:
    """Parse the retry_delay from a 429 response, fallback to 60s."""
    try:
        m = re.search(r'retry_delay\s*\{[^}]*seconds:\s*(\d+)', str(exc))
        if m:
            return float(m.group(1)) + 5.0
    except Exception:
        pass
    return 65.0

# ── LLM call wrapper ──────────────────────────────────────────────────────────
def _call_llm_sync(fn, label: str, max_retries: int = 6):
    """
    Thread-safe, throttled, retrying synchronous LLM API call.
    Uses exponential backoff on top of the API-reported retry delay.
    """
    global _llm_last_call, _llm_penalty

    with _llm_lock:
        _llm_last_call = _wait(_llm_last_call, LLM_RATE_LIMIT_DELAY, _llm_penalty, label)

        for attempt in range(1, max_retries + 1):
            try:
                result = fn()
                # Success — gradually reduce penalty (decay by 50% per success)
                _llm_penalty = max(0.0, _llm_penalty * 0.5)
                return result
            except Exception as e:
                is_quota = "429" in str(e) or "quota" in str(e).lower() or "resource" in str(e).lower()
                if is_quota and attempt < max_retries:
                    api_wait = _parse_retry_delay(e)
                    # Add exponential backoff on top: 1x, 1.5x, 2x, 2.5x, 3x
                    backoff = api_wait * (1.0 + (attempt - 1) * 0.5)
                    # Raise penalty so future calls also wait longer
                    _llm_penalty = min(60.0, _llm_penalty + 15.0)
                    print(f"[Rate Limiter] 429 on '{label}' — backing off {backoff:.0f}s (attempt {attempt}/{max_retries}, penalty now +{_llm_penalty:.0f}s)...")
                    time.sleep(backoff)
                    _llm_last_call = time.time()
                else:
                    raise
        raise RuntimeError(f"Gemini LLM still failing after {max_retries} retries.")

# ── Embedding call wrapper ────────────────────────────────────────────────────
def _call_emb_sync(fn, label: str, max_retries: int = 4):
    """Thread-safe, throttled, retrying synchronous Embedding API call."""
    global _emb_last_call

    with _emb_lock:
        _emb_last_call = _wait(_emb_last_call, EMBED_RATE_LIMIT_DELAY, 0.0, label)

        for attempt in range(1, max_retries + 1):
            try:
                return fn()
            except Exception as e:
                is_quota = "429" in str(e) or "quota" in str(e).lower()
                if is_quota and attempt < max_retries:
                    wait = _parse_retry_delay(e)
                    print(f"[Rate Limiter] 429 on '{label}' — waiting {wait:.0f}s (attempt {attempt}/{max_retries})...")
                    time.sleep(wait)
                    _emb_last_call = time.time()
                else:
                    raise
        raise RuntimeError(f"Gemini Embedding still failing after {max_retries} retries.")


# ── Patch Gemini LLM ──────────────────────────────────────────────────────────
_orig_complete = Gemini.complete
_orig_chat     = Gemini.chat

def _llm_complete(self, *a, **kw):
    return _call_llm_sync(
        lambda: _orig_complete.__get__(self, type(self))(*a, **kw),
        "LLM.complete()"
    )

def _llm_chat(self, *a, **kw):
    return _call_llm_sync(
        lambda: _orig_chat.__get__(self, type(self))(*a, **kw),
        "LLM.chat()"
    )

Gemini.complete = _llm_complete
Gemini.chat     = _llm_chat


# ── Patch GeminiEmbedding — TRUE BATCH API ────────────────────────────────────
# genai.embed_content() accepts a list of texts in a SINGLE API call.
# So the whole batch (default 10 texts) costs ONE rate-limit slot, not N slots.

_orig_get_text_emb  = GeminiEmbedding._get_text_embedding
_orig_get_query_emb = GeminiEmbedding._get_query_embedding

# --- sync single ---
def _emb_text(self, text, *a, **kw):
    return _call_emb_sync(
        lambda: _orig_get_text_emb.__get__(self, type(self))(text, *a, **kw),
        "Embed.single()"
    )

# --- sync BATCH: ONE API call for the entire list -------------------------
def _emb_texts(self, texts, *a, **kw):
    """
    True batch embedding: embeds all texts in ONE genai.embed_content() call.
    ONE rate-limit slot consumed regardless of batch size (default: 10 texts).
    """
    texts = list(texts)
    def _do_batch():
        result = genai.embed_content(
            model=self.model_name,
            content=texts,
            task_type=self.task_type,
            title=self.title,
        )
        embeddings = result.get("embedding", [])
        # Single-text batch returns a flat list — wrap it
        if texts and embeddings and not isinstance(embeddings[0], list):
            embeddings = [embeddings]
        return embeddings

    return _call_emb_sync(_do_batch, f"Embed.batch({len(texts)} texts)")

# --- sync query ---
def _emb_query(self, query, *a, **kw):
    return _call_emb_sync(
        lambda: _orig_get_query_emb.__get__(self, type(self))(query, *a, **kw),
        "Embed.query()"
    )

# --- async: delegate to sync via thread executor (Gemini has no async embed) ---
async def _aemb_text(self, text, *a, **kw):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _emb_text(self, text, *a, **kw))

async def _aemb_texts(self, texts, *a, **kw):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _emb_texts(self, texts, *a, **kw))

async def _aemb_query(self, query, *a, **kw):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _emb_query(self, query, *a, **kw))

GeminiEmbedding._get_text_embedding   = _emb_text
GeminiEmbedding._get_text_embeddings  = _emb_texts
GeminiEmbedding._get_query_embedding  = _emb_query
GeminiEmbedding._aget_text_embedding  = _aemb_text
GeminiEmbedding._aget_text_embeddings = _aemb_texts
GeminiEmbedding._aget_query_embedding = _aemb_query

print(f"[Rate Limiter] Active — LLM: {LLM_RATE_LIMIT_DELAY}s gap (3 RPM) | Embedding: {EMBED_RATE_LIMIT_DELAY}s gap (5 RPM) | Adaptive penalty ON | True batch embedding ON.")
