import time
import asyncio
from llama_index.llms.gemini import Gemini

# Store original methods
original_complete = Gemini.complete
original_chat = Gemini.chat
original_acomplete = Gemini.acomplete
original_achat = Gemini.achat

# Rate limit state
_last_request_time = 0.0
RATE_LIMIT_DELAY = 12.0  # Seconds between requests to enforce 5 RPM limit

def _get_sleep_time() -> float:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < RATE_LIMIT_DELAY:
        return RATE_LIMIT_DELAY - elapsed
    return 0.0

def _update_last_request_time():
    global _last_request_time
    _last_request_time = time.time()

# Wrapper implementations
def rate_limited_complete(self, *args, **kwargs):
    sleep_time = _get_sleep_time()
    if sleep_time > 0:
        print(f"[Rate Limiter] Sleeping for {sleep_time:.2f}s to respect 5 RPM limit...")
        time.sleep(sleep_time)
    
    print("[Rate Limiter] Calling Gemini API (complete)...")
    try:
        bound_method = original_complete.__get__(self, self.__class__)
        res = bound_method(*args, **kwargs)
        return res
    finally:
        _update_last_request_time()

def rate_limited_chat(self, *args, **kwargs):
    sleep_time = _get_sleep_time()
    if sleep_time > 0:
        print(f"[Rate Limiter] Sleeping for {sleep_time:.2f}s to respect 5 RPM limit...")
        time.sleep(sleep_time)
    
    print("[Rate Limiter] Calling Gemini API (chat)...")
    try:
        bound_method = original_chat.__get__(self, self.__class__)
        res = bound_method(*args, **kwargs)
        return res
    finally:
        _update_last_request_time()

async def rate_limited_acomplete(self, *args, **kwargs):
    sleep_time = _get_sleep_time()
    if sleep_time > 0:
        print(f"[Rate Limiter] Sleeping async for {sleep_time:.2f}s to respect 5 RPM limit...")
        await asyncio.sleep(sleep_time)
    
    print("[Rate Limiter] Calling Gemini API (acomplete)...")
    try:
        bound_method = original_acomplete.__get__(self, self.__class__)
        res = await bound_method(*args, **kwargs)
        return res
    finally:
        _update_last_request_time()

async def rate_limited_achat(self, *args, **kwargs):
    sleep_time = _get_sleep_time()
    if sleep_time > 0:
        print(f"[Rate Limiter] Sleeping async for {sleep_time:.2f}s to respect 5 RPM limit...")
        await asyncio.sleep(sleep_time)
    
    print("[Rate Limiter] Calling Gemini API (achat)...")
    try:
        bound_method = original_achat.__get__(self, self.__class__)
        res = await bound_method(*args, **kwargs)
        return res
    finally:
        _update_last_request_time()

# Apply the monkeypatch
Gemini.complete = rate_limited_complete
Gemini.chat = rate_limited_chat
Gemini.acomplete = rate_limited_acomplete
Gemini.achat = rate_limited_achat

print("[Rate Limiter] Successfully monkeypatched Gemini LLM for rate limiting (5 RPM limit).")
