import os
import json
import hashlib
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader,
    StorageContext, load_index_from_storage, Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv

load_dotenv()

# Configure global Settings for Google Gemini LLM and Embeddings
Settings.llm = Gemini(model="models/gemini-3.5-flash")
Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001")

CACHE_DIR = "data/cache"

def _cache_path(fname: str) -> str:
    """Return the directory where this document's index is cached."""
    # Use the filename (url-safe) as the cache folder name
    safe = fname.replace(" ", "_").replace("/", "_")
    return os.path.join(CACHE_DIR, safe)

def _meta_path(fname: str) -> str:
    return os.path.join(_cache_path(fname), "_meta.json")

def _get_file_mtime(file_path: str) -> float:
    return os.path.getmtime(file_path)

def _is_cache_valid(fname: str, file_path: str) -> bool:
    """Returns True if a valid cached index exists and the source file hasn't changed."""
    meta_file = _meta_path(fname)
    cache_dir = _cache_path(fname)
    if not os.path.exists(meta_file):
        return False
    if not os.path.exists(os.path.join(cache_dir, "docstore.json")):
        return False
    try:
        with open(meta_file) as f:
            meta = json.load(f)
        return meta.get("mtime") == _get_file_mtime(file_path)
    except Exception:
        return False

def _save_meta(fname: str, file_path: str):
    meta = {"mtime": _get_file_mtime(file_path), "source": fname}
    with open(_meta_path(fname), "w") as f:
        json.dump(meta, f)

def index_single_document(file_path: str) -> VectorStoreIndex:
    """
    Build (or load from disk cache) a VectorStoreIndex for a single document.
    On cache hit: loads instantly with zero API calls.
    On cache miss: embeds and saves to disk for future runs.
    """
    fname = os.path.basename(file_path)
    cache_dir = _cache_path(fname)

    # --- Cache HIT: load from disk ---
    if _is_cache_valid(fname, file_path):
        print(f"[Cache] Loading '{fname}' from disk cache (no API calls needed).")
        storage_ctx = StorageContext.from_defaults(persist_dir=cache_dir)
        index = load_index_from_storage(storage_ctx)
        return index

    # --- Cache MISS: embed and persist ---
    print(f"[Indexing] Embedding '{fname}' (first time — will be cached for future runs).")
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()

    for doc in documents:
        doc.metadata["filename"] = fname

    # Larger chunks = fewer embedding API calls (halved from 512 → 1024)
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
    index = VectorStoreIndex.from_documents(documents, transformations=[splitter])

    # Persist to disk
    os.makedirs(cache_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=cache_dir)
    _save_meta(fname, file_path)
    print(f"[Cache] Saved '{fname}' index to disk. Future runs will skip embedding.")

    return index

def index_all_documents(doc_dir: str) -> dict:
    """
    Build one cached index per document (PDF or TXT).
    Returns {filename: VectorStoreIndex}.
    Documents already cached are loaded from disk instantly.
    """
    indexes = {}
    if not os.path.exists(doc_dir):
        return indexes

    for fname in os.listdir(doc_dir):
        if fname.endswith(".pdf") or fname.endswith(".txt"):
            path = os.path.join(doc_dir, fname)
            indexes[fname] = index_single_document(path)

    return indexes

def invalidate_cache(fname: str):
    """Remove the disk cache for a specific document (call when file is deleted/replaced)."""
    import shutil
    cache_dir = _cache_path(fname)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"[Cache] Cleared cache for '{fname}'.")
