import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from dotenv import load_dotenv

# Load env variables (contains GOOGLE_API_KEY)
load_dotenv()

# Configure global Settings for Google Gemini LLM and Embeddings
Settings.llm = Gemini(model="models/gemini-2.5-flash")
Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001")

def index_single_document(file_path: str) -> VectorStoreIndex:
    """Build a LlamaIndex VectorStoreIndex for a single PDF."""
    reader = SimpleDirectoryReader(input_files=[file_path])
    documents = reader.load_data()

    # Tag each document with its filename for citation
    for doc in documents:
        doc.metadata["filename"] = os.path.basename(file_path)

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter]
    )
    return index

def index_all_documents(doc_dir: str) -> dict:
    """Build one index per PDF. Returns {filename: index}."""
    indexes = {}
    if not os.path.exists(doc_dir):
        return indexes
        
    for fname in os.listdir(doc_dir):
        if fname.endswith(".pdf"):
            path = os.path.join(doc_dir, fname)
            print(f"Indexing: {fname}")
            indexes[fname] = index_single_document(path)
    return indexes
