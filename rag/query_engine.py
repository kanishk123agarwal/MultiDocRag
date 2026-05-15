from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from dotenv import load_dotenv

# Load env variables (contains GOOGLE_API_KEY)
load_dotenv()

def build_multi_doc_engine(indexes: dict) -> SubQuestionQueryEngine:
    """Build a SubQuestionQueryEngine over all document indexes."""
    # Configure Gemini settings globally
    Settings.llm = Gemini(model="models/gemini-2.5-flash")
    Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001")

    tools = []
    for doc_name, index in indexes.items():
        # Normalize tool name for LlamaIndex validation constraints
        tool_name = doc_name.replace(".pdf", "").replace(" ", "_").replace("-", "_").replace(".", "_")
        tool = QueryEngineTool(
            query_engine=index.as_query_engine(similarity_top_k=3),
            metadata=ToolMetadata(
                name=tool_name,
                description=f"Provides information from the document: {doc_name}"
            )
        )
        tools.append(tool)

    engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=tools,
        use_async=False
    )
    return engine

def query_each_document(indexes: dict, question: str) -> dict:
    """Query each document independently. Returns {doc_name: answer_text}."""
    # Configure Gemini settings globally
    Settings.llm = Gemini(model="models/gemini-2.5-flash")
    Settings.embed_model = GeminiEmbedding(model_name="models/gemini-embedding-001")
    
    results = {}
    for doc_name, index in indexes.items():
        qe = index.as_query_engine(similarity_top_k=3)
        response = qe.query(question)
        results[doc_name] = str(response)
    return results
