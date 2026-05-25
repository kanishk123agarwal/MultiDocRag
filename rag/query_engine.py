from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine


def build_multi_doc_engine(indexes: dict) -> SubQuestionQueryEngine:
    """
    Build a SubQuestionQueryEngine over all document indexes.
    Settings (LLM + embeddings) are already configured globally by indexer.py
    at module import time — no need to re-assign here.
    """
    tools = []
    for doc_name, index in indexes.items():
        # Normalize tool name: LlamaIndex tool names must be alphanumeric + underscore
        tool_name = (
            doc_name
            .replace(".pdf", "")
            .replace(".txt", "")
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "_")
        )
        tool = QueryEngineTool(
            query_engine=index.as_query_engine(similarity_top_k=6),
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
