import os
import gradio as gr
from dotenv import load_dotenv
from rag.indexer import index_all_documents

# Load env variables (contains GOOGLE_API_KEY)
load_dotenv()

indexes = {}

def upload_and_index(files):
    global indexes
    if not files:
        return "No files uploaded."
    
    os.makedirs("data/docs", exist_ok=True)
    saved_files = []
    for f in files:
        dest = os.path.join("data/docs", os.path.basename(f.name))
        with open(f.name, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())
        saved_files.append(os.path.basename(f.name))
        
    # Re-index all documents in data/docs/
    indexes = index_all_documents("data/docs")
    return f"Indexed {len(indexes)} document(s): {', '.join(indexes.keys())}"

def ask_question(question):
    if not indexes:
        return "Please upload and index documents first."
    
    results = []
    for doc_name, index in indexes.items():
        # Setup query engine with top_k=3
        qe = index.as_query_engine(similarity_top_k=3)
        response = qe.query(question)
        results.append(f"### 📄 {doc_name}\n{response}\n---")
    
    return "\n\n".join(results)

# Build Gradio UI
with gr.Blocks(title="MultiDoc RAG - Phase 2") as demo:
    gr.Markdown("# 📄 MultiDoc RAG — Indexing & Querying")
    gr.Markdown("Upload PDF research documents, build vector indexes, and query them to verify retrieval.")
    
    with gr.Row():
        upload = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"])
        index_btn = gr.Button("Index Documents", variant="primary")
        
    index_status = gr.Textbox(label="Status", interactive=False)
    index_btn.click(upload_and_index, inputs=upload, outputs=index_status)
    
    question = gr.Textbox(label="Your Question", placeholder="e.g. Summarize the main conclusion of the document.")
    ask_btn = gr.Button("Query Documents", variant="secondary")
    
    answers_out = gr.Markdown(label="Answers per Document")
    ask_btn.click(ask_question, inputs=question, outputs=answers_out)

if __name__ == "__main__":
    demo.launch()
