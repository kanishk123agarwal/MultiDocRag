import os
import gradio as gr
from dotenv import load_dotenv
from rag.indexer import index_all_documents
from rag.query_engine import build_multi_doc_engine, query_each_document

# Load env variables (contains GOOGLE_API_KEY)
load_dotenv()

indexes = {}
multi_doc_engine = None

def upload_and_index(files):
    global indexes, multi_doc_engine
    if not files:
        return "No files uploaded."
    
    os.makedirs("data/docs", exist_ok=True)
    for f in files:
        dest = os.path.join("data/docs", os.path.basename(f.name))
        with open(f.name, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())
        
    # Re-index all documents in data/docs/
    indexes = index_all_documents("data/docs")
    if indexes:
        multi_doc_engine = build_multi_doc_engine(indexes)
        
    return f"Indexed {len(indexes)} document(s): {', '.join(indexes.keys())}"

def ask_question(question):
    global indexes, multi_doc_engine
    if not indexes:
        return "Please upload and index documents first.", "Please upload and index documents first."
    
    # 1. Query each document independently
    doc_answers = query_each_document(indexes, question)
    answers_md = "\n\n".join([f"### 📄 {doc}:\n{ans}\n---" for doc, ans in doc_answers.items()])
    
    # 2. Query using SubQuestion Query Engine
    if not multi_doc_engine:
        multi_doc_engine = build_multi_doc_engine(indexes)
        
    try:
        response = multi_doc_engine.query(question)
        synthesized_md = str(response)
    except Exception as e:
        synthesized_md = f"Error in SubQuestion Engine: {e}"
        
    return answers_md, synthesized_md

# Build Gradio UI
with gr.Blocks(title="MultiDoc RAG - Phase 3") as demo:
    gr.Markdown("# 📄 MultiDoc RAG — SubQuestion Query Engine")
    gr.Markdown("Upload PDF research documents, build vector indexes, and query them using independent retrieval and a joint SubQuestion synthesis engine.")
    
    with gr.Row():
        upload = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"])
        index_btn = gr.Button("Index Documents", variant="primary")
        
    index_status = gr.Textbox(label="Status", interactive=False)
    index_btn.click(upload_and_index, inputs=upload, outputs=index_status)
    
    question = gr.Textbox(label="Your Question", placeholder="e.g. Compare the conclusions across these papers.")
    ask_btn = gr.Button("Submit Query", variant="secondary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📄 Answers per Document")
            answers_out = gr.Markdown()
        with gr.Column():
            gr.Markdown("### 🧠 Synthesized Final Answer (SubQuestion Engine)")
            synthesized_out = gr.Markdown()
            
    ask_btn.click(ask_question, inputs=question, outputs=[answers_out, synthesized_out])

if __name__ == "__main__":
    demo.launch()
