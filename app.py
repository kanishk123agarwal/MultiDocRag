import os
import gradio as gr
from dotenv import load_dotenv
from rag.indexer import index_all_documents
from rag.query_engine import build_multi_doc_engine, query_each_document
from rag.conflict_detector import detect_conflicts

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
        return (
            "Please upload and index documents first.",
            "Please upload and index documents first.",
            "Please upload and index documents first."
        )
    
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
        
    # 3. Detect conflicts
    try:
        conflict = detect_conflicts(question, doc_answers)
        if conflict["conflict_found"]:
            conflict_md = f"""⚠️ **Conflict Detected!**

**Explanation:** {conflict['explanation']}

| | Document | Factual Claim |
|---|---|---|
| **A** | {conflict['doc_a']} | {conflict['claim_a']} |
| **B** | {conflict['doc_b']} | {conflict['claim_b']} |"""
        else:
            conflict_md = "✅ **No conflict detected** across the sources."
    except Exception as e:
        conflict_md = f"Error in Conflict Detection: {e}"
        
    return answers_md, synthesized_md, conflict_md

# Build Gradio UI
with gr.Blocks(title="MultiDoc RAG - Phase 4") as demo:
    gr.Markdown("# 📄 MultiDoc RAG — Conflict Detection")
    gr.Markdown("Upload PDF research documents, build vector indexes, and query them. The system will detect and flag contradictions across sources.")
    
    with gr.Row():
        upload = gr.File(label="Upload PDFs", file_count="multiple", file_types=[".pdf"])
        index_btn = gr.Button("Index Documents", variant="primary")
        
    index_status = gr.Textbox(label="Status", interactive=False)
    index_btn.click(upload_and_index, inputs=upload, outputs=index_status)
    
    question = gr.Textbox(label="Your Question", placeholder="e.g. Compare the conclusions across these papers.")
    ask_btn = gr.Button("Submit Query", variant="secondary")
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📄 Answers per Document")
            answers_out = gr.Markdown()
        with gr.Column(scale=2):
            gr.Markdown("### 🧠 Synthesized Final Answer")
            synthesized_out = gr.Markdown()
        with gr.Column(scale=3):
            gr.Markdown("### ⚠️ Conflict Analysis")
            conflict_out = gr.Markdown()
            
    ask_btn.click(ask_question, inputs=question, outputs=[answers_out, synthesized_out, conflict_out])

if __name__ == "__main__":
    demo.launch()
