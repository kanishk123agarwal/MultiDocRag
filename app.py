import os
import time
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

def clear_documents():
    global indexes, multi_doc_engine
    indexes = {}
    multi_doc_engine = None
    if os.path.exists("data/docs"):
        for fname in os.listdir("data/docs"):
            if fname.endswith(".pdf") or fname.endswith(".txt"):
                try:
                    os.remove(os.path.join("data/docs", fname))
                except Exception:
                    pass
    return "All documents cleared from workspace."

def ask_question(question, run_each, run_synthesized, run_conflicts):
    global indexes, multi_doc_engine
    if not indexes:
        err_msg = "Please upload and index documents first."
        yield err_msg, err_msg, err_msg, "❌ No documents indexed. Please upload PDFs and click 'Index Documents' first."
        return
    
    status_log = "🚦 Starting execution flow...\n"
    answers_md = "⌛ Pending run..." if run_each else "🚫 Disabled."
    synthesized_md = "⌛ Pending run..." if run_synthesized else "🚫 Disabled."
    conflict_md = "⌛ Pending run..." if run_conflicts else "🚫 Disabled."
    
    yield answers_md, synthesized_md, conflict_md, status_log
    
    doc_answers = {}
    
    # 1. Query each document independently
    if run_each:
        status_log += f"📄 Step 1: Querying each of the {len(indexes)} documents individually...\n"
        yield answers_md, synthesized_md, conflict_md, status_log
        
        answers_list = []
        for i, (doc, index) in enumerate(indexes.items(), 1):
            status_log += f"👉 [{i}/{len(indexes)}] Querying: {doc}...\n"
            yield answers_md, synthesized_md, conflict_md, status_log
            
            try:
                qe = index.as_query_engine(similarity_top_k=3)
                ans = str(qe.query(question))
                doc_answers[doc] = ans
                answers_list.append(f"### 📄 {doc}:\n{ans}\n---")
                answers_md = "\n\n".join(answers_list)
                status_log += f"   ✅ Received answer from {doc}\n"
            except Exception as e:
                status_log += f"   ❌ Error querying {doc}: {e}\n"
                answers_list.append(f"### 📄 {doc}:\nError: {e}\n---")
                answers_md = "\n\n".join(answers_list)
            yield answers_md, synthesized_md, conflict_md, status_log
    else:
        status_log += "📄 Step 1: Individual document queries skipped.\n"
        yield answers_md, synthesized_md, conflict_md, status_log
        
    # 2. Query using SubQuestion Query Engine
    if run_synthesized:
        status_log += "🧠 Step 2: Running SubQuestion Query Engine (synthesizing global answer)...\n"
        status_log += "   ℹ️ Note: This makes multiple sub-queries. Calls are auto-paced (12s delay) to stay under the 5 RPM limit.\n"
        yield answers_md, synthesized_md, conflict_md, status_log
        
        if not multi_doc_engine:
            multi_doc_engine = build_multi_doc_engine(indexes)
            
        try:
            response = multi_doc_engine.query(question)
            synthesized_md = str(response)
            status_log += "✅ Step 2: Global synthesized answer generated.\n"
        except Exception as e:
            synthesized_md = f"Error in SubQuestion Engine: {e}"
            status_log += f"❌ Step 2: Error in SubQuestion Engine: {e}\n"
        yield answers_md, synthesized_md, conflict_md, status_log
    else:
        status_log += "🧠 Step 2: Synthesized global answer skipped.\n"
        yield answers_md, synthesized_md, conflict_md, status_log
        
    # 3. Detect conflicts
    if run_conflicts:
        if not doc_answers:
            # Conflict detection needs per-document answers. If step 1 was skipped, run queries silently.
            status_log += "⚠️ Step 3: Conflict detection requires individual document answers. Querying documents now...\n"
            yield answers_md, synthesized_md, conflict_md, status_log
            
            answers_list = []
            for i, (doc, index) in enumerate(indexes.items(), 1):
                status_log += f"👉 [{i}/{len(indexes)}] Querying: {doc}...\n"
                yield answers_md, synthesized_md, conflict_md, status_log
                
                try:
                    qe = index.as_query_engine(similarity_top_k=3)
                    ans = str(qe.query(question))
                    doc_answers[doc] = ans
                    answers_list.append(f"### 📄 {doc}:\n{ans}\n---")
                    answers_md = "\n\n".join(answers_list)
                except Exception as e:
                    doc_answers[doc] = f"Error: {e}"
                yield answers_md, synthesized_md, conflict_md, status_log
        
        status_log += "⚠️ Step 3: Running conflict detection analysis across sources...\n"
        yield answers_md, synthesized_md, conflict_md, status_log
        
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
            status_log += "✅ Step 3: Conflict analysis completed.\n"
        except Exception as e:
            conflict_md = f"Error in Conflict Detection: {e}"
            status_log += f"❌ Step 3: Error in Conflict Detection: {e}\n"
        yield answers_md, synthesized_md, conflict_md, status_log
    else:
        status_log += "⚠️ Step 3: Conflict analysis skipped.\n"
        yield answers_md, synthesized_md, conflict_md, status_log
        
    status_log += "🎉 Done! All requested tasks finished."
    yield answers_md, synthesized_md, conflict_md, status_log

# Build Gradio UI
with gr.Blocks(title="MultiDoc RAG - Phase 4") as demo:
    gr.Markdown("# 📄 MultiDoc RAG — Conflict Detection")
    gr.Markdown("Upload PDF research documents, build vector indexes, and query them. The system will detect and flag contradictions across sources.")
    
    with gr.Row():
        upload = gr.File(label="Upload PDFs or TXT files", file_count="multiple", file_types=[".pdf", ".txt"])
        with gr.Column():
            index_btn = gr.Button("Index Documents", variant="primary")
            clear_btn = gr.Button("Clear All Documents", variant="stop")
        
    index_status = gr.Textbox(label="Status", interactive=False)
    index_btn.click(upload_and_index, inputs=upload, outputs=index_status)
    clear_btn.click(clear_documents, outputs=index_status)
    
    question = gr.Textbox(label="Your Question", placeholder="e.g. Compare the conclusions across these papers.")
    
    with gr.Row():
        run_each_chk = gr.Checkbox(label="Query each document individually", value=True)
        run_synthesized_chk = gr.Checkbox(label="Synthesize global answer (SubQuestion Engine - Slow)", value=False)
        run_conflicts_chk = gr.Checkbox(label="Run Conflict Analysis", value=True)
        
    ask_btn = gr.Button("Submit Query", variant="secondary")
    
    status_log_out = gr.Textbox(label="Execution Progress Logs", lines=5, max_lines=10, interactive=False)
    
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
            
    ask_btn.click(
        ask_question, 
        inputs=[question, run_each_chk, run_synthesized_chk, run_conflicts_chk], 
        outputs=[answers_out, synthesized_out, conflict_out, status_log_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
