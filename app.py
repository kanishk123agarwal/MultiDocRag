import os
import concurrent.futures
import gradio as gr
from dotenv import load_dotenv
import rag.rate_limiter  # noqa: F401 — must be imported first to patch Gemini + GeminiEmbedding
from rag.indexer import index_all_documents, invalidate_cache
from rag.query_engine import build_multi_doc_engine
from rag.conflict_detector import detect_conflicts

# Load env variables (contains GOOGLE_API_KEY)
load_dotenv()

indexes = {}
multi_doc_engine = None

# Custom CSS for a beautiful, clean bright Light Theme with subtle shadows
custom_css = """
body {
    background-color: #f9fafb;
    color: #1f2937;
    font-family: 'Inter', -apple-system, sans-serif;
}
.gradio-container {
    background-color: #f9fafb !important;
    max-width: 1350px !important;
    margin: 0 auto !important;
    padding: 20px !important;
    border: none !important;
}
/* Card Panels (Bright theme with soft shadow) */
.glass-panel {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    margin-bottom: 20px !important;
}
.gradient-header {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.5rem;
    margin-bottom: 0.2rem;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #4b5563;
    margin-bottom: 2rem;
    font-size: 1.1rem;
}
/* Buttons with solid premium colors */
.accent-btn {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
.accent-btn:hover {
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3) !important;
    transform: translateY(-1px);
}
.danger-btn {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
.danger-btn:hover {
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3) !important;
    transform: translateY(-1px);
}
.secondary-btn {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: white !important;
    border: none !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
.secondary-btn:hover {
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    transform: translateY(-1px);
}
/* Input background and text overrides for high contrast in light mode */
input:not([type="checkbox"]), textarea, select {
    background-color: #ffffff !important;
    color: #1f2937 !important;
    border: 1px solid #d1d5db !important;
}
/* Checkbox fix: ensure tick marks are visible and clickable */
input[type="checkbox"] {
    appearance: auto !important;
    -webkit-appearance: checkbox !important;
    width: 18px !important;
    height: 18px !important;
    cursor: pointer !important;
    accent-color: #4f46e5 !important;
    opacity: 1 !important;
    background-color: unset !important;
    border: unset !important;
    pointer-events: auto !important;
}
.nav-tabs button.selected {
    border-bottom: 2px solid #4f46e5 !important;
    color: #4f46e5 !important;
}
"""

def get_workspace_documents_status():
    global indexes
    doc_dir = "data/docs"
    if not os.path.exists(doc_dir):
        return "📁 *Workspace is empty.*", []
        
    files = [f for f in os.listdir(doc_dir) if f.endswith(".pdf") or f.endswith(".txt")]
    if not files:
        return "📁 *Workspace is empty.*", []
        
    md = f"### 📁 Workspace Documents ({len(files)})\n"
    doc_names = []
    for f in sorted(files):
        doc_names.append(f)
        path = os.path.join(doc_dir, f)
        size_bytes = os.path.getsize(path)
        size_str = f"{size_bytes/1024:.1f} KB" if size_bytes < 1024*1024 else f"{size_bytes/(1024*1024):.1f} MB"
        
        is_indexed = f in indexes
        status_badge = "🟢 **Indexed**" if is_indexed else "🟡 *Pending Indexing*"
        md += f"- 📄 `{f}` ({size_str}) — {status_badge}\n"
        
    return md, doc_names

def get_all_workspace_file_paths():
    doc_dir = "data/docs"
    if not os.path.exists(doc_dir):
        return []
    return [os.path.join(doc_dir, f) for f in os.listdir(doc_dir) if f.endswith(".pdf") or f.endswith(".txt")]

def handle_incremental_upload(files):
    if not files:
        md, doc_names = get_workspace_documents_status()
        paths = get_all_workspace_file_paths()
        return None, paths, md, gr.update(choices=doc_names, value=None)
        
    os.makedirs("data/docs", exist_ok=True)
    for f in files:
        dest = os.path.join("data/docs", os.path.basename(f.name))
        with open(f.name, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())
            
    md, doc_names = get_workspace_documents_status()
    paths = get_all_workspace_file_paths()
    # Returns None to clear the temporary uploader box, and updates the workspace files list
    return None, paths, md, gr.update(choices=doc_names, value=None)

def handle_workspace_change(files):
    global indexes, multi_doc_engine
    doc_dir = "data/docs"
    
    # If uploader is cleared or empty
    if not files:
        if os.path.exists(doc_dir):
            for fname in os.listdir(doc_dir):
                if fname.endswith(".pdf") or fname.endswith(".txt"):
                    try:
                        os.remove(os.path.join(doc_dir, fname))
                    except Exception:
                        pass
        indexes = {}
        multi_doc_engine = None
        md, doc_names = get_workspace_documents_status()
        return [], md, gr.update(choices=[], value=None)
        
    # Get basenames of currently kept files in the display uploader
    display_basenames = [os.path.basename(f.name) for f in files]
    
    # Delete any document from workspace that was crossed out (removed from display_basenames)
    deleted_any = False
    if os.path.exists(doc_dir):
        for fname in os.listdir(doc_dir):
            if fname.endswith(".pdf") or fname.endswith(".txt"):
                if fname not in display_basenames:
                    try:
                        os.remove(os.path.join(doc_dir, fname))
                        invalidate_cache(fname)  # clear stale disk cache
                        deleted_any = True
                        if fname in indexes:
                            del indexes[fname]
                    except Exception:
                        pass
                        
    if deleted_any:
        if indexes:
            multi_doc_engine = build_multi_doc_engine(indexes)
        else:
            multi_doc_engine = None
            
    md, doc_names = get_workspace_documents_status()
    paths = get_all_workspace_file_paths()
    return paths, md, gr.update(choices=doc_names, value=None)

def delete_selected_document(doc_name):
    global indexes, multi_doc_engine
    if not doc_name:
        md, doc_names = get_workspace_documents_status()
        paths = get_all_workspace_file_paths()
        return "Please select a document to delete.", paths, md, gr.update(choices=doc_names, value=None)
        
    path = os.path.join("data/docs", doc_name)
    if os.path.exists(path):
        try:
            os.remove(path)
            invalidate_cache(doc_name)  # clear stale disk cache
        except Exception as e:
            md, doc_names = get_workspace_documents_status()
            paths = get_all_workspace_file_paths()
            return f"Error deleting file: {e}", paths, md, gr.update(choices=doc_names, value=None)
            
    if doc_name in indexes:
        del indexes[doc_name]
        
    # Rebuild query engine
    if indexes:
        multi_doc_engine = build_multi_doc_engine(indexes)
    else:
        multi_doc_engine = None
        
    md, doc_names = get_workspace_documents_status()
    paths = get_all_workspace_file_paths()
    return f"Successfully deleted '{doc_name}' from workspace.", paths, md, gr.update(choices=doc_names, value=None)

def clear_documents():
    global indexes, multi_doc_engine
    doc_dir = "data/docs"
    if os.path.exists(doc_dir):
        for fname in os.listdir(doc_dir):
            if fname.endswith(".pdf") or fname.endswith(".txt"):
                try:
                    os.remove(os.path.join(doc_dir, fname))
                    invalidate_cache(fname)  # clear stale disk cache
                except Exception:
                    pass
    indexes = {}
    multi_doc_engine = None
                    
    md, doc_names = get_workspace_documents_status()
    return "All documents cleared from workspace.", [], md, gr.update(choices=doc_names, value=None)

def index_workspace(progress=gr.Progress()):
    global indexes, multi_doc_engine
    doc_dir = "data/docs"

    progress(0.1, desc="🔍 Checking workspace directory...")
    if not os.path.exists(doc_dir):
        md, doc_names = get_workspace_documents_status()
        paths = get_all_workspace_file_paths()
        return "No documents found to index.", paths, md, gr.update(choices=doc_names, value=None)

    files = [f for f in os.listdir(doc_dir) if f.endswith(".pdf") or f.endswith(".txt")]
    if not files:
        md, doc_names = get_workspace_documents_status()
        paths = get_all_workspace_file_paths()
        return "No documents found to index.", paths, md, gr.update(choices=doc_names, value=None)

    # Check which files have a valid disk cache
    from rag.indexer import _is_cache_valid
    cached = [f for f in files if _is_cache_valid(f, os.path.join(doc_dir, f))]
    new_docs = [f for f in files if f not in cached]

    if new_docs:
        progress(0.3, desc=f"⚡ Embedding {len(new_docs)} new doc(s) via Gemini API (throttled at 5 RPM)...")
    else:
        progress(0.3, desc="⚡ Loading all documents from disk cache (no API calls)...")

    indexes = index_all_documents(doc_dir)

    progress(0.8, desc="🧠 Configuring SubQuestion Query Engine...")
    if indexes:
        multi_doc_engine = build_multi_doc_engine(indexes)

    md, doc_names = get_workspace_documents_status()
    paths = get_all_workspace_file_paths()

    if cached and new_docs:
        status_msg = f"✅ {len(cached)} loaded from cache, {len(new_docs)} newly embedded."
    elif cached:
        status_msg = f"⚡ All {len(cached)} document(s) loaded from cache instantly — no API calls used."
    else:
        status_msg = f"✅ Embedded and indexed {len(indexes)} document(s) successfully."

    return status_msg, paths, md, gr.update(choices=doc_names, value=None)

def get_workspace_documents_status_on_load():
    md, doc_names = get_workspace_documents_status()
    paths = get_all_workspace_file_paths()
    return paths, md, gr.update(choices=doc_names, value=None)

def _query_single_doc(args):
    """Worker function to query a single document index. Returns (doc_name, answer_or_error)."""
    doc, index, question = args
    try:
        qe = index.as_query_engine(similarity_top_k=6)
        ans = str(qe.query(question))
        return doc, ans, None
    except Exception as e:
        return doc, None, str(e)


def ask_question(question, run_each, run_synthesized, run_conflicts):
    global indexes, multi_doc_engine
    if not indexes:
        err_msg = "Please upload and index documents first."
        yield err_msg, err_msg, err_msg, "❌ No documents indexed. Please index documents first."
        return

    status_log = "🚦 Starting execution flow...\n"
    answers_md = "⌛ Pending run..." if run_each else "🚫 Disabled."
    synthesized_md = "⌛ Pending run..." if run_synthesized else "🚫 Disabled."
    conflict_md = "⌛ Pending run..." if run_conflicts else "🚫 Disabled."

    yield answers_md, synthesized_md, conflict_md, status_log

    doc_answers = {}

    # 1. Query each document independently (parallel)
    if run_each:
        status_log += f"📄 Step 1: Querying each of the {len(indexes)} documents in parallel...\n"
        yield answers_md, synthesized_md, conflict_md, status_log

        docs_list = list(indexes.items())
        args_list = [(doc, index, question) for doc, index in docs_list]

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(docs_list), 4)) as executor:
            futures = {executor.submit(_query_single_doc, args): args[0] for args in args_list}
            answers_list = []
            for future in concurrent.futures.as_completed(futures):
                doc, ans, err = future.result()
                if err:
                    status_log += f"   ❌ Error querying {doc}: {err}\n"
                    answers_list.append(f"### 📄 {doc}:\nError: {err}\n---")
                else:
                    doc_answers[doc] = ans
                    answers_list.append(f"### 📄 {doc}:\n{ans}\n---")
                    status_log += f"   ✅ Received answer from {doc}\n"
                answers_md = "\n\n".join(answers_list)
                yield answers_md, synthesized_md, conflict_md, status_log
    else:
        status_log += "📄 Step 1: Individual document queries skipped.\n"
        yield answers_md, synthesized_md, conflict_md, status_log

    # 2. Query using SubQuestion Query Engine
    if run_synthesized:
        status_log += "🧠 Step 2: Running SubQuestion Query Engine (synthesizing global answer)...\n"
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
        if len(indexes) < 2:
            conflict_md = "ℹ️ **Conflict analysis skipped:** At least 2 documents must be uploaded and indexed to perform conflict detection."
            status_log += "ℹ️ Step 3: Conflict detection skipped (only 1 document is indexed).\n"
            yield answers_md, synthesized_md, conflict_md, status_log
        else:
            if not doc_answers:
                # Conflict detection needs per-document answers. If step 1 was skipped, run queries in parallel.
                status_log += "⚠️ Step 3: Conflict detection requires individual answers. Querying all documents in parallel...\n"
                yield answers_md, synthesized_md, conflict_md, status_log

                docs_list = list(indexes.items())
                args_list = [(doc, index, question) for doc, index in docs_list]
                answers_list = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(docs_list), 4)) as executor:
                    futures = {executor.submit(_query_single_doc, args): args[0] for args in args_list}
                    for future in concurrent.futures.as_completed(futures):
                        doc, ans, err = future.result()
                        if err:
                            doc_answers[doc] = f"Error: {err}"
                            answers_list.append(f"### 📄 {doc}:\nError: {err}\n---")
                        else:
                            doc_answers[doc] = ans
                            answers_list.append(f"### 📄 {doc}:\n{ans}\n---")
                        status_log += f"   ✅ Queried {doc}\n"
                        answers_md = "\n\n".join(answers_list)
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
with gr.Blocks(title="MultiDoc RAG - Light Theme", css=custom_css) as demo:
    gr.HTML('<h1 class="gradient-header">📄 MultiDoc RAG — Conflict Detector</h1>')
    gr.HTML('<p class="subtitle">Upload research papers (PDF/TXT), ask questions, and detect contradictions automatically.</p>')
    
    with gr.Row():
        # Left Workspace Column (scale=4)
        with gr.Column(scale=4, elem_classes="glass-panel"):
            gr.Markdown("## 🗂️ Document Workspace")
            
            # Temporary upload slot (resets on upload)
            upload = gr.File(
                label="📁 Upload Files (Adds to workspace)",
                file_count="multiple",
                file_types=[".pdf", ".txt"]
            )
            
            # Interactive Workspace view with "X" close buttons for files
            workspace_display = gr.File(
                label="💼 Files Currently in Workspace (Click 'X' to remove)",
                file_count="multiple",
                interactive=True
            )
            
            # Current Files Status List
            uploaded_files_list = gr.Markdown("### 📁 Workspace Files Status\n*(Workspace is empty)*")
            
            # Action Buttons
            with gr.Row():
                index_btn = gr.Button("⚙️ Index Workspace", elem_classes="accent-btn")
                clear_btn = gr.Button("🗑️ Clear Workspace", elem_classes="danger-btn")
                
            index_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### 🛠️ Quick Delete Dropdown")
            with gr.Row():
                delete_dropdown = gr.Dropdown(label="Select file to delete", choices=[], interactive=True)
                delete_btn = gr.Button("❌ Delete Selected", elem_classes="danger-btn")
                
        # Right Control & Query Column (scale=7)
        with gr.Column(scale=7, elem_classes="glass-panel"):
            gr.Markdown("## 🔍 Query Interface")
            
            question = gr.Textbox(
                label="Your Question", 
                placeholder="e.g. Compare the accuracy of the proposed models across these papers.",
                lines=2
            )
            
            with gr.Row():
                run_each_chk = gr.Checkbox(label="Query each doc individually", value=True, interactive=True)
                run_synthesized_chk = gr.Checkbox(label="Synthesize global answer (Slow)", value=False, interactive=True)
                run_conflicts_chk = gr.Checkbox(label="Run Conflict Analysis", value=True, interactive=True)
                
            ask_btn = gr.Button("🚀 Submit Query", variant="secondary", elem_classes="secondary-btn")
            
            status_log_out = gr.Textbox(label="Execution Progress Logs", lines=5, max_lines=10, interactive=False)
            
            # Beautiful Tabbed Outputs
            with gr.Tabs(elem_classes="nav-tabs"):
                with gr.Tab("📄 Answers per Document"):
                    answers_out = gr.Markdown()
                with gr.Tab("🧠 Synthesized Final Answer"):
                    synthesized_out = gr.Markdown()
                with gr.Tab("⚠️ Conflict Analysis"):
                    conflict_out = gr.Markdown()

    # Event Bindings
    # 1. Sequential file uploading (Updates workspace_display and clears itself)
    upload.upload(
        handle_incremental_upload,
        inputs=upload,
        outputs=[upload, workspace_display, uploaded_files_list, delete_dropdown]
    )
    
    # 2. When user deletes a file from the workspace list via the "X" button
    workspace_display.change(
        handle_workspace_change,
        inputs=workspace_display,
        outputs=[workspace_display, uploaded_files_list, delete_dropdown]
    )
    
    # 3. File index trigger
    index_btn.click(
        index_workspace,
        outputs=[index_status, workspace_display, uploaded_files_list, delete_dropdown]
    )
    
    # 4. Clear workspace trigger
    clear_btn.click(
        clear_documents,
        outputs=[index_status, workspace_display, uploaded_files_list, delete_dropdown]
    )
    
    # 5. Individual file delete trigger
    delete_btn.click(
        delete_selected_document,
        inputs=delete_dropdown,
        outputs=[index_status, workspace_display, uploaded_files_list, delete_dropdown]
    )
    
    # 6. Page load initialization
    demo.load(
        get_workspace_documents_status_on_load,
        outputs=[workspace_display, uploaded_files_list, delete_dropdown]
    )
    
    # 7. Submit query trigger
    ask_btn.click(
        ask_question, 
        inputs=[question, run_each_chk, run_synthesized_chk, run_conflicts_chk], 
        outputs=[answers_out, synthesized_out, conflict_out, status_log_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
