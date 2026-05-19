# MultiDoc RAG — Conflict Detection Across Documents

A Retrieval-Augmented Generation (RAG) system that queries multiple files independently, synthesizes a global response, and automatically flags factual contradictions (conflicts) between sources.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10+-green?style=flat-square)
![Google Gemini](https://img.shields.io/badge/LLM-Gemini%203.5%20Flash-purple?style=flat-square)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange?style=flat-square)

---

## 🔍 Core Features

- **Sequential Multi-Folder Uploads:** Upload files sequentially one-by-one from different folders. The app accumulates them in the workspace rather than overwriting previous uploads.
- **Synchronized Workspace File List:** Real-time visual tracking of documents showing file names, sizes, and indexing status (🟢 **Indexed** vs 🟡 *Pending Indexing*).
- **Direct Workspace Deletion:** Delete specific files using native **"X" close buttons** next to files, or wipe the whole workspace using the **Clear Workspace** button.
- **Tabbed Results Layout:** Results are organized into three high-contrast dashboard tabs:
  1. *Answers per Document* (independent retrievals).
  2. *Synthesized Final Answer* (global summary).
  3. *Conflict Analysis* (detailed factual contradictions table).
- **Single-Document Protection Constraint:** Disables conflict analysis when only 1 document is uploaded to avoid redundant LLM calls, returning a clean notification.
- **Real-Time Indexing Progress:** Displays a visual loading spinner and progress bar during indexing operations via Gradio's progress tracking.

---

## 🛠️ Technology Stack & Architectures

| Component | Technology | Description |
|---|---|---|
| **RAG Framework** | LlamaIndex | Orchestrates data loaders, index creation, and retrieval tools. |
| **LLM** | Google Gemini 3.5 Flash | Handles sub-query execution and conflict reasoning. |
| **Embeddings** | Google Gemini Embedding | Vectorizes text chunks (`models/gemini-embedding-001`). |
| **Query Engine** | SubQuestionQueryEngine | Splits a complex query into targeted sub-questions for individual files. |
| **User Interface** | Gradio | Responsive bright light-mode theme with glassmorphic layouts. |

---

## 💡 Core Techniques & Algorithms

### 1. Sub-Question Fact Comparison
Instead of blending all text chunks into a single prompt (which dilutes source attribution), the system builds a separate `VectorStoreIndex` for each document. 
* The **`SubQuestionQueryEngine`** splits the user query into sub-questions (e.g. *"What does Document A say about X?"* and *"What does Document B say about X?"*).
* It queries the indices independently, preserving clean factual claims and clear source attribution.

### 2. LLM Conflict Detection Prompting
The independent answers are fed into a dedicated prompt chain. The LLM is instructed to act as a strict fact-checker and format conflicts in a standardized notation:
```
CONFLICT_FOUND: yes/no
EXPLANATION: <reconciled summary of the contradiction>
DOCUMENT_A: <first contradicting doc>
DOCUMENT_B: <second contradicting doc>
CLAIM_A: <factual statement in Doc A>
CLAIM_B: <contradictory factual statement in Doc B>
```
If a conflict is detected, the Gradio frontend parses this block and renders a warning panel alongside a side-by-side comparison table.

### 3. API Quota Rate-Limiting (Pacing Monkeypatch)
The Google Gemini Free Tier restricts API requests to **5 Requests Per Minute (RPM)**. Because RAG pipelines fire multiple queries and embeddings concurrently, this limit is easily breached.

We implemented a **Python descriptor-binding monkeypatch** in [rate_limiter.py](file:///home/kanishkagarwal/Documents/MultiDocRag/rag/rate_limiter.py):
- Intercepts LlamaIndex `Gemini` client calls (`complete`, `chat`, `acomplete`, `achat`).
- Enforces a rolling **12-second delay** between requests.
- Uses `__get__(self, self.__class__)` binding to correctly route calls to the original underlying LlamaIndex wrappers, bypassing signature validation exceptions (`missing a required argument: 'messages'`).

---

## 🚀 How to Setup and Run

### 1. Prepare Environment
Ensure you have Python 3.10+ installed. Clone this repository, create a virtual environment, and install dependencies:
```bash
# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configure API Key
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Launch the Application
Run the Gradio server:
```bash
python app.py
```
Open `http://localhost:7861` (or the printed port) in your browser.
