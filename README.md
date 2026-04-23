# 🔬 ScholarAI: Research Paper Helper
**Live Deployment:** [https://scholaraii.streamlit.app/](https://scholaraii.streamlit.app/)

ScholarAI is a Multimodal Retrieval-Augmented Generation (RAG) system designed to transform how research papers are analyzed and understood. Instead of relying on traditional keyword-based search, it uses semantic embeddings with FAISS to retrieve contextually relevant information and generate structured insights. The system processes full research content, enabling automated summarization, question answering, and cross-paper comparison through workflow orchestration using LangGraph.

A key feature of ScholarAI is its multimodal capability, where it extracts and interprets figures and charts, converting visual data into meaningful textual explanations. Additionally, it includes a rewrite and improvement module that refines weak sections of research papers, enhances clarity, and provides actionable feedback while preserving the original technical intent. By combining semantic retrieval, structured reasoning, and content enhancement, ScholarAI delivers more accurate, efficient, and comprehensive research insights compared to traditional academic analysis tools.

## 🚀 Features

- **Multi-Page Web App**: A sleek, dark-mode web interface for traditional browser-based navigation.
- **AI-Powered Summarization**: Uses high-capacity models (GPT-OSS-120B) to extract structured TLDR, Problem, Method, and Results.
- **Vision Figure Analysis**: Automatically extracts and analyzes paper figures using Vision-Llama models.
- **ArXiv Comparative Study**:
  - **Tiered Search**: Uses smart boolean queries to find the most relevant papers.
  - **Relevance Validation**: AI-driven scoring to filter out noise and focus on quality benchmarks.
  - **Domain Identification**: Automatically classifies research fields (e.g., Computer Vision, AI).
- **RAG-based Q&A**: Grounded question answering using local FAISS vector embeddings of the uploaded paper.
- **Direct PDF Rewriting**: Identifies weak sections and applies AI-improved rewrites directly back into the original PDF.

## 🛠️ Tech Stack

- **Large Language Models**: Groq Cloud (GPT-OSS-120B, Llama 3.1 70B/8B, Llama 4 Scout).
- **Orchestration**: LangGraph for building robust, state-controlled AI workflows.
- **Backend/UI**: Flask (Web Development).
- **Document Processing**: PyMuPDF (fitz) for PDF parsing and in-place text replacement.
- **Vector Search**: Sentence-Transformers & FAISS (local embeddings).

## 📦 Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jainsj0605/ScholarAI.git
   cd ScholarAI/research-helper
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**:
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_actual_key_here
   ```

## 🏃 Running the Application

Launch the traditional multi-page app:
```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000`.

## 📄 License
MIT License
