# 🔬 ScholarAI: Research Paper Helper
**Live Deployment:** [https://scholaraii.streamlit.app/](https://scholaraii.streamlit.app/)

ScholarAI is an advanced, AI-driven research pipeline that significantly outperforms standard literature review tools by orchestrating a deterministic multi-engine search across 4 major academic databases (ArXiv, Semantic Scholar, OpenAlex, and CrossRef). It guarantees high-fidelity outputs through a strict 2-level filtering gatekeeper that enforces a 100-character minimum abstract length and immediately discards generic metadata placeholders. Unlike traditional systems that rely solely on brief abstracts, ScholarAI dynamically resolves DOIs and downloads open-access PDFs to extract up to 8,000 characters of full-text methodology and results data per paper. By algorithmically reconstructing complex API data formats and applying vector-based semantic re-ranking to isolate the top 6 most relevant papers, ScholarAI eliminates the common issue of "Not Reported" data gaps—ultimately generating a comprehensive, 5-dimensional comparative analysis backed by empirical evidence, exact dataset scales, and robust technical methodologies.

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
