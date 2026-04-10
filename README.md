# 🔬 ScholarAI: Research Paper Helper

ScholarAI is an AI-powered assistant designed to streamline the research process. It transforms complex research papers into actionable insights through automated summarization, vision-based figure analysis, and comparative studies against recent arXiv research.

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
