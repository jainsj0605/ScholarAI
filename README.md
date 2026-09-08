# ScholarAI 🎓

ScholarAI is a multimodal Retrieval-Augmented Generation (RAG) assistant that analyzes, critiques, compares, and suggests publication-grade enhancements for research papers. 

Powered by **Groq** (`openai/gpt-oss-120b` with automatic fallback to `llama-3.3-70b-versatile`, plus `meta-llama/llama-4-scout-17b-16e-instruct` for vision), **Sentence-Transformers**, **FAISS**, **PyMuPDF**, and **LangGraph**, ScholarAI extracts deep insights from academic PDFs, retrieves related literature across four research repositories, critiques weak sections, and provides actionable, publication-ready suggested rewrites.

---

## ✨ Features & Current Capabilities

- 📄 **Smart PDF Parsing & Multimodal Extraction**  
  Extracts document text and figures using PyMuPDF. Analyzes visual diagrams, charts, and architectural figures using LLaMA-4 Scout Vision.

- 📋 **Full-Coverage Semantic Summarization**  
  Applies targeted RAG queries across the entire paper (abstract, methodology, experimental results, conclusions) rather than blind character truncation. Generates fact-grounded summaries with strict rules against generic filler.

- 💬 **Vector-Grounded Interactive Q&A**  
  Chunk-level embeddings via `all-MiniLM-L6-v2` stored in an in-memory FAISS L2 vector index. Provides precise, context-bounded answers to user questions.

- 🔍 **Multi-Engine Literature Retrieval & Reranking**  
  Extracts high-density keywords and executes parallel searches across:
  - **ArXiv** (Boolean keyword queries with submission date sorting)
  - **Semantic Scholar** (Graph API)
  - **OpenAlex** (Inverted-index abstract reconstruction)
  - **CrossRef** (Metadata & DOI discovery)  
  Enriches missing abstracts via automated DOI lookups and reranks candidates with an engineering publisher bonus (+0.10 for IEEE, Springer, ACM, Elsevier, Nature).

- 📊 **Automated Comparative Study**  
  Compares the uploaded paper against top-ranked literature, synthesizing key differences, recent improvements in competing papers, missing concepts, and original strengths.

- ✏️ **Publication Critique & Suggested Section Rewrites**  
  Identifies weak or vague statements, quotes original snippets from the paper, provides concrete critique, and presents publication-grade suggested rewrites side-by-side for review.

- 📥 **Comprehensive Analysis Report Export**  
  Compiles a downloadable multi-page analytical report (PDF) including the paper summary, figure insights, interactive Q&A history, comparative study, and the complete critique with recommended rewrites for the author's reference.

- 🛡️ **429 Rate-Limit Fallback Chain**  
  Automatically falls back from `openai/gpt-oss-120b` to `llama-3.3-70b-versatile` if rate limits are reached, ensuring uninterrupted analysis.

---

## 🏗️ Repository Architecture

```
ScholarAI/
├── run.py                 # Application launcher (`python run.py`)
├── requirements.txt       # Production dependencies
├── .env.example           # Environment template
├── .gitignore             # Git ignore configuration
├── README.md              # Project documentation
├── src/                   # Core application package
│   ├── __init__.py
│   ├── app.py             # Flask application & route handlers
│   ├── config.py          # Central configuration & model fallback settings
│   ├── llm.py             # Groq client & 429 rate-limit fallback handler
│   ├── rag.py             # PDF text/image parser, FAISS vector index & retrieval
│   ├── search.py          # Multi-engine search (ArXiv, S2, OpenAlex, Crossref) & reranking
│   ├── workflows.py       # LangGraph state machine pipelines (upload, compare, improve, qa)
│   ├── pdf_builder.py     # PDF report synthesis, layout formatting, and appendix compilation
│   └── templates/         # Modern dark-themed Jinja2 templates
│       ├── base.html      # Base navigation & state persistence
│       ├── upload.html    # Drag-and-drop PDF dropzone & tabbed summary
│       ├── qa.html        # Vector RAG Q&A interface
│       ├── compare.html   # Literature discovery & comparative study view
│       ├── improve.html   # Critique & rewrite side-by-side viewer
│       └── download.html  # Export center for full analysis & edited PDF
└── tests/                 # Automated test suite
    ├── __init__.py
    ├── test_rag.py        # Chunking & FAISS semantic retrieval tests
    ├── test_routes.py     # Flask route integration tests
    └── test_search.py     # Search sanitization & venue normalization tests
```

---

## 🛠️ Tech Stack

- **Web Framework:** Flask
- **LLM & Vision Provider:** Groq (`openai/gpt-oss-120b`, `llama-3.3-70b-versatile`, `meta-llama/llama-4-scout-17b-16e-instruct`)
- **Orchestration:** LangGraph
- **Vector Search & Embeddings:** FAISS CPU, Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Document Processing:** PyMuPDF (`fitz`)
- **External Search APIs:** ArXiv API, Semantic Scholar API, OpenAlex API, CrossRef API

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 to 3.14
- A [Groq API Key](https://console.groq.com/)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/jainsj0605/ScholarAI.git
   cd ScholarAI
   ```

2. **Create & Activate a Virtual Environment**
   ```bash
   # On Windows (PowerShell):
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # On macOS/Linux:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   Copy the `.env.example` template:
   ```bash
   # Windows:
   copy .env.example .env

   # Linux / macOS:
   cp .env.example .env
   ```
   Add your Groq API key in `.env`:
   ```env
   GROQ_API_KEY=gsk_your_groq_api_key_here
   FLASK_DEBUG=True
   PORT=5000
   ```

5. **Run the Application**
   ```bash
   python run.py
   ```
   Open `http://127.0.0.1:5000` in your web browser.

---

## 🧪 Running Tests

ScholarAI includes unit and integration tests covering vector retrieval, search sanitization, venue normalization, and web endpoints:

```bash
python -m unittest discover -s tests -v
```

---

## 🔒 Security & Best Practices

- **Never commit your `.env` file** containing your API keys to version control. It is ignored by `.gitignore`.
- Temporary upload artifacts (`paper.pdf` and `temp_*.png`) are automatically excluded from git tracking.
