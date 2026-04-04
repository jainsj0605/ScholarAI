# ScholarAI 🎓

ScholarAI is a powerful Multimodal Retrieval-Augmented Generation (RAG) system designed to help researchers, students, and academics interact with, analyze, compare, and improve research papers.

Built with Python, Flask, Groq, and Sentence-Transformers, ScholarAI allows you to upload PDF research papers and intuitively extract valuable insights from them.

## ✨ Features
- **Smart PDF Uploads:** Easily upload and parse complex academic PDF documents.
- **Interactive Q&A:** Ask questions about specific sections of a research paper and get precise, context-aware answers.
- **Cross-Paper Comparison:** Upload multiple research papers and perform critical side-by-side comparative analysis.
- **Paper Improvement:** Get actionable feedback and rewriting suggestions for abstracts, methodologies, and conclusions.

## 🛠️ Tech Stack
- **Backend Framework:** Flask (Python)
- **LLM Provider:** Groq
- **Document Parsing:** PyMuPDF
- **Embeddings:** Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector Database:** FAISS CPU
- **Agentic Workflows:** LangGraph

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed on your system.

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/jainsj0605/ScholarAI.git
   cd ScholarAI
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\Activate.ps1
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory and add your Groq API Key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

5. **Run the Application**
   ```bash
   python app.py
   ```
   *The application will be accessible at `http://127.0.0.1:5000`.*

## 🔒 Security Note
Do not commit your `.env` file containing your API keys to GitHub. Ensure `.env` is listed in your `.gitignore`.
