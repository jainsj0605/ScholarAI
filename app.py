import os, json, re, io, base64, requests, time, tempfile
from urllib.parse import quote_plus

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import fitz
import faiss
import numpy as np
from typing import TypedDict, List, Optional

from groq import Groq
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

# =========================
# CONFIG
# =========================
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
client        = Groq(api_key=GROQ_API_KEY)
TEXT_MODEL    = "openai/gpt-oss-120b"
FALLBACK_MODEL = "llama-3.3-70b-versatile"
FAST_MODEL    = "llama-3.1-8b-instant"
VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"

# Cache the heavy model so it loads only once
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()
dimension    = 384

# Session-level FAISS index
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(dimension)
    st.session_state.documents = []

# ArXiv category mapping
ARXIV_CATEGORIES = {
    "cs.AI": "Artificial Intelligence", "cs.CV": "Computer Vision",
    "cs.LG": "Machine Learning", "cs.CL": "Computation and Language",
    "cs.NE": "Neural/Evolutionary Computing", "cs.RO": "Robotics",
    "stat.ML": "Machine Learning", "cs.CR": "Cryptography",
    "cs.IR": "Information Retrieval", "cs.SE": "Software Engineering",
    "cs.DC": "Distributed Computing", "cs.HC": "Human-Computer Interaction",
}

def get_domain_name(cat_code):
    if not cat_code: return "Research"
    if cat_code in ARXIV_CATEGORIES: return ARXIV_CATEGORIES[cat_code]
    prefix = cat_code.split('.')[0]
    if prefix == "cs": return "Computer Science"
    if prefix == "stat": return "Statistics"
    if "physics" in cat_code or cat_code.startswith("hep-"): return "Physics"
    if "math" in cat_code: return "Mathematics"
    return cat_code.upper()

# =========================
# STATE
# =========================
class PaperState(TypedDict):
    text: str
    images: List[str]
    chunks: List[str]
    summary: str
    vision: List[str]
    topic: List[str]
    papers: List[dict]
    comparison: str
    comparison_table: str
    comp_arch: str
    comp_opt: str
    comp_bench: str
    comp_innov: str
    improvements: str
    edits: List[dict]
    query: str
    answer: str
    error: Optional[str]

# =========================
# UTILS
# =========================
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def llm(prompt: str, model: str = TEXT_MODEL, max_chars: int = 24000) -> str:
    # Truncate prompt to prevent 413 or TPM errors
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars] + "\n\n[Context truncated due to size limits...]"
    
    current_model = model
    try:
        # Initial Attempt
        res = client.chat.completions.create(
            model=current_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000, # Reduced to leave room for input tokens
            temperature=0.3  # Slightly lower for more precision
        )
        content = res.choices[0].message.content
        # Failsafe: Strip accidental instruction leakage
        content = re.sub(r'^<<< SYSTEM INSTRUCTIONS >>>', '', content, flags=re.MULTILINE).strip()
        content = re.sub(r'^### .* ###', '', content, flags=re.MULTILINE).strip()
        
        # Hard Failsafe: Ensure it ends on a full sentence
        if not content.strip().endswith(('.', '!', '?', ']', '\"', '\'')):
            last_period = max(content.rfind('.'), content.rfind('!'), content.rfind('?'))
            if last_period != -1:
                content = content[:last_period + 1] + "\n\n[Section complete]"
        return content
    except Exception as e:
        # Fallback Logic: Only switch if the primary hits a rate limit
        err_msg = str(e).lower()
        if "429" in err_msg or "limit" in err_msg or "413" in err_msg:
            try:
                # When falling back, use a smaller max_chars if it was a 413
                fallback_limit = 10000 if "413" in err_msg else max_chars
                res = client.chat.completions.create(
                    model=FALLBACK_MODEL,
                    messages=[{"role": "user", "content": prompt[:fallback_limit]}],
                    max_tokens=2000,
                    temperature=0.3
                )
                return res.choices[0].message.content
            except Exception as e2:
                return f"Error (Both Models Busy/Limited): {e2}"
        return f"Error: {e}"

def distill_context(context: str) -> str:
    """Extracts critical technical points including Architecture, Optimization, and Innovation."""
    if not context or "Error" in context: return "No comparative data available."
    
    distilled = []
    
    # Mapping of headers to labels and character caps
    sections = [
        (r'## 2\.1 Architectural Delta(.*?)(?=##|$)', "ARCHITECTURAL GAPS", 1500),
        (r'## 2\.2 Methodology & Objectives(.*?)(?=##|$)', "METHODOLOGY SHORTFALLS", 1200),
        (r'## 2\.3 Benchmark Parity(.*?)(?=##|$)', "BENCHMARK COMPARISONS", 1200),
        (r'## 2\.4 Innovation Uniqueness(.*?)(?=##|$)', "NOVELTY ANALYSIS", 1200)
    ]
    
    for pattern, label, cap in sections:
        match = re.search(pattern, context, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            if content:
                distilled.append(f"[{label}]\n{content[:cap]}")
        
    if not distilled:
        return context[:3000] # Fallback to first 3k chars if parsing fails
        
    return "\n\n".join(distilled)

def parse_pdf(file_path):
    doc = fitz.open(file_path)
    text, images = "", []
    for page in doc:
        text += page.get_text()
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_path = os.path.join(tempfile.gettempdir(), f"temp_{xref}.png")
            with open(img_path, "wb") as f:
                f.write(base_image["image"])
            images.append(img_path)
    return text, images

def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

def store_embeddings(chunks):
    st.session_state.faiss_index = faiss.IndexFlatL2(dimension)
    st.session_state.documents = []
    embs = embed_model.encode(chunks)
    st.session_state.faiss_index.add(np.array(embs))
    st.session_state.documents.extend(chunks)

def retrieve(query, k=3):
    docs = st.session_state.documents
    idx = st.session_state.faiss_index
    if not docs: return []
    emb = embed_model.encode([query])
    k = min(k, len(docs))
    _, I = idx.search(np.array(emb), k)
    return [docs[i] for i in I[0] if i < len(docs)]

def search_semantic_scholar(query):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&limit=5&fields=title,abstract,year,url,venue"
    try:
        res = requests.get(url, timeout=12)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("data", []):
                papers.append({
                    "title": item.get("title", "Untitled"),
                    "summary": item.get("abstract") or "No abstract available.",
                    "year": str(item.get("year", "")),
                    "link": item.get("url", ""),
                    "venue": item.get("venue") or "Semantic Scholar"
                })
            return papers
    except: pass
    return []

def search_openalex(query):
    url = f"https://api.openalex.org/works?search={requests.utils.quote(query)}&limit=5"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("results", []):
                papers.append({
                    "title": item.get("display_name", "Untitled"),
                    "summary": (item.get("abstract_inverted_index") or "No abstract.").strip()[:500],
                    "year": str(item.get("publication_year", "")),
                    "link": item.get("doi") or f"https://openalex.org/{item.get('id').split('/')[-1]}",
                    "venue": (item.get("primary_location") or {}).get("source", {}).get("display_name", "OpenAlex")
                })
            return papers
    except: pass
    return []

def search_crossref(query):
    url = f"https://api.crossref.org/works?query={requests.utils.quote(query)}&rows=5"
    try:
        res = requests.get(url, timeout=12)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("message", {}).get("items", []):
                papers.append({
                    "title": item.get("title", ["Untitled"])[0],
                    "summary": "Engineering research record found in CrossRef. No abstract available via metadata API.",
                    "year": str(item.get("published-print", {}).get("date-parts", [[""]])[0][0]),
                    "link": item.get("URL", ""),
                    "venue": item.get("container-title", ["CrossRef"])[0]
                })
            return papers
    except: pass
    return []

def search_arxiv(query):
    cleaned_query = re.sub(r'^(Topic|Keywords|Search):\s*', '', query, flags=re.IGNORECASE).strip()
    cleaned_query = re.sub(r'^[“"‘\']*(.*?)[”"’\']*$', r'\1', cleaned_query).strip()
    if not cleaned_query: return []

    def perform_search(q_text, sort_by_date=True):
        words = [w for w in re.split(r'\s+', q_text) if w]
        if not words: return []
        q = "+AND+".join([f"all:{quote_plus(w)}" for w in words])
        url = f"https://export.arxiv.org/api/query?search_query={q}&start=0&max_results=5"
        if sort_by_date: url += "&sortBy=submittedDate&sortOrder=descending"
        else: url += "&sortBy=relevance"
        try:
            res = requests.get(url, timeout=25)
            papers = []
            if res.status_code == 200:
                entries = re.findall(r'<entry>(.*?)</entry>', res.text, re.DOTALL)
                for entry in entries:
                    t_m = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                    s_m = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                    id_m = re.search(r'<id>(.*?)</id>', entry, re.DOTALL)
                    p_m = re.search(r'<published>(.*?)</published>', entry, re.DOTALL)
                    if t_m and s_m:
                        papers.append({
                            "title": re.sub(r'\s+', ' ', t_m.group(1)).strip(),
                            "summary": re.sub(r'\s+', ' ', s_m.group(1)).strip(),
                            "year": p_m.group(1)[:4] if p_m else "",
                            "link": id_m.group(1).strip() if id_m else "",
                            "venue": "ArXiv"
                        })
            return papers
        except: return []

    words = [w for w in re.split(r'\s+', cleaned_query) if w]
    for count in [len(words), 3, 2]:
        if count > len(words): continue
        results = perform_search(" ".join(words[:count]), sort_by_date=True)
        if results: return results
    return perform_search(cleaned_query, sort_by_date=False)

# =========================
# LANGGRAPH NODES
# =========================
def node_summarize(state):
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Reviewer
CONSTRAINTS: Provide a MASSIVE TECHNICAL ANALYSIS. Avoid brief bullet points.
Use exhaustive analytical paragraphs to describe architecture, methodology, and theoretical foundations.
MANDATORY: Detail specific decimal scores and mathematical components found in the text.

## Executive Summary
(Exhaustive high-level analysis of impact and novelty)

## Problem & Motivation
(Deep dive into the research gap and dataset challenges)

## Architecture & Methodology
(Granular description of backbones, attention mechanisms, loss functions, and inference blocks)

## Theoretical Contributions
(Detailed analysis of novel theorems, proofs, or conceptual shifts)

## Results & Benchmarks
(Exhaustive numerical results on every dataset mentioned, including delta comparison)

## Limitations
(Technical constraints, edge cases, and hardware requirements)

### PAPER TEXT DATA ###
{state['text'][:8000]}
"""
    state["summary"] = llm(prompt)
    return state

def node_vision(state):
    results = []
    for img_path in state["images"][:3]:
        try:
            b64 = encode_image(img_path)
            res = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": "Analyze this research paper figure:\n- **Type**: What kind?\n- **Key Insights**: What does it show?\n- **Importance**: Why significant?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]}]
            )
            results.append(res.choices[0].message.content)
        except Exception as e:
            results.append(f"Vision error: {e}")
    state["vision"] = results
    return state

def node_extract_topic(state):
    prompt = f"Extract the main research topic (3-4 essential terms) from this summary. Return ONLY the keywords separated by spaces. No quotes, no preamble, and no symbols.\n\nSummary:\n{state['summary']}"
    state["topic"] = llm(prompt).strip().replace('"', '').replace("'", "")
    return state

def node_arxiv_search(state):
    query = state["topic"]
    if not query or "Error:" in query:
        state["papers"] = []
        return state
        
    arxiv_p = search_arxiv(query)
    crossref_p = search_crossref(query)
    openalex_p = search_openalex(query)
    semantic_p = search_semantic_scholar(query)
    
    all_p = crossref_p + openalex_p + semantic_p + arxiv_p
    unique = []
    seen = set()
    for p in all_p:
        slug = re.sub(r'[^a-z0-9]', '', p['title'].lower())
        if slug and slug not in seen:
            unique.append(p)
            seen.add(slug)
    state["papers"] = unique[:6]
    return state

def node_compare_table(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']} ({p.get('venue','Research')}): {p['summary'][:1500]}" for p in state["papers"]])
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Reviewer
TASK: Generate a COMPACT QUANTITATIVE COMPARISON TABLE.
METRICS: Architecture, Dataset, Accuracy/mIoU/F1, Parameters.
LIMIT: Max 5 rows. Return ONLY the table.

### CONTEXT ###
Original Summary: {state['summary'][:1500]}
Context: {combined}

## 1. Quantitative Comparison Table
| Architecture | Dataset | Accuracy/mIoU/F1 | Parameters |
| :--- | :--- | :--- | :--- |"""
    state["comparison_table"] = llm(prompt)
    return state

def node_compare_arch(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary'][:1500]}" for p in state["papers"]])
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Reviewer | TASK: 2.1 Architectural Delta
CONSTRAINTS: Provide a MASSIVE TECHNICAL DEEP DIVE. Avoid brief bullets.
Identify exactly how the original paper's architecture (backbone, attention, fusion, etc.) differs from each related work.
MANDATORY: Mention related papers by title or year in your analysis.

### CONTEXT ###
Original: {state['summary'][:2000]}
Related Research: {combined}

## 2.1 Architectural Delta
(Exhaustive technical analysis with specific paper citations)"""
    state["comp_arch"] = llm(prompt)
    return state

def node_compare_opt(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary'][:1500]}" for p in state["papers"]])
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Reviewer | TASK: 2.2 Methodology & Objectives
CONSTRAINTS: Provide an EXHAUSTIVE COMPARISON of research methodologies, mathematical objectives, or training strategies.
MANDATORY: Detail specific objectives (e.g., Loss functions, physical constraints, or experimental protocols) and how this paper's approach differs from the competitors.
MANDATORY: Cite related papers by title/year.

### CONTEXT ###
Original: {state['summary'][:2000]}
Related Research: {combined}

## 2.2 Methodology & Objectives
(Deep methodological and strategic comparison)
"""
    state["comp_opt"] = llm(prompt)
    return state

def node_compare_bench(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary'][:1500]}" for p in state["papers"]])
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Reviewer | TASK: 2.3 Benchmark Parity
CONSTRAINTS: Provide a GRANULAR NUMERICAL COMPARISON.
Compare this paper's performance on SHARED DATASETS (e.g., COD10K, NC4K) vs. each related paper.
MANDATORY: Use specific decimal scores and cite the papers providing those scores.

### CONTEXT ###
Original: {state['summary'][:2000]}
Related Research: {combined}

## 2.3 Benchmark Parity
(Numerical comparison and fairness analysis)"""
    state["comp_bench"] = llm(prompt)
    return state

def node_compare_innov(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary'][:1500]}" for p in state["papers"]])
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Reviewer | TASK: 2.4 Innovation Uniqueness
CONSTRAINTS: Provide an EXHAUSTIVE CONCEPTUAL ANALYSIS.
Isolate the "First-of-its-kind" novelties versus incremental improvements from prior work. 
MANDATORY: Explicitly contrast against the specific related papers provided.

### CONTEXT ###
Original: {state['summary'][:2000]}
Related Research: {combined}

## 2.4 Innovation Uniqueness
(Categorical analysis of novelty vs prior art)"""
    state["comp_innov"] = llm(prompt)
    return state

def node_improve(state):
    # DISTILL context to avoid TPM limits and focus the LLM
    important_context = distill_context(state['comparison'])
    
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Senior Technical Editor
TASK: Identify exactly 3-5 SPECIFIC weak sections in the paper that need technical improvement.

CONSTRAINTS:
- AVOID GENERIC FEEDBACK (e.g., "improve clarity", "add more detail").
- FOCUS ON TECHNICAL GAPS: Lack of specific architectural justifications, missing benchmark comparisons, weak methodological grounding (Optimization, Loss functions, or core Research Objectives), or incremental novelty vs total innovation.
- REFERENCE the Comparative Analysis directly (Architectural, Methodology/Optimization, Benchmarks, Innovation).
- Start your response DIRECTLY with '## Improvement Strategy'.

### DISTILLED COMPARATIVE CONTEXT ###
{important_context}

### RELEVANT PAPER SNIPPETS ###
{state['text'][:5000]}

## Improvement Strategy
"""
    state["improvements"] = llm(prompt)
    return state

def node_rewrite(state):
    if "Error" in state["improvements"]:
        state["edits"] = []
        return state

    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Author
TASK: Rewrite the weak sections identified to address technical gaps (architectural detail, benchmark context).
CONSTRAINTS:
- Return a VALID JSON array ONLY.
- Format: [{{"section": "Precise Section Name", "original": "Original text snippet", "rewritten": "Improved technical text"}}]
- Ensure section names are unique and match the paper's structure.
- Maximum 5 rewrites.

### ANALYSIS OF WEAKNESSES ###
{state['improvements']}

### FULL PAPER CONTEXT (TRUNCATED) ###
{state['text'][:6000]}

### OUTPUT ###
"""
    raw = llm(prompt)
    try:
        raw = re.sub(r'```(?:json)?', '', raw).strip()
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            edits = json.loads(m.group())
            # Basic validation to ensure they aren't all the same
            unique_edits = []
            seen_sections = set()
            for ed in edits:
                sec = ed.get("section", "General")
                if sec not in seen_sections:
                    unique_edits.append(ed)
                    seen_sections.add(sec)
            state["edits"] = unique_edits
        else:
            state["edits"] = []
    except:
        state["edits"] = []
    return state

def node_qa(state):
    chunks = retrieve(state["query"])
    if not chunks:
        state["answer"] = "No document loaded."
        return state
    context = "\n".join(chunks)
    prompt = f"""Answer based on paper context using markdown.
Context: {context}
Question: {state['query']}
Answer:"""
    state["answer"] = llm(prompt)
    return state

# =========================
# LANGGRAPH PIPELINES
# =========================
@st.cache_resource
def build_graphs():
    g1 = StateGraph(PaperState)
    g1.add_node("summarize", node_summarize)
    g1.add_node("vision", node_vision)
    g1.add_node("extract_topic", node_extract_topic)
    g1.set_entry_point("summarize")
    g1.add_edge("summarize", "vision")
    g1.add_edge("vision", "extract_topic")
    g1.add_edge("extract_topic", END)

    g2 = StateGraph(PaperState)
    g2.add_node("arxiv_search", node_arxiv_search)
    g2.add_node("compare_table", node_compare_table)
    g2.add_node("compare_arch", node_compare_arch)
    g2.add_node("compare_opt", node_compare_opt)
    g2.add_node("compare_bench", node_compare_bench)
    g2.add_node("compare_innov", node_compare_innov)
    g2.set_entry_point("arxiv_search")
    g2.add_edge("arxiv_search", "compare_table")
    g2.add_edge("compare_table", "compare_arch")
    g2.add_edge("compare_arch", "compare_opt")
    g2.add_edge("compare_opt", "compare_bench")
    g2.add_edge("compare_bench", "compare_innov")
    g2.add_edge("compare_innov", END)

    g3 = StateGraph(PaperState)
    g3.add_node("improve", node_improve)
    g3.add_node("rewrite", node_rewrite)
    g3.set_entry_point("improve")
    g3.add_edge("improve", "rewrite")
    g3.add_edge("rewrite", END)

    g4 = StateGraph(PaperState)
    g4.add_node("qa", node_qa)
    g4.set_entry_point("qa")
    g4.add_edge("qa", END)

    return g1.compile(), g2.compile(), g3.compile(), g4.compile()

upload_graph, compare_graph, improve_graph, qa_graph = build_graphs()

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="ScholarAI", page_icon="🔬", layout="wide")

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .paper-card {
        background: #1a1f2e; border: 1px solid #2a3347; border-radius: 12px;
        padding: 20px; margin-bottom: 16px;
    }
    .domain-badge {
        background: #2a3a1e; color: #4caf82; padding: 2px 10px;
        border-radius: 20px; font-size: 0.75rem; display: inline-block;
    }
    .year-badge {
        background: #1e2a3a; color: #7c9ef8; padding: 2px 10px;
        border-radius: 20px; font-size: 0.75rem; display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 ScholarAI: Research Paper Helper")
st.caption("Upload a PDF → Get AI Summary, Q&A, Multi-Engine Comparison, and Improvements")

# Initialize session state
for key in ["summary", "vision", "topic", "papers", "comparison", "comparison_table",
            "comp_arch", "comp_opt", "comp_bench", "comp_innov",
            "improvements", "edits", "text", "images", "chunks", "qa_history"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["vision", "topic", "papers", "edits",
                                                "images", "chunks", "qa_history"] else ""

# --- SIDEBAR: Upload ---
with st.sidebar:
    st.header("📄 Upload Paper")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and st.button("🚀 Analyze Paper", type="primary", use_container_width=True):
        with st.spinner("Parsing PDF & running AI analysis..."):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(uploaded_file.read())
            tmp.close()

            text, images = parse_pdf(tmp.name)
            chunks = chunk_text(text)
            store_embeddings(chunks)

            init = {
                "text": text, "images": images, "chunks": chunks,
                "summary": "", "vision": [], "topic": [],
                "papers": [], "comparison": "", "improvements": "",
                "edits": [], "query": "", "answer": "", "error": None
            }
            result = upload_graph.invoke(init)

            st.session_state.text = text
            st.session_state.images = images
            st.session_state.chunks = chunks
            st.session_state.summary = result["summary"]
            st.session_state.vision = result["vision"]
            st.session_state.topic = result["topic"]
            st.session_state.pdf_path = tmp.name

        st.success("✅ Analysis complete!")

    if st.session_state.topic:
        topic_str = st.session_state.topic if isinstance(st.session_state.topic, str) else ", ".join(st.session_state.topic)
        st.info(f"**Topic:** {topic_str}")

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "💬 Q&A", "🔍 Compare", "✏️ Improve"])

# --- TAB 1: Summary ---
with tab1:
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
        if st.session_state.vision:
            st.divider()
            st.subheader("🖼️ Figure Analysis")
            for i, v in enumerate(st.session_state.vision):
                with st.expander(f"Figure {i+1}", expanded=(i == 0)):
                    st.markdown(v)
    else:
        st.info("👈 Upload a PDF in the sidebar to get started.")

# --- TAB 2: Q&A ---
with tab2:
    if not st.session_state.text:
        st.info("Upload a paper first to ask questions.")
    else:
        query = st.chat_input("Ask anything about the paper...")
        for item in st.session_state.qa_history:
            with st.chat_message("user"):
                st.write(item["q"])
            with st.chat_message("assistant"):
                st.markdown(item["a"])

        if query:
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    init = {
                        "text": "", "images": [], "chunks": [],
                        "summary": "", "vision": [], "topic": [],
                        "papers": [], "comparison": "", "improvements": "",
                        "edits": [], "query": query, "answer": "", "error": None
                    }
                    result = qa_graph.invoke(init)
                    st.markdown(result["answer"])
                    st.session_state.qa_history.append({"q": query, "a": result["answer"]})

# --- TAB 3: Compare ---
with tab3:
    if not st.session_state.summary:
        st.info("Upload a paper first.")
    else:
        if st.button("🔍 Run Comparative Study", type="primary"):
            with st.spinner("Searching multi-engine academic sources... (30-60s)"):
                init = {
                    "text": "", "images": [], "chunks": [],
                    "summary": st.session_state.summary, "vision": [],
                    "topic": st.session_state.topic,
                    "papers": [], "comparison": "", "improvements": "",
                    "edits": [], "query": "", "answer": "", "error": None
                }
                result = compare_graph.invoke(init)
                st.session_state.papers = result["papers"]
                st.session_state.comparison_table = result["comparison_table"]
                st.session_state.comp_arch = result["comp_arch"]
                st.session_state.comp_opt = result["comp_opt"]
                st.session_state.comp_bench = result["comp_bench"]
                st.session_state.comp_innov = result["comp_innov"]

        if st.session_state.papers:
            st.subheader("📚 Related Papers Found")
            for p in st.session_state.papers:
                st.markdown(f"""
<div class="paper-card">
    <span class="year-badge">{p['year']}</span>
    <span class="domain-badge">{p.get('venue', 'Academic Source')}</span>
    <br><strong style="color:#7c9ef8">{p['title']}</strong>
    <p style="color:#999;font-size:0.85rem">{p['summary'][:300]}...</p>
    {'<a href="' + p["link"] + '" target="_blank">View Source →</a>' if p.get("link") else ''}
</div>""", unsafe_allow_html=True)

        if st.session_state.comparison_table:
            st.divider()
            st.subheader("📊 1. Quantitative Comparison Table")
            st.markdown(st.session_state.comparison_table)

        if st.session_state.comp_arch:
            st.divider()
            st.subheader("📊 2. Technical Deep Dive")
            st.markdown("### 2.1 Architectural Delta")
            st.markdown(st.session_state.comp_arch)
            
        if st.session_state.comp_opt:
            st.markdown("### 2.2 Methodology & Objectives")
            st.markdown(st.session_state.comp_opt)
            
        if st.session_state.comp_bench:
            st.markdown("### 2.3 Benchmark Parity")
            st.markdown(st.session_state.comp_bench)
            
        if st.session_state.comp_innov:
            st.markdown("### 2.4 Innovation Uniqueness")
            st.markdown(st.session_state.comp_innov)

# --- TAB 4: Improve ---
with tab4:
    if not st.session_state.comp_arch:
        st.info("Run the Comparative Study first (Compare tab).")
    else:
        if st.button("✏️ Analyze & Rewrite Sections", type="primary"):
            with st.spinner("Identifying weak sections & generating rewrites..."):
                # Combine modular results for the improvement engine
                combined_comparison = f"""
                {st.session_state.comp_arch}
                {st.session_state.comp_opt}
                {st.session_state.comp_bench}
                {st.session_state.comp_innov}
                """
                init = {
                    "text": st.session_state.text, "images": [], "chunks": [],
                    "summary": st.session_state.summary, "vision": [],
                    "topic": st.session_state.topic,
                    "papers": [], "comparison": combined_comparison,
                    "improvements": "", "edits": [],
                    "query": "", "answer": "", "error": None
                }
                result = improve_graph.invoke(init)
                st.session_state.improvements = result["improvements"]
                st.session_state.edits = result["edits"]

        if st.session_state.improvements:
            st.subheader("📋 Improvement Analysis")
            if "Error" in st.session_state.improvements:
                st.error(st.session_state.improvements)
            else:
                st.markdown(st.session_state.improvements)

        if st.session_state.edits:
            st.divider()
            st.subheader(f"✏️ {len(st.session_state.edits)} Sections Rewritten")
            for ed in st.session_state.edits:
                title = ed.get('section', 'Unknown Section')
                with st.expander(f"Section: {title}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("ORIGINAL")
                        orig = ed.get("original", "")
                        st.text(orig[:500] + ("..." if len(orig)>500 else ""))
                    with col2:
                        st.caption("REWRITTEN ✨")
                        st.markdown(ed.get("rewritten", ""))
        elif st.session_state.improvements and "Error" not in st.session_state.improvements:
             pass # Removed "No specific technical rewrites" message as requested
