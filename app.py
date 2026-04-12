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

def llm(prompt: str, model: str = TEXT_MODEL) -> str:
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.5
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

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
    """Primary Semantic Search Engine with Rate-Limit Awareness"""
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
    """Global Academic Catalog - High reliability for engineering topics"""
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
    """Engineering Meta-Registry (IEEE, Elsevier, Wiley, etc.)"""
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
    # Basic cleaning - handle "Topic:", smart quotes, and labels
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
                            "link": id_m.group(1).strip() if id_m else ""
                        })
            return papers
        except: return []

    words = [w for w in re.split(r'\s+', cleaned_query) if w]
    
    # Tier 1: Iterative AND (Date Sorted)
    for count in [len(words), 3, 2]:
        if count > len(words): continue
        results = perform_search(" ".join(words[:count]), sort_by_date=True)
        if results: return results
        
    # Tier 2: Iterative AND (Relevance Sorted)
    for count in [len(words), 2]:
        if count > len(words): continue
        results = perform_search(" ".join(words[:count]), sort_by_date=False)
        if results: return results
        
    # Tier 3: Broad OR Fallback (Relevance)
    try:
        q_or = "+OR+".join([f"all:{quote_plus(w)}" for w in words[:2]])
        url = f"https://export.arxiv.org/api/query?search_query={q_or}&max_results=5&sortBy=relevance"
        res = requests.get(url, timeout=20)
        papers = []
        if res.status_code == 200:
            entries = re.findall(r'<entry>(.*?)</entry>', res.text, re.DOTALL)
            for entry in entries:
                t_m = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                s_m = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                if t_m and s_m:
                    papers.append({
                        "title": re.sub(r'\s+', ' ', t_m.group(1)).strip(),
                        "summary": re.sub(r'\s+', ' ', s_m.group(1)).strip(),
                        "year": "", "link": ""
                    })
        return papers
    except: return []

def validate_relevance(summary, candidate):
    prompt = f"Paper A: {summary[:800]}\nPaper B: {candidate['title']} - {candidate['summary'][:800]}\nScore 1-10 on relevance. JSON only: {{\"score\": X}}"
    res = llm(prompt, model=FAST_MODEL)
    try:
        data = json.loads(re.search(r'\{.*\}', res, re.DOTALL).group())
        return int(data.get("score", 0))
    except: return 0

# =========================
# LANGGRAPH NODES
# =========================
def node_summarize(state):
    prompt = f"""Analyze this research paper and provide a structured summary using markdown:
## TLDR
One sentence summary.
## Problem
What problem does it solve?
## Method
What approach/method is used?
## Results
Key results and metrics.
## Limitations
Known limitations.

Paper text:
{state['text'][:4000]}"""
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
    
    # Run all discovery engines
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

def node_compare(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary'][:1000]}" for p in state["papers"]])
    prompt = f"""Compare original paper with recent research using markdown.
### CRITICAL: You MUST complete the 'Quick Take-Away Table' fully. Do not stop mid-generation.
Original: {state['summary'][:1500]}
Recent: {combined}

## Strategic Comparison
(Provide a deep analysis here)

## Quick Take-Away Table
| Aspect | This Paper | Recent Work | Innovation |
| :--- | :--- | :--- | :--- |"""
    state["comparison"] = llm(prompt)
    return state

def node_improve(state):
    prompt = f"""Identify sections that need improvement based on the comparative analysis.
### CRITICAL: You MUST complete the 'Discussion & Limitations' table fully.
Paper text: {state['text'][:6000]}
Comparative analysis: {state['comparison']}

## Improvement Strategy
...
## Discussion & Limitations
| Issue | Why it matters | Suggested fix |
| :--- | :--- | :--- |"""
    state["improvements"] = llm(prompt)
    return state

def node_rewrite(state):
    prompt = f"""Rewrite specific weak sections identified in the analysis. 
### CRITICAL: You MUST return a VALID JSON array. If you stop early, the app will fail.
Paper: {state['text'][:8000]}
Analysis: {state['improvements']}
Output JSON array ONLY: [{{"section":"Section Name","original":"EXACT original text snippet","rewritten":"improved text"}}]"""
    raw = llm(prompt)
    try:
        raw = re.sub(r'```(?:json)?', '', raw).strip()
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        state["edits"] = json.loads(m.group()) if m else []
    except: state["edits"] = []
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
    g2.add_node("compare", node_compare)
    g2.set_entry_point("arxiv_search")
    g2.add_edge("arxiv_search", "compare")
    g2.add_edge("compare", END)

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
st.caption("Upload a PDF → Get AI Summary, Q&A, ArXiv Comparison, and Improvements")

# Initialize session state
for key in ["summary", "vision", "topic", "papers", "comparison",
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
            # Save uploaded file temporarily
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
        topic_str = ", ".join(st.session_state.topic) if isinstance(st.session_state.topic, list) else st.session_state.topic
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
        # Display history
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
            with st.spinner("Searching ArXiv & validating relevance... (this may take 30-60s)"):
                init = {
                    "text": "", "images": [], "chunks": [],
                    "summary": st.session_state.summary, "vision": [],
                    "topic": st.session_state.topic,
                    "papers": [], "comparison": "", "improvements": "",
                    "edits": [], "query": "", "answer": "", "error": None
                }
                result = compare_graph.invoke(init)
                st.session_state.papers = result["papers"]
                st.session_state.comparison = result["comparison"]

        if st.session_state.papers:
            st.subheader("📚 Related Papers Found")
            for p in st.session_state.papers:
                st.markdown(f"""
<div class="paper-card">
    <span class="year-badge">{p['year']}</span>
    <span class="domain-badge">{p.get('domain', 'Research')}</span>
    <br><strong style="color:#7c9ef8">{p['title']}</strong>
    <p style="color:#999;font-size:0.85rem">{p['summary'][:300]}...</p>
    {'<a href="' + p["link"] + '" target="_blank">View on ArXiv →</a>' if p.get("link") else ''}
</div>""", unsafe_allow_html=True)

        if st.session_state.comparison:
            st.divider()
            st.subheader("📊 Comparative Analysis")
            st.markdown(st.session_state.comparison)

# --- TAB 4: Improve ---
with tab4:
    if not st.session_state.comparison:
        st.info("Run the Comparative Study first (Compare tab).")
    else:
        if st.button("✏️ Analyze & Rewrite Sections", type="primary"):
            with st.spinner("Identifying weak sections & generating rewrites..."):
                init = {
                    "text": st.session_state.text, "images": [], "chunks": [],
                    "summary": st.session_state.summary, "vision": [],
                    "topic": st.session_state.topic,
                    "papers": [], "comparison": st.session_state.comparison,
                    "improvements": "", "edits": [],
                    "query": "", "answer": "", "error": None
                }
                result = improve_graph.invoke(init)
                st.session_state.improvements = result["improvements"]
                st.session_state.edits = result["edits"]

        if st.session_state.improvements:
            st.subheader("📋 Improvement Analysis")
            st.markdown(st.session_state.improvements)

        if st.session_state.edits:
            st.divider()
            st.subheader(f"✏️ {len(st.session_state.edits)} Sections Rewritten")
            for ed in st.session_state.edits:
                with st.expander(f"Section: {ed.get('section', 'Unknown')}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption("ORIGINAL")
                        st.text(ed.get("original", "")[:300] + "...")
                    with col2:
                        st.caption("REWRITTEN ✨")
                        st.markdown(ed.get("rewritten", ""))
