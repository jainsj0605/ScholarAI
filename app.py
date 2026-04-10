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
FALLBACK_MODEL = "llama-3.1-70b-versatile"
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
    retries = 2
    current_model = model
    for attempt in range(retries + 1):
        try:
            res = client.chat.completions.create(
                model=current_model,
                messages=[{"role": "user", "content": prompt}]
            )
            return res.choices[0].message.content
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg.lower():
                if attempt < retries:
                    wait_time = 4
                    match = re.search(r'Please try again in ([\d\.]+)s', err_msg)
                    if match: wait_time = float(match.group(1)) + 0.5
                    time.sleep(wait_time)
                    continue
                elif current_model == TEXT_MODEL:
                    current_model = FALLBACK_MODEL
                    time.sleep(1)
                    try:
                        res = client.chat.completions.create(
                            model=current_model,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        return res.choices[0].message.content
                    except Exception as e2:
                        return f"Error (Fallback): {e2}"
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

def search_arxiv(keywords):
    if not keywords: return []
    def perform_search(q, sort_by_date=True):
        url = f"https://export.arxiv.org/api/query?search_query={q}&start=0&max_results=8"
        if sort_by_date: url += "&sortBy=submittedDate&sortOrder=descending"
        else: url += "&sortBy=relevance"
        try:
            res = requests.get(url, timeout=25)
            papers = []
            if res.status_code == 200:
                entries = re.findall(r'<entry>(.*?)</entry>', res.text, re.DOTALL)
                for entry in entries:
                    t = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                    s = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                    i = re.search(r'<id>(.*?)</id>', entry, re.DOTALL)
                    p = re.search(r'<published>(.*?)</published>', entry, re.DOTALL)
                    c = re.search(r'<category term="(.*?)"', entry)
                    if t and s:
                        papers.append({
                            "title": re.sub(r'\s+', ' ', t.group(1)).strip(),
                            "summary": re.sub(r'\s+', ' ', s.group(1)).strip(),
                            "year": p.group(1)[:4] if p else "",
                            "link": i.group(1).strip() if i else "",
                            "domain": get_domain_name(c.group(1)) if c else "Research"
                        })
            return papers
        except: return []
    # Tiered search
    tier1 = " AND ".join([f'all:{quote_plus(kw)}' for kw in keywords])
    tier2 = " AND ".join([f'(ti:{quote_plus(kw)} OR abs:{quote_plus(kw)})' for kw in keywords])
    specific = sorted(keywords, key=len, reverse=True)[:2]
    tier3 = " AND ".join([f'abs:{quote_plus(kw)}' for kw in specific])
    for q in [tier1, tier2, tier3]:
        for sort in [True, False]:
            results = perform_search(q, sort)
            if results: return results
    return []

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
    prompt = f"Analyze summary and extract 3-4 specific keywords. JSON list only: [\"kw1\", \"kw2\"].\nSummary: {state['summary']}"
    res = llm(prompt)
    try: state["topic"] = json.loads(re.search(r'\[.*\]', res, re.DOTALL).group())
    except: state["topic"] = [res.strip()]
    return state

def node_arxiv_search(state):
    candidates = search_arxiv(state["topic"])
    validated = []
    for p in candidates:
        if len(validated) >= 5: break
        if validate_relevance(state["summary"], p) >= 6: validated.append(p)
    state["papers"] = validated or candidates[:3]
    return state

def node_compare(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary'][:1000]}" for p in state["papers"]])
    prompt = f"""Compare original paper with recent research using markdown.
Original: {state['summary'][:1500]}
Recent: {combined}"""
    state["comparison"] = llm(prompt)
    return state

def node_improve(state):
    prompt = f"""Identify sections that need improvement.
Paper text: {state['text'][:6000]}
Comparative analysis: {state['comparison']}
Use markdown with section headings."""
    state["improvements"] = llm(prompt)
    return state

def node_rewrite(state):
    prompt = f"""Rewrite weak sections. Paper: {state['text'][:8000]}
Analysis: {state['improvements']}
Output JSON array: [{{"section":"X","original":"first 150 chars","rewritten":"improved text"}}]"""
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
