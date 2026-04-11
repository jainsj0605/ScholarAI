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
# PAGE CONFIG (must be first)
# =========================
st.set_page_config(page_title="ScholarAI", page_icon="🔬", layout="wide")

# =========================
# API KEY CHECK
# =========================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("""
    ⚠️ **GROQ_API_KEY is not set.**

    To fix this on Streamlit Cloud:
    1. Go to your app → click ⚙️ **Settings** (top right)
    2. Click **Secrets** tab
    3. Add: `GROQ_API_KEY = "your_key_here"`
    4. Click **Save** → **Reboot app**

    Get your free key at [console.groq.com](https://console.groq.com)
    """)
    st.stop()

client        = Groq(api_key=GROQ_API_KEY)
TEXT_MODEL    = "openai/gpt-oss-120b"
FALLBACK_MODEL = "llama-3.3-70b-versatile"
FAST_MODEL    = "llama-3.1-8b-instant"
VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"

# =========================
# CONFIG / CONSTANTS
# =========================
PAGE_W, PAGE_H = 595, 842
MARGIN = 55
TW = PAGE_W - 2 * MARGIN

ARXIV_CATEGORIES = {
    "cs.AI": "Artificial Intelligence", "cs.CV": "Computer Vision",
    "cs.LG": "Machine Learning", "cs.CL": "Computation & Language",
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
# CACHED RESOURCES
# =========================
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()
dimension = 384

# =========================
# SESSION STATE INIT
# =========================
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = faiss.IndexFlatL2(dimension)
    st.session_state.documents = []

for key in ["summary", "comparison", "improvements", "text", "pdf_path"]:
    if key not in st.session_state:
        st.session_state[key] = ""

for key in ["vision", "topic", "papers", "edits", "images", "chunks", "qa_history"]:
    if key not in st.session_state:
        st.session_state[key] = []

# =========================
# LANGGRAPH STATE
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

def llm(prompt: str, model: str = None) -> str:
    if model is None:
        model = TEXT_MODEL
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
                    if match:
                        wait_time = float(match.group(1)) + 0.5
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

def search_arxiv(query):
    # Legacy logic: Clean up query and use strict Boolean AND search
    cleaned_query = re.sub(r'^(Topic|Keywords|Search):\s*', '', query, flags=re.IGNORECASE).strip()
    cleaned_query = re.sub(r'^[“"‘\']*(.*?)[”"’\']*$', r'\1', cleaned_query).strip()
    if not cleaned_query: return []

    def perform_search(q_text, sort_by_date=True):
        words = [w for w in re.split(r'\s+', q_text) if w]
        if not words: return []
        q = "+AND+".join([f"all:{quote_plus(w)}" for w in words])
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

    # Layered Fallback Strategy
    # 1. Try strict search with date sorting
    results = perform_search(cleaned_query, sort_by_date=True)
    # 2. If no results, try broader fallback (fewer keywords) with date sorting
    if not results:
        words = cleaned_query.split()
        if len(words) > 3:
            results = perform_search(" ".join(words[:2]), sort_by_date=True)
    # 3. If STILL no results, try relevance for better matches
    if not results:
        results = perform_search(cleaned_query, sort_by_date=False)
    return results

def validate_relevance(summary, candidate):
    prompt = f"Paper A: {summary[:800]}\nPaper B: {candidate['title']} - {candidate['summary'][:800]}\nScore 1-10 relevance. JSON only: {{\"score\": X}}"
    res = llm(prompt, model=FAST_MODEL)
    try:
        data = json.loads(re.search(r'\{.*\}', res, re.DOTALL).group())
        return int(data.get("score", 0))
    except:
        return 0

# =========================
# PDF BUILDER
# =========================
def _strip_md(text):
    text = re.sub(r'^#{1,4}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    return text.strip()

class PageWriter:
    def __init__(self, doc):
        self.doc = doc
        self.page = None
        self.y = 0
        self._new_page()

    def _new_page(self):
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = MARGIN + 10

    def _ensure_space(self, needed=20):
        if self.y + needed > PAGE_H - MARGIN:
            self._new_page()

    def write_line(self, text, fontsize=10, bold=False, color=(0,0,0), indent=0):
        self._ensure_space(fontsize + 6)
        fn = "hebo" if bold else "helv"
        self.page.insert_text((MARGIN + indent, self.y), text,
                               fontsize=fontsize, fontname=fn, color=color)
        self.y += fontsize + 5

    def write_wrapped(self, text, fontsize=10, bold=False,
                      color=(0.1,0.1,0.1), indent=0, line_h=15):
        fn = "hebo" if bold else "helv"
        cpl = max(1, int((TW - indent) / (fontsize * 0.52)))
        words = text.split()
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if len(test) > cpl:
                self._ensure_space(line_h)
                self.page.insert_text((MARGIN + indent, self.y), line,
                                       fontsize=fontsize, fontname=fn, color=color)
                self.y += line_h
                line = w
            else:
                line = test
        if line:
            self._ensure_space(line_h)
            self.page.insert_text((MARGIN + indent, self.y), line,
                                   fontsize=fontsize, fontname=fn, color=color)
            self.y += line_h

    def write_divider(self, color=(0.7,0.7,0.8)):
        self._ensure_space(10)
        self.page.draw_line((MARGIN, self.y), (PAGE_W - MARGIN, self.y),
                             color=color, width=0.5)
        self.y += 8

    def write_section_heading(self, text, color=(0.2,0.3,0.75)):
        self.y += 8
        self._ensure_space(30)
        self.write_line(text, fontsize=12, bold=True, color=color)
        self.write_divider()

    def write_markdown_block(self, md_text, fontsize=10):
        for raw in md_text.split('\n'):
            raw = raw.rstrip()
            if not raw:
                self.y += 5
                continue
            if re.match(r'^#{1,4} ', raw):
                text = re.sub(r'^#{1,4} ', '', raw)
                self.write_line(_strip_md(text), fontsize=fontsize+1,
                                bold=True, color=(0.15,0.25,0.65))
                self.y += 2
            elif re.match(r'^[\-\*\+] ', raw.strip()):
                text = "•  " + _strip_md(raw.strip()[2:])
                self.write_wrapped(text, fontsize=fontsize,
                                   color=(0.1,0.1,0.1), indent=8)
            elif re.match(r'^\d+\. ', raw.strip()):
                self.write_wrapped(_strip_md(raw.strip()), fontsize=fontsize,
                                   color=(0.1,0.1,0.1), indent=8)
            else:
                self.write_wrapped(_strip_md(raw), fontsize=fontsize,
                                   color=(0.15,0.15,0.15))

    def cover_page(self, title, subtitle, topic):
        self._new_page()
        self.page.draw_rect(fitz.Rect(0, 0, PAGE_W, 6),
                             color=(0.2,0.3,0.75), fill=(0.2,0.3,0.75))
        self.y = PAGE_H // 3
        self.write_line(title, fontsize=20, bold=True,
                        color=(0.2,0.3,0.75), indent=(TW - len(title)*11)//2)
        self.y += 6
        self.write_divider(color=(0.2,0.3,0.75))
        self.write_wrapped(subtitle, fontsize=11, color=(0.4,0.4,0.4))
        if topic:
            self.y += 8
            topic_str = ", ".join(topic) if isinstance(topic, list) else topic
            self.write_wrapped(f"Topic: {topic_str}", fontsize=10, color=(0.5,0.5,0.5))
        self.write_line("Generated by ScholarAI (Groq + LangGraph)",
                        fontsize=8, color=(0.6,0.6,0.6))
        self.page.draw_rect(fitz.Rect(0, PAGE_H-6, PAGE_W, PAGE_H),
                             color=(0.2,0.3,0.75), fill=(0.2,0.3,0.75))


def build_analysis_pdf(original_pdf_path, summary, vision, comparison,
                       improvements, edits, papers, topic, qa_history=None):
    orig = fitz.open(original_pdf_path)
    applied_edits = []

    for edit in edits:
        original_snip = edit.get("original", "").strip()
        rewritten = edit.get("rewritten", "").strip()
        section = edit.get("section", "")
        if not original_snip or not rewritten:
            continue
        found = False
        for key_len in [120, 80, 50, 30]:
            search_key = original_snip[:key_len].strip()
            if not search_key:
                continue
            for page in orig:
                hits = page.search_for(search_key)
                if not hits:
                    continue
                rect = hits[0]
                line_count = max(3, len(original_snip) // 70)
                rep_rect = fitz.Rect(
                    rect.x0 - 1, rect.y0 - 2,
                    page.rect.width - MARGIN + 10,
                    rect.y0 + line_count * 13 + 10
                ) & page.rect
                page.add_redact_annot(rep_rect, fill=(1, 1, 1))
                page.apply_redactions()
                fs = 10
                x, y = rep_rect.x0 + 1, rep_rect.y0 + fs + 1
                max_chars = max(1, int(rep_rect.width / (fs * 0.52)))
                words = rewritten.split()
                line = ""
                for w in words:
                    test = (line + " " + w).strip()
                    if len(test) > max_chars:
                        if y + fs < rep_rect.y1:
                            page.insert_text((x, y), line, fontsize=fs,
                                             fontname="helv", color=(0,0,0))
                            y += fs + 3
                        line = w
                    else:
                        line = test
                if line and y + fs < rep_rect.y1:
                    page.insert_text((x, y), line, fontsize=fs,
                                     fontname="helv", color=(0,0,0))
                applied_edits.append({"section": section, "rewritten": rewritten})
                found = True
                break
            if found:
                break

    analysis = fitz.open()
    pw = PageWriter(analysis)

    # Section 1: Summary
    pw.cover_page("SECTION 1 — PAPER SUMMARY",
                  "AI-generated structured summary of the uploaded paper", topic)
    pw._new_page()
    pw.write_section_heading("Paper Summary")
    pw.write_markdown_block(summary)
    if vision:
        pw.write_section_heading("Figure Analysis")
        for i, v in enumerate(vision):
            pw.write_line(f"Figure {i+1}", fontsize=10, bold=True, color=(0.8,0.5,0.1))
            pw.write_markdown_block(v)
            pw.y += 6

    # Section 2: Q&A
    pw.cover_page("SECTION 2 — Q&A",
                  "Questions and answers from the interactive session", topic)
    pw._new_page()
    pw.write_section_heading("Q&A Session")
    if qa_history:
        for item in qa_history:
            pw.write_line(f"Q: {item['q']}", fontsize=10, bold=True, color=(0.9,0.6,0.1))
            pw.y += 2
            pw.write_markdown_block(item['a'])
            pw.y += 8
    else:
        pw.write_wrapped("No questions were asked during this session.",
                         fontsize=10, color=(0.4,0.4,0.4))

    # Section 3: Comparative Study
    pw.cover_page("SECTION 3 — COMPARATIVE STUDY",
                  "Comparison with recent arXiv research on the same topic", topic)
    pw._new_page()
    pw.write_section_heading("Related Papers Found on arXiv")
    for p in papers:
        pw.write_line(f"[{p['year']}] [{p.get('domain','')}] {p['title']}",
                      fontsize=9, bold=True, color=(0.2,0.3,0.7))
        pw.write_wrapped(p['summary'][:300], fontsize=9, color=(0.4,0.4,0.4))
        pw.y += 4
    pw.write_section_heading("Comparative Analysis")
    pw.write_markdown_block(comparison)

    # Section 4: Improvements & Rewrites
    pw.cover_page("SECTION 4 — IMPROVEMENTS & REWRITES",
                  "AI-identified weak sections and their rewritten versions", topic)
    pw._new_page()
    pw.write_section_heading("Improvement Analysis")
    pw.write_markdown_block(improvements)
    if applied_edits:
        pw.write_section_heading("Rewritten Sections (applied to paper)")
        for ed in applied_edits:
            pw.write_line(f"Section: {ed['section']}", fontsize=10,
                          bold=True, color=(0.1,0.5,0.3))
            pw.write_markdown_block(ed['rewritten'])
            pw.y += 8
    elif edits:
        pw.write_wrapped(
            "Note: Automatic in-place replacement could not locate exact text snippets. "
            "Rewritten sections are shown below for manual reference.",
            fontsize=10, color=(0.7,0.4,0.1))
        pw.y += 8
        for ed in edits:
            if ed.get("rewritten"):
                pw.write_line(f"Section: {ed.get('section','')}",
                              fontsize=10, bold=True, color=(0.1,0.5,0.3))
                pw.write_markdown_block(ed["rewritten"])
                pw.y += 8

    orig.insert_pdf(analysis)
    buf = io.BytesIO()
    orig.save(buf, garbage=4, deflate=True)
    orig.close()
    analysis.close()
    buf.seek(0)
    return buf

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
    state["papers"] = search_arxiv(state["topic"])
    return state

def node_compare(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary'][:1000]}"
                             for p in state["papers"]])
    prompt = f"""Compare the original paper with recent research. Use markdown with these sections:
## Key Differences
## Improvements in Recent Work
## Missing Ideas
## Strengths of Original

Original Paper: {state['summary'][:1500]}
Recent Research: {combined}"""
    state["comparison"] = llm(prompt)
    return state

def node_improve(state):
    prompt = f"""You are an expert research advisor. Read this paper and identify sections that need improvement.
Paper text: {state['text'][:6000]}
Comparative analysis: {state['comparison']}
Use markdown with headings like ## Abstract, ## Introduction, ## Methodology, ## Results, ## Conclusion."""
    state["improvements"] = llm(prompt)
    return state

def node_rewrite(state):
    prompt = f"""Rewrite weak sections of this research paper to improve quality.
Full paper text: {state['text'][:8000]}
Improvement analysis: {state['improvements']}

Output a JSON array ONLY (no other text, no markdown fences):
[
  {{
    "section": "Abstract",
    "original": "copy the FIRST 150 characters of that section EXACTLY as they appear in the paper text above",
    "rewritten": "the complete improved text for this entire section"
  }}
]

CRITICAL: "original" must be copied CHARACTER FOR CHARACTER from the paper text."""
    raw = llm(prompt)
    try:
        raw = re.sub(r'```(?:json)?', '', raw).strip()
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        state["edits"] = json.loads(m.group()) if m else []
    except:
        state["edits"] = []
    return state

def node_qa(state):
    chunks = retrieve(state["query"])
    if not chunks:
        state["answer"] = "No document loaded. Please upload a paper first."
        return state
    context = "\n".join(chunks)
    prompt = f"""Answer based on the paper context using markdown.
Context: {context}
Question: {state['query']}
Answer:"""
    state["answer"] = llm(prompt)
    return state

# =========================
# LANGGRAPH PIPELINES
# =========================
@st.cache_resource
def build_graphs(v="1.0.1"):
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

upload_graph, compare_graph, improve_graph, qa_graph = build_graphs(v="1.0.1")

# =========================
# STREAMLIT UI
# =========================
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .paper-card {
        background: #1a1f2e; border: 1px solid #2a3347;
        border-radius: 12px; padding: 20px; margin-bottom: 16px;
    }
    .domain-badge {
        background: #2a3a1e; color: #4caf82; padding: 2px 10px;
        border-radius: 20px; font-size: 0.75rem; display: inline-block; margin-right: 6px;
    }
    .year-badge {
        background: #1e2a3a; color: #7c9ef8; padding: 2px 10px;
        border-radius: 20px; font-size: 0.75rem; display: inline-block; margin-right: 6px;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2d5a27, #4caf82);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 1.5rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 ScholarAI: Research Paper Helper")
st.caption("Upload a PDF → Get AI Summary, Q&A, ArXiv Comparison, Improvements & Download")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("📄 Upload Paper")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and st.button("🚀 Analyze Paper", type="primary", use_container_width=True):
        with st.spinner("Parsing PDF & running AI pipeline..."):
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
            # Reset downstream results on new upload
            st.session_state.comparison = ""
            st.session_state.improvements = ""
            st.session_state.edits = []
            st.session_state.papers = []
            st.session_state.qa_history = []

        st.success("✅ Analysis complete!")

    if st.session_state.get("topic"):
        topic_str = ", ".join(st.session_state.topic) if isinstance(st.session_state.topic, list) else st.session_state.topic
        st.info(f"**Topic:** {topic_str}")

    # ---- DOWNLOAD BUTTON ----
    if st.session_state.summary and st.session_state.get("pdf_path"):
        st.divider()
        st.subheader("📥 Download Report")
        if st.button("📄 Generate Full PDF Report", use_container_width=True):
            with st.spinner("Building PDF... this takes ~10 seconds"):
                try:
                    pdf_buf = build_analysis_pdf(
                        original_pdf_path=st.session_state.pdf_path,
                        summary=st.session_state.summary,
                        vision=st.session_state.vision,
                        comparison=st.session_state.comparison or "(Run Compare tab first)",
                        improvements=st.session_state.improvements or "(Run Improve tab first)",
                        edits=st.session_state.edits,
                        papers=st.session_state.papers,
                        topic=st.session_state.topic,
                        qa_history=st.session_state.qa_history
                    )
                    st.session_state.pdf_buf = pdf_buf.getvalue()
                    st.success("PDF ready!")
                except Exception as e:
                    st.error(f"PDF error: {e}")

        if st.session_state.get("pdf_buf"):
            st.download_button(
                label="⬇️ Download paper_analyzed.pdf",
                data=st.session_state.pdf_buf,
                file_name="paper_analyzed.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# =========================
# MAIN TABS
# =========================
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
        for item in st.session_state.qa_history:
            with st.chat_message("user"):
                st.write(item["q"])
            with st.chat_message("assistant"):
                st.markdown(item["a"])

        query = st.chat_input("Ask anything about the paper...")
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
            with st.spinner("Searching ArXiv & validating relevance... (30-60s)"):
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
            st.subheader(f"📚 {len(st.session_state.papers)} Related Papers Found")
            for p in st.session_state.papers:
                link_html = f'<a href="{p["link"]}" target="_blank" style="color:#7c9ef8">View on ArXiv →</a>' if p.get("link") else ""
                st.markdown(f"""
<div class="paper-card">
    <span class="year-badge">{p['year']}</span>
    <span class="domain-badge">{p.get('domain', 'Research')}</span>
    <br><strong style="color:#e0e0e0">{p['title']}</strong>
    <p style="color:#999;font-size:0.85rem;margin-top:8px">{p['summary'][:300]}...</p>
    {link_html}
</div>""", unsafe_allow_html=True)

        if st.session_state.comparison:
            st.divider()
            st.subheader("📊 Comparative Analysis")
            st.markdown(st.session_state.comparison)

# --- TAB 4: Improve ---
with tab4:
    if not st.session_state.comparison:
        st.info("Run the **Compare** tab first to generate the comparative analysis.")
    else:
        if st.button("✏️ Analyze & Rewrite Sections", type="primary"):
            with st.spinner("Identifying weak sections & generating rewrites... (30-60s)"):
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
