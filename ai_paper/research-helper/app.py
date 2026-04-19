import os, json, re, io, base64, requests, time
from urllib.parse import quote
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from dotenv import load_dotenv
load_dotenv()

import fitz
import faiss
import numpy as np
from typing import TypedDict, List, Optional

from flask import Flask, request, jsonify, send_file, render_template
from groq import Groq
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
import concurrent.futures
from api_search import search_arxiv, search_semantic_scholar, search_openalex, search_crossref, clean_query, enrich_missing_abstracts

# =========================
# CONFIG
# =========================
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
client        = Groq(api_key=GROQ_API_KEY)
TEXT_MODEL    = "openai/gpt-oss-120b"
VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"

app          = Flask(__name__)
embed_model  = SentenceTransformer("all-MiniLM-L6-v2")
dimension    = 384
index        = faiss.IndexFlatL2(dimension)
documents: List[str] = []

# =========================
# LANGGRAPH STATE
# =========================
class PaperState(TypedDict):
    text: str
    images: List[str]
    chunks: List[str]
    summary: str
    vision: List[str]
    topic: str
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

def llm(prompt: str) -> str:
    try:
        res = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def parse_pdf(file_path):
    doc = fitz.open(file_path)
    text, images = "", []
    for i, page in enumerate(doc):
        text += page.get_text()
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_path = f"temp_{xref}.png"
            with open(img_path, "wb") as f:
                f.write(base_image["image"])
            images.append(img_path)
    return text, images

def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

def store_embeddings(chunks):
    global documents, index
    index = faiss.IndexFlatL2(dimension)
    documents = []
    embs = embed_model.encode(chunks)
    index.add(np.array(embs))
    documents.extend(chunks)

def retrieve(query, k=3):
    if not documents: return []
    emb = embed_model.encode([query])
    k = min(k, len(documents))
    _, I = index.search(np.array(emb), k)
    return [documents[i] for i in I[0] if i < len(documents)]

# Engineering publishers get a relevance bonus to surface IEEE/Springer over noise
ENGINEERING_PUBLISHERS = {
    "ieee", "springer", "elsevier", "wiley", "acm", "nature", "taylor",
    "iet", "inspec", "iospress", "hindawi", "mdpi", "sage", "emerald"
}

def _engineering_bonus(paper):
    """Return +0.10 if from a prestigious engineering publisher, else 0."""
    venue = paper.get("venue", "").lower()
    if any(pub in venue for pub in ENGINEERING_PUBLISHERS):
        return 0.10
    return 0.0

def semantic_rerank(query_summary, candidate_list):
    """Sorts papers by semantic similarity + engineering publisher bonus."""
    if not candidate_list or not query_summary:
        return candidate_list
    try:
        safe_candidates = [p for p in candidate_list if p.get('title')]
        if not safe_candidates: return []
        
        texts = [f"{p['title']} {p.get('summary', '')}" for p in safe_candidates]
        query_emb = embed_model.encode([query_summary])[0]
        candidate_embs = embed_model.encode(texts)
        
        norm_q = np.linalg.norm(query_emb)
        for i, p in enumerate(safe_candidates):
            emb = candidate_embs[i]
            norm_e = np.linalg.norm(emb)
            base_score = np.dot(query_emb, emb) / (norm_q * norm_e) if (norm_q > 0 and norm_e > 0) else 0
            # Engineering bonus: prestigious publishers score higher
            p["relevance_score"] = float(base_score) + _engineering_bonus(p)
            
        safe_candidates.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        top_score = safe_candidates[0].get("relevance_score", 0)
        dynamic_threshold = max(0.15, min(0.6, top_score * 0.75))
        
        filtered = [p for p in safe_candidates if p.get("relevance_score", 0) >= dynamic_threshold]
        if len(filtered) < 5:
            # Threshold was too strict — return top-5 by score regardless
            return safe_candidates[:5]
        return filtered
    except Exception as e:
        print(f"Reranking error: {e}")
        return candidate_list[:6]

# Modular search functions moved to api_search.py

# =========================
# LANGGRAPH NODES
# =========================
def node_summarize(state: PaperState) -> PaperState:
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

def node_vision(state: PaperState) -> PaperState:
    results = []
    for img_path in state["images"][:3]:
        try:
            b64 = encode_image(img_path)
            res = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": "Analyze this research paper figure using markdown:\n- **Type**: What kind of figure?\n- **Key Insights**: What does it show?\n- **Importance**: Why is it significant?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]}]
            )
            results.append(res.choices[0].message.content)
        except Exception as e:
            results.append(f"Vision error: {e}")
    state["vision"] = results
    return state

def node_extract_topic(state: PaperState) -> PaperState:
    """Step 1: Extract high-density keywords for engineering-grade search."""
    prompt = f"""Extract 3-4 high-density technical keywords from this paper summary.
Focus on core methodology and technical domain (e.g., 'Amplitude Modulation', 'Deep Learning').
Return ONLY keywords separated by commas.

Summary:
{state['summary']}"""
    raw = llm(prompt).strip()
    state["topic"] = raw
    return state

def node_arxiv_search(state: PaperState) -> PaperState:
    """Step 3 & 4: Parallel Multi-Engine Search with Tiered Fallback."""
    raw_keywords = state["topic"]
    cleaned = clean_query(raw_keywords)
    keywords = [k.strip() for k in cleaned.split(',') if k.strip()]
    
    if not keywords:
        state["papers"] = []
        return state

    def run_tiered_arxiv(k_list):
        # Tier 1: Bold Boolean Search with Category Locking (CS fallback)
        q = "+AND+".join([f"all:{k.replace(' ', '+')}" for k in k_list])
        results = search_arxiv(q, sort_by="submittedDate")
        if results: return results
        # Tier 2: Relevance fallback
        q_simple = "+AND+".join([f"all:{k.replace(' ', '+')}" for k in k_list[:2]])
        return search_arxiv(q_simple, sort_by="relevance")

    # Step 4: Run Engines in Parallel
    query_str = " ".join(keywords)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        f_arxiv = executor.submit(run_tiered_arxiv, keywords)
        f_s2    = executor.submit(search_semantic_scholar, query_str)
        f_oa    = executor.submit(search_openalex, query_str)
        f_cr    = executor.submit(search_crossref, query_str)
        
        arxiv_p = f_arxiv.result()
        s2_p    = f_s2.result()
        oa_p    = f_oa.result()
        cr_p    = f_cr.result()

    # Step 6: Deduplication by Title Slug + Abstract Quality Gate
    all_raw = arxiv_p + s2_p + oa_p + cr_p
    seen_slugs = {}  # slug -> paper (keep best version)
    
    for p in all_raw:
        slug = re.sub(r'[^a-z0-9]', '', p['title'].lower())
        if not slug:
            continue
        # If we've seen this paper, keep the version with a real abstract
        if slug in seen_slugs:
            existing = seen_slugs[slug]
            if not existing.get("has_abstract") and p.get("has_abstract"):
                seen_slugs[slug] = p  # upgrade to version with abstract
        else:
            seen_slugs[slug] = p
    
    # Step 7: Abstract Enrichment — fetch missing abstracts via DOI/title lookup
    # This fixes Crossref papers that are paywalled and don't return abstracts
    all_unique = list(seen_slugs.values())
    enriched = enrich_missing_abstracts(all_unique)
    
    # TIERED QUALITY GATE: First prefer papers with real abstracts.
    # But if that yields too few, fall back to papers with any summary text
    # (prevents the pipeline from returning only 1 paper due to over-filtering)
    top_tier = [p for p in enriched if p.get("has_abstract")]
    
    if len(top_tier) >= 4:
        # Enough high-quality papers — use them
        unique_candidates = top_tier
    elif top_tier:
        # Top-tier is thin — supplement with papers that have any summary
        second_tier = [p for p in enriched 
                       if not p.get("has_abstract") and p.get("summary", "").strip()]
        unique_candidates = top_tier + second_tier
    else:
        # Last resort: any paper with a non-empty summary
        unique_candidates = [p for p in enriched if p.get("summary", "").strip()]
    
    if not unique_candidates:
        unique_candidates = enriched[:5]  # absolute fallback

    # Step 8: Final Selection (Top 8 — more papers = richer comparison)
    reranked = semantic_rerank(state["summary"], unique_candidates)
    state["papers"] = reranked[:8]
    return state

def node_compare(state: PaperState) -> PaperState:
    if not state.get("papers") or (len(state["papers"]) == 1 and state["papers"][0]["title"] == "Search failed"):
        state["comparison"] = "Deeply sorry, but I couldn't retrieve related papers from arXiv at this moment. This can happen if their service is temporarily down or if the extracted topic is too specific. You can try editing the search topic manually and running the comparison again."
        return state

    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary']}" for p in state["papers"] if p.get('has_abstract')])
    
    # Count papers with actual content vs without
    papers_with_data = [p for p in state["papers"] if p.get('has_abstract')]
    papers_without = [p for p in state["papers"] if not p.get('has_abstract')]
    
    skip_notice = ""
    if papers_without:
        skip_names = ", ".join([f'"{p["title"]}"' for p in papers_without])
        skip_notice = f"\n\nNOTE: The following papers were found but their full abstracts are not publicly available (behind paywall/login): {skip_names}. Do NOT include these in your analysis — only analyze papers for which abstract text is provided above."
    
    prompt = f"""You are a research analyst. Compare the original paper with recent research using markdown.

IMPORTANT RULES:
- ONLY analyze papers for which abstract/summary text is provided below.
- Do NOT write "Not Reported" or "No specific data" for any paper. If a paper lacks enough information, simply exclude it from the analysis.
- Every paper you mention MUST have concrete evidence from its abstract.

Original Paper Summary:
{state['summary']}

Recent Related Research (with abstracts):
{combined}{skip_notice}

## Key Differences
How does the original differ from recent work?

## Improvements in Recent Work
What have newer papers improved upon?

## Missing Ideas
What concepts from recent research are absent in the original?

## Strengths of Original
What does the original do well?"""
    state["comparison"] = llm(prompt)
    return state

def node_improve(state: PaperState) -> PaperState:
    """Identify weak sections and explain what needs improvement."""
    full_text = state["text"]
    prompt = f"""You are an expert research advisor. Read this paper and identify sections that need improvement.

Paper text:
{full_text[:6000]}

Comparative analysis with recent work:
{state['comparison']}

For each section that needs improvement, explain specifically what is weak and what should be changed.
Use markdown with section headings like ## Abstract, ## Introduction, ## Methodology, ## Results, ## Conclusion."""
    state["improvements"] = llm(prompt)
    return state

def node_rewrite_sections(state: PaperState) -> PaperState:
    """Rewrite each weak section. Return structured edits with original and rewritten text."""
    full_text = state["text"]

    prompt = f"""You are rewriting weak sections of a research paper to improve quality.

Full paper text:
{full_text[:8000]}

Improvement analysis:
{state['improvements']}

Your task: For each section that needs improvement, provide the rewritten version.

Output a JSON array ONLY (no other text, no markdown fences):
[
  {{
    "section": "Abstract",
    "original": "copy the FIRST 150 characters of that section EXACTLY as they appear in the paper text above",
    "rewritten": "the complete improved text for this entire section"
  }}
]

CRITICAL rules:
- "original" must be copied CHARACTER FOR CHARACTER from the paper text (it is used to find and replace the text)
- Keep "original" short (100-150 chars) — just enough to uniquely identify the location
- "rewritten" should be the full improved section text
- Only include sections whose text you can find verbatim in the paper"""

    raw = llm(prompt)

    edits = []
    try:
        # Strip markdown code fences if present
        raw = re.sub(r'```(?:json)?', '', raw).strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            edits = json.loads(match.group())
    except Exception:
        edits = []

    state["edits"] = edits
    return state

def node_qa(state: PaperState) -> PaperState:
    chunks = retrieve(state["query"])
    if not chunks:
        state["answer"] = "No document loaded. Please upload a paper first."
        return state
    context = "\n".join(chunks)
    prompt = f"""You are a research assistant. Answer based on the paper context using markdown.

Context:
{context}

Question: {state['query']}

Answer:"""
    state["answer"] = llm(prompt)
    return state

# =========================
# LANGGRAPH PIPELINES
# =========================
def build_upload_graph():
    g = StateGraph(PaperState)
    g.add_node("summarize", node_summarize)
    g.add_node("vision", node_vision)
    g.add_node("extract_topic", node_extract_topic)
    g.set_entry_point("summarize")
    g.add_edge("summarize", "vision")
    g.add_edge("vision", "extract_topic")
    g.add_edge("extract_topic", END)
    return g.compile()

def build_compare_graph():
    g = StateGraph(PaperState)
    g.add_node("arxiv_search", node_arxiv_search)
    g.add_node("compare", node_compare)
    g.set_entry_point("arxiv_search")
    g.add_edge("arxiv_search", "compare")
    g.add_edge("compare", END)
    return g.compile()

def build_improve_graph():
    g = StateGraph(PaperState)
    g.add_node("improve", node_improve)
    g.add_node("rewrite_sections", node_rewrite_sections)
    g.set_entry_point("improve")
    g.add_edge("improve", "rewrite_sections")
    g.add_edge("rewrite_sections", END)
    return g.compile()

def build_qa_graph():
    g = StateGraph(PaperState)
    g.add_node("qa", node_qa)
    g.set_entry_point("qa")
    g.add_edge("qa", END)
    return g.compile()

upload_graph  = build_upload_graph()
compare_graph = build_compare_graph()
improve_graph = build_improve_graph()
qa_graph      = build_qa_graph()

# =========================
# PDF BUILDER
# =========================
PAGE_W, PAGE_H = 595, 842
MARGIN = 55
TW = PAGE_W - 2 * MARGIN   # text width

def _strip_md(text):
    text = re.sub(r'^#{1,4}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    return text.strip()

class PageWriter:
    """Helper that writes text to fitz pages with auto page-break."""
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
        """Render markdown text block with basic formatting."""
        for raw in md_text.split('\n'):
            raw = raw.rstrip()
            if not raw:
                self.y += 5
                continue
            # Heading
            if re.match(r'^#{1,4} ', raw):
                text = re.sub(r'^#{1,4} ', '', raw)
                self.write_line(_strip_md(text), fontsize=fontsize+1,
                                bold=True, color=(0.15,0.25,0.65))
                self.y += 2
            # Bullet
            elif re.match(r'^[\-\*\+] ', raw.strip()):
                text = "•  " + _strip_md(raw.strip()[2:])
                self.write_wrapped(text, fontsize=fontsize,
                                   color=(0.1,0.1,0.1), indent=8)
            # Numbered list
            elif re.match(r'^\d+\. ', raw.strip()):
                self.write_wrapped(_strip_md(raw.strip()), fontsize=fontsize,
                                   color=(0.1,0.1,0.1), indent=8)
            else:
                self.write_wrapped(_strip_md(raw), fontsize=fontsize,
                                   color=(0.15,0.15,0.15))

    def cover_page(self, title, subtitle, topic):
        """Write a styled cover/divider page."""
        self._new_page()
        # Top accent bar
        self.page.draw_rect(fitz.Rect(0, 0, PAGE_W, 6), color=(0.2,0.3,0.75), fill=(0.2,0.3,0.75))
        self.y = PAGE_H // 3
        self.write_line(title, fontsize=20, bold=True,
                        color=(0.2,0.3,0.75), indent=(TW - len(title)*11)//2)
        self.y += 6
        self.write_divider(color=(0.2,0.3,0.75))
        self.write_wrapped(subtitle, fontsize=11, color=(0.4,0.4,0.4))
        if topic:
            self.y += 8
            self.write_wrapped(f"Topic: {topic}", fontsize=10, color=(0.5,0.5,0.5))
        self.write_line("Generated by openai/gpt-oss-120b via Groq + LangGraph",
                        fontsize=8, color=(0.6,0.6,0.6))
        # Bottom bar
        self.page.draw_rect(fitz.Rect(0, PAGE_H-6, PAGE_W, PAGE_H),
                             color=(0.2,0.3,0.75), fill=(0.2,0.3,0.75))


def build_analysis_pdf(original_pdf_path, summary, vision, comparison,
                       improvements, edits, papers, topic, qa_text=""):
    """
    Build the final PDF:
      - Original paper pages (with in-place text replacements applied)
      - Page: Summary
      - Page: Q&A placeholder
      - Page: Comparative Study
      - Page: Improvements & Rewrites
    """
    # ---- Apply text replacements to original PDF ----
    orig = fitz.open(original_pdf_path)
    applied_edits = []

    for edit in edits:
        original_snip = edit.get("original", "").strip()
        rewritten     = edit.get("rewritten", "").strip()
        section       = edit.get("section", "")
        if not original_snip or not rewritten:
            continue

        found = False
        # Try progressively shorter search keys
        for key_len in [120, 80, 50, 30]:
            search_key = original_snip[:key_len].strip()
            if not search_key:
                continue
            for page in orig:
                hits = page.search_for(search_key)
                if not hits:
                    continue
                rect = hits[0]

                # Estimate how many lines the original text spans
                line_count = max(3, len(original_snip) // 70)
                # Build replacement rect: full width of text column, enough height
                rep_rect = fitz.Rect(
                    rect.x0 - 1,
                    rect.y0 - 2,
                    page.rect.width - MARGIN + 10,
                    rect.y0 + line_count * 13 + 10
                ) & page.rect

                # White-out original text
                page.add_redact_annot(rep_rect, fill=(1, 1, 1))
                page.apply_redactions()

                # Re-insert rewritten text word-wrapped into same rect
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

    # ---- Build analysis appendix ----
    analysis = fitz.open()
    pw = PageWriter(analysis)

    # --- PAGE 1: Summary ---
    pw.cover_page("SECTION 1 — PAPER SUMMARY", "AI-generated structured summary of the uploaded paper", topic)
    pw._new_page()
    pw.write_section_heading("Paper Summary")
    pw.write_markdown_block(summary)

    if vision:
        pw.write_section_heading("Figure Analysis")
        for i, v in enumerate(vision):
            pw.write_line(f"Figure {i+1}", fontsize=10, bold=True, color=(0.8,0.5,0.1))
            pw.write_markdown_block(v)
            pw.y += 6

    # --- PAGE 2: Q&A ---
    pw.cover_page("SECTION 2 — Q&A", "Questions and answers from the interactive session", topic)
    pw._new_page()
    pw.write_section_heading("Q&A Session")
    if qa_text:
        for block in qa_text.split("---"):
            block = block.strip()
            if not block:
                continue
            lines = block.split("\n\n", 1)
            q_line = lines[0].replace("Q: ", "").strip() if lines else ""
            a_text = lines[1].replace("A: ", "").strip() if len(lines) > 1 else ""
            pw.write_line(f"Q: {q_line}", fontsize=10, bold=True, color=(0.9, 0.6, 0.1))
            pw.y += 2
            pw.write_markdown_block(a_text)
            pw.y += 8
    else:
        pw.write_wrapped("No questions were asked during this session.", fontsize=10, color=(0.4,0.4,0.4))

    # --- PAGE 3: Comparative Study ---
    pw.cover_page("SECTION 3 — COMPARATIVE STUDY", "Comparison with recent arXiv research on the same topic", topic)
    pw._new_page()
    pw.write_section_heading("Related Papers Found on arXiv")
    for p in papers:
        pw.write_line(f"[{p['year']}] {p['title']}", fontsize=9, bold=True, color=(0.2,0.3,0.7))
        pw.write_wrapped(p['summary'][:300], fontsize=9, color=(0.4,0.4,0.4))
        pw.y += 4

    pw.write_section_heading("Comparative Analysis")
    pw.write_markdown_block(comparison)

    # --- PAGE 4: Improvements & Rewrites ---
    pw.cover_page("SECTION 4 — IMPROVEMENTS & REWRITES",
                  "AI-identified weak sections and their rewritten versions", topic)
    pw._new_page()
    pw.write_section_heading("Improvement Analysis")
    pw.write_markdown_block(improvements)

    if applied_edits:
        pw.write_section_heading("Rewritten Sections (applied to paper)")
        for ed in applied_edits:
            pw.write_line(f"Section: {ed['section']}", fontsize=10, bold=True, color=(0.1,0.5,0.3))
            pw.write_markdown_block(ed['rewritten'])
            pw.y += 8
    else:
        pw.write_wrapped(
            "Note: Automatic in-place replacement could not locate the exact text snippets "
            "in the PDF (this can happen with scanned or complex-layout PDFs). "
            "The rewritten sections are shown below for manual reference.",
            fontsize=10, color=(0.7,0.4,0.1))
        pw.y += 8
        for ed in edits:
            if ed.get("rewritten"):
                pw.write_line(f"Section: {ed.get('section','')}", fontsize=10,
                              bold=True, color=(0.1,0.5,0.3))
                pw.write_markdown_block(ed["rewritten"])
                pw.y += 8

    # ---- Merge original (edited) + analysis ----
    orig.insert_pdf(analysis)
    buf = io.BytesIO()
    orig.save(buf, garbage=4, deflate=True)
    orig.close()
    analysis.close()
    buf.seek(0)
    return buf

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("upload.html", page="upload")

@app.route("/qa")
def page_qa():
    return render_template("qa.html", page="qa")

@app.route("/compare")
def page_compare():
    return render_template("compare.html", page="compare")

@app.route("/improve")
def page_improve():
    return render_template("improve.html", page="improve")

@app.route("/download")
def page_download():
    return render_template("download.html", page="download")

@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files["file"]
        path = "paper.pdf"
        file.save(path)
        text, images = parse_pdf(path)
        chunks = chunk_text(text)
        store_embeddings(chunks)
        init: PaperState = {
            "text": text, "images": images, "chunks": chunks,
            "summary": "", "vision": [], "topic": "",
            "papers": [], "comparison": "", "improvements": "",
            "edits": [], "query": "", "answer": "", "error": None
        }
        result = upload_graph.invoke(init)
        return jsonify({"summary": result["summary"], "vision": result["vision"],
                        "topic": result["topic"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    try:
        query = request.json.get("query", "")
        if not query:
            return jsonify({"error": "No query provided"}), 400
        init: PaperState = {
            "text": "", "images": [], "chunks": [],
            "summary": "", "vision": [], "topic": "",
            "papers": [], "comparison": "", "improvements": "",
            "edits": [], "query": query, "answer": "", "error": None
        }
        result = qa_graph.invoke(init)
        return jsonify({"answer": result["answer"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compare", methods=["POST"])
def compare():
    try:
        summary = request.json.get("summary", "")
        topic   = request.json.get("topic", "")
        init: PaperState = {
            "text": "", "images": [], "chunks": [],
            "summary": summary, "vision": [], "topic": topic,
            "papers": [], "comparison": "", "improvements": "",
            "edits": [], "query": "", "answer": "", "error": None
        }
        result = compare_graph.invoke(init)
        return jsonify({"papers": result["papers"], "comparison": result["comparison"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/improve", methods=["POST"])
def improve():
    try:
        summary    = request.json.get("summary", "")
        comparison = request.json.get("comparison", "")
        full_text  = ""
        if os.path.exists("paper.pdf"):
            doc = fitz.open("paper.pdf")
            for page in doc:
                full_text += page.get_text()
            doc.close()
        init: PaperState = {
            "text": full_text, "images": [], "chunks": [],
            "summary": summary, "vision": [], "topic": "",
            "papers": [], "comparison": comparison, "improvements": "",
            "edits": [], "query": "", "answer": "", "error": None
        }
        result = improve_graph.invoke(init)
        return jsonify({"improvements": result["improvements"], "edits": result["edits"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate-pdf", methods=["POST"])
def generate_pdf():
    try:
        data         = request.json
        summary      = data.get("summary", "")
        vision       = data.get("vision", [])
        comparison   = data.get("comparison", "")
        improvements = data.get("improvements", "")
        edits        = data.get("edits", [])
        papers       = data.get("papers", [])
        topic        = data.get("topic", "Research Paper")
        qa_text      = data.get("qa_text", "")

        if not os.path.exists("paper.pdf"):
            return jsonify({"error": "Original PDF not found."}), 400

        buf = build_analysis_pdf("paper.pdf", summary, vision, comparison,
                                 improvements, edits, papers, topic, qa_text)
        return send_file(buf, mimetype="application/pdf",
                         as_attachment=True, download_name="paper_analyzed.pdf")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)