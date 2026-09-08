import re
import json
import concurrent.futures
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from src.llm import llm, analyze_figure
from src.rag import retrieve
from src.search import (
    clean_query,
    search_arxiv,
    search_semantic_scholar,
    search_openalex,
    search_crossref,
    enrich_missing_abstracts,
    semantic_rerank,
)

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
# LANGGRAPH NODES
# =========================
def node_summarize(state: PaperState) -> PaperState:
    """Generate structured summary using RAG across all key paper sections."""
    section_queries = [
        "abstract problem statement motivation",
        "methodology approach proposed method",
        "results experiments evaluation metrics",
        "conclusion limitations future work",
    ]
    seen, context_parts = set(), []
    for q in section_queries:
        for chunk in retrieve(q, k=3):
            if chunk not in seen:
                seen.add(chunk)
                context_parts.append(chunk)
    context = "\n\n".join(context_parts)

    prompt = f"""You are a precise research analyst. Summarize this paper using ONLY facts found in the text below.

STRICT RULES — violations will make your response useless:
- NEVER write generic phrases like "The authors propose...", "This paper presents...", "The study shows..."
- ALWAYS use the EXACT method name, algorithm name, or system name from the paper (e.g. "ResNet-50", "OFDM", "BERT", "Doppler-aware power control")
- ALWAYS include EXACT numbers: accuracy %, dB gains, dataset size, parameter count, benchmark scores
- If the paper uses a specific dataset, name it exactly (e.g. "ImageNet", "KITTI", "MNIST")
- If a baseline is compared, name it exactly
- Do NOT invent or infer — only report what is literally in the text

## TLDR
One sentence. Must include: the specific problem + the exact method name + the top result with a number.

## Problem
What exact technical problem or gap does this paper address? Quote the specific claim or limitation it targets.

## Method
Name the exact proposed system/algorithm/architecture. Describe its key components using terms from the paper.

## Results
List EXACT metrics: e.g. "Achieved 94.3% accuracy on [dataset], outperforming [baseline] by 2.1%". No vague statements.

## Limitations
Quote or paraphrase the actual limitations the authors acknowledge. Do NOT invent limitations.

Paper text (sampled from full document via semantic retrieval):
{context}"""
    state["summary"] = llm(prompt)
    return state

def node_vision(state: PaperState) -> PaperState:
    """Analyze up to 3 extracted figures using multimodal vision model."""
    results = []
    for img_path in state["images"][:3]:
        results.append(analyze_figure(img_path))
    state["vision"] = results
    return state

def node_extract_topic(state: PaperState) -> PaperState:
    """Extract high-density technical keywords from summary for academic search."""
    prompt = f"""Extract 3-4 high-density technical keywords from this paper summary.
Focus on core methodology and technical domain (e.g., 'Amplitude Modulation', 'Deep Learning').
Return ONLY keywords separated by commas.

Summary:
{state['summary']}"""
    state["topic"] = llm(prompt).strip()
    return state

def node_arxiv_search(state: PaperState) -> PaperState:
    """Parallel multi-engine search (ArXiv, Semantic Scholar, OpenAlex, Crossref) with enrichment & reranking."""
    raw_keywords = state["topic"]
    cleaned = clean_query(raw_keywords)
    keywords = [k.strip() for k in cleaned.split(',') if k.strip()]

    if not keywords:
        state["papers"] = []
        return state

    def run_tiered_arxiv(k_list):
        q = "+AND+".join([f"all:{k.replace(' ', '+')}" for k in k_list])
        results = search_arxiv(q, sort_by="submittedDate")
        if results:
            return results
        q_simple = "+AND+".join([f"all:{k.replace(' ', '+')}" for k in k_list[:2]])
        return search_arxiv(q_simple, sort_by="relevance")

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

    all_raw = arxiv_p + s2_p + oa_p + cr_p
    seen_slugs = {}

    for p in all_raw:
        slug = re.sub(r'[^a-z0-9]', '', p['title'].lower())
        if not slug:
            continue
        if slug in seen_slugs:
            existing = seen_slugs[slug]
            if not existing.get("has_abstract") and p.get("has_abstract"):
                seen_slugs[slug] = p
        else:
            seen_slugs[slug] = p

    all_unique = list(seen_slugs.values())
    enriched = enrich_missing_abstracts(all_unique)

    top_tier = [p for p in enriched if p.get("has_abstract")]
    if len(top_tier) >= 4:
        unique_candidates = top_tier
    elif top_tier:
        second_tier = [p for p in enriched if not p.get("has_abstract") and p.get("summary", "").strip()]
        unique_candidates = top_tier + second_tier
    else:
        unique_candidates = [p for p in enriched if p.get("summary", "").strip()]

    if not unique_candidates:
        unique_candidates = enriched[:5]

    reranked = semantic_rerank(state["summary"], unique_candidates)
    state["papers"] = reranked[:8]
    return state

def node_compare(state: PaperState) -> PaperState:
    """Compare uploaded paper with recent academic literature."""
    if not state.get("papers") or (len(state["papers"]) == 1 and state["papers"][0]["title"] == "Search failed"):
        state["comparison"] = (
            "Deeply sorry, but I couldn't retrieve related papers at this moment. "
            "This can happen if external APIs are temporarily unreachable or if the search topic is too specific. "
            "You can edit the search topic manually and re-run the comparative study."
        )
        return state

    combined = "\n\n".join([
        f"[{p['year']}] {p['title']} ({p.get('venue', 'Academic Source')}): {p['summary']}"
        for p in state["papers"] if p.get('has_abstract')
    ])

    papers_without = [p for p in state["papers"] if not p.get('has_abstract')]
    skip_notice = ""
    if papers_without:
        skip_names = ", ".join([f'"{p["title"]}"' for p in papers_without])
        skip_notice = f"\n\nNOTE: The following papers lack full public abstracts: {skip_names}. Do NOT infer details for them."

    prompt = f"""You are a research analyst. Compare the original paper with recent research using markdown.

IMPORTANT RULES:
- ONLY analyze papers for which abstract/summary text is provided below.
- Do NOT write "Not Reported" or "No specific data" for any paper. If a paper lacks enough information, exclude it from analysis.
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
    """Identify weak sections with publication-quality critique."""
    improve_queries = [
        "abstract introduction background",
        "methodology approach proposed solution",
        "results experiments analysis discussion",
        "conclusion limitations future scope",
    ]
    seen, context_parts = set(), []
    for q in improve_queries:
        for chunk in retrieve(q, k=3):
            if chunk not in seen:
                seen.add(chunk)
                context_parts.append(chunk)
    context = "\n\n".join(context_parts)

    prompt = f"""You are a senior research advisor reviewing this paper for publication-level quality.
Your job is to find SPECIFIC, CONCRETE weaknesses — not generic advice.

STRICT RULES:
- Quote the EXACT weak sentence or paragraph from the paper text (use quotation marks)
- Explain WHY that specific sentence is weak (too vague, missing numbers, missing justification, etc.)
- Suggest the EXACT type of content that should replace it (e.g. "add ablation study comparing X vs Y", "cite [specific metric] from the results")
- NEVER write generic advice like "Improve clarity", "Add more details", "Strengthen the argument"
- If a section is actually strong, say so — do not fabricate weaknesses
- Reference the comparative analysis to point out what competing papers do that this paper does not

Paper text (full document via semantic retrieval):
{context}

Comparative analysis with recent related work:
{state['comparison'][:1500]}

For each weak section, provide:
1. The exact quoted text that is weak
2. The specific reason it is weak
3. A concrete, actionable fix

Use markdown headings: ## Abstract, ## Introduction, ## Methodology, ## Results, ## Conclusion"""
    state["improvements"] = llm(prompt)
    return state

def node_rewrite_sections(state: PaperState) -> PaperState:
    """Generate structured section rewrites for in-place PDF replacement."""
    rewrite_queries = [
        "abstract introduction",
        "methodology proposed method",
        "results experiments metrics",
        "conclusion limitations",
    ]
    seen, context_parts = set(), []
    for q in rewrite_queries:
        for chunk in retrieve(q, k=3):
            if chunk not in seen:
                seen.add(chunk)
                context_parts.append(chunk)
    context = "\n\n".join(context_parts)

    prompt = f"""You are a research paper editor tasked with rewriting specific weak sections.
The rewritten text must sound like it was written by the original authors — technical, precise, grounded in the paper's own data.

STRICT RULES for rewrites:
- Use the EXACT same technical terminology, variable names, and method names as in the original paper
- Include REAL numbers from the paper (accuracy, loss values, dataset sizes, etc.) — do NOT invent statistics
- Do NOT add claims the paper does not support
- Strengthen weak sentences by adding specificity: replace vague claims with exact measurement references
- The rewritten version must be directly usable in the paper — no placeholders like "[add result here]"
- Match the academic writing style of the original

Full paper text (full document via semantic retrieval):
{context}

Improvement analysis:
{state['improvements'][:1500]}

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
        raw_clean = re.sub(r'```(?:json)?', '', raw).strip()
        match = re.search(r'\[.*\]', raw_clean, re.DOTALL)
        if match:
            edits = json.loads(match.group())
    except Exception:
        edits = []

    state["edits"] = edits
    return state

def node_qa(state: PaperState) -> PaperState:
    """Answer question strictly grounded in retrieved paper chunks."""
    chunks = retrieve(state["query"])
    if not chunks:
        state["answer"] = "No document loaded or no matching context found. Please upload a paper first."
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
# GRAPH PIPELINE BUILDERS
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
