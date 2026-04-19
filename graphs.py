import re
import json
from typing import TypedDict, List, Optional
import streamlit as st
from langgraph.graph import StateGraph, END

from config import client, VISION_MODEL
from utils import llm, distill_context, retrieve, encode_image, store_figure_description, rerank_papers
from api_search import search_arxiv, search_crossref, search_openalex, search_semantic_scholar

class PaperState(TypedDict):
    text: str
    images: List[str]
    chunks: List[str]
    summary: str
    vision: List[str]
    topic: List[str]
    papers: List[dict]
    comparison: str
    comp_problem: str
    comp_method: str
    comp_data: str
    comp_results: str
    comp_eval: str
    improvements: str
    edits: List[dict]
    query: str
    answer: str
    error: Optional[str]

def node_summarize(state):
    system_prompt = """You are a Senior Technical Reviewer. Your goal is to provide a MASSIVE, technically exhaustive analysis of the research paper.

### MANDATORY MATH FORMATTING ###
- Use ONLY LaTeX for mathematical symbols and equations.
- Use '$' for inline math (e.g., $E = mc^2$) and '$$' for block equations.
- Convert raw symbols like 2πTsϵ into clean LaTeX notation like $2\pi T_s \epsilon$.
- NEVER use '\[' or '\]' markers.

### REQUIRED STRUCTURE ###
You must output exactly these sections in order:
## Executive Summary
(Detailed technical overview of novelty and core innovation)

## Architecture & Methodology
(Deep dive into systems, backbones, algorithms, and logic units)

## Performance Analysis
(Theoretical analysis of channel models, mathematical derivations, and performance bounds)

## Simulation & Results
(Exhaustive summary of numerical data, benchmarks, and Delta improvements)

## Conclusion
(Technical summary of findings and future research directions)

### CONSTRAINTS ###
- START your response directly with '## Executive Summary'.
- DO NOT echo any of the input paper text.
- Be technically dense and exhaustive."""

    user_prompt = f"""<PAPER_CONTENT_TO_ANALYZE>
{state['text'][:7000]}
</PAPER_CONTENT_TO_ANALYZE>"""

    raw_response = llm(user_prompt, system_prompt=system_prompt)
    
    # --- CLEANING SAFETY NET ---
    marker = "## Executive Summary"
    if marker in raw_response:
        cleaned = raw_response[raw_response.find(marker):].strip()
        state["summary"] = cleaned
    else:
        state["summary"] = raw_response
        
    return state

def analyze_single_image(figure: dict) -> str:
    """
    Analyzes a research figure with full context grounding.
    Accepts a figure dict: {path, caption, page_num, context, figure_index}
    Stores the AI description into FAISS for Q&A retrieval.
    """
    if isinstance(figure, str):
        # Fallback for cached sessions
        img_path = figure
        caption = ""
        context = ""
        page_num = "?"
        figure_index = "?"
    else:
        img_path = figure.get("path")
        caption = figure.get("caption", "").strip()
        page_num = figure.get("page_num", "?")
        context = figure.get("context", "")[:1200]
        figure_index = figure.get("figure_index", "?")

    # Build contextual text block for the prompt
    context_block = ""
    if caption:
        context_block += f"**Figure Caption (extracted from paper):** {caption}\n\n"
    if context:
        context_block += f"**Surrounding Paragraph Text (±100 words from paper):**\n{context}\n\n"
    if str(page_num) != "?":
        context_block += f"**Source:** Page {page_num} of the uploaded paper.\n"

    research_prompt = f"""You are analyzing Figure {figure_index} from a research paper.

You are provided with:
1. The ACTUAL FIGURE IMAGE (visual)
2. The CAPTION that the authors wrote for this figure
3. The SURROUNDING TEXT from the paper where this figure is discussed

Use ALL three sources together to answer the following:

---
{context_block}
---

**Your Analysis Tasks:**
- **Figure Type**: What kind of figure is this? (e.g., Architecture diagram, Results graph, Ablation table, Confusion matrix, etc.)
- **Key Visual Data**: What specific numbers, labels, axes, or patterns are visible in the image?
- **Author's Claim**: Based on the caption and surrounding text, what is the author trying to prove with this figure?
- **Claim Validation**: Does the visual content of the image actually support the author's claim? Are there any inconsistencies or missing details?
- **Research Significance**: Why is this figure important to the paper's contribution?

Be precise. Quote numbers you can see in the image. Do not hallucinate data not visible in the figure."""

    try:
        b64 = encode_image(img_path)
        res = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": research_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]}],
            max_tokens=1200,
            temperature=0.2
        )
        description = res.choices[0].message.content

        # Step 4: Store in FAISS so Q&A can retrieve figure content
        store_figure_description(figure_index, description)

        return description
    except Exception as e:
        return f"Vision error: {e}"

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
        # Relaxed filtering: Keep papers even with minimal metadata summaries
        # Just ensure they have a title and some summary text
        if not p.get("title") or not p.get("summary") or len(p["summary"]) < 20:
            continue
            
        slug = re.sub(r'[^a-z0-9]', '', p['title'].lower())
        if slug and slug not in seen:
            unique.append(p)
            seen.add(slug)
    
    # NEW: Semantic re-ranking using the original paper summary
    state["papers"] = rerank_papers(state["summary"], unique, top_k=6)
    return state


def node_compare_problem(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary'][:1500]}" for p in state["papers"]])
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Reviewer | TASK: 1. Problem & Objective
CONSTRAINTS: What is the paper trying to solve? Compare the primary objective facing the original paper versus the related research (e.g., Engineering system design, Medical disease treatment, Economics policy, etc.).
MANDATORY: Structure your response strictly paper-by-paper. Use bold subheadings or bullet points (e.g., **[Year] [Title]**) to visually separate the comparison.
MANDATORY: DO NOT include any meta-commentary, recommendations, "Bottom lines", or hypothetical examples. If details are missing, simply skip or state 'Not Reported'. Stay strictly analytical.

### CONTEXT ###
Original: {state['summary'][:2000]}
Related Research: {combined}

## 1. Problem & Objective
"""
    state["comp_problem"] = llm(prompt, disable_failsafe=True)
    return state

def node_compare_method(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary'][:1500]}" for p in state["papers"]])
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Reviewer | TASK: 2. Methodology & Approach
CONSTRAINTS: How is the problem solved? Compare the approaches used (e.g., Algorithms, Experiments, Theoretical models, Surveys) against the related works.
MANDATORY: Structure your response strictly paper-by-paper. Use bold subheadings or bullet points.
MANDATORY: DO NOT include any meta-commentary, recommendations, "Bottom lines", or hypothetical examples. Stay analytical.

### CONTEXT ###
Original: {state['summary'][:2000]}
Related Research: {combined}

## 2. Methodology & Approach
"""
    state["comp_method"] = llm(prompt, disable_failsafe=True)
    return state

def node_compare_data(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary'][:1500]}" for p in state["papers"]])
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Reviewer | TASK: 3. Data & Evidence
CONSTRAINTS: What data is used? Compare the exact evidence, datasets, case studies, or simulations used in this paper versus each related paper.
MANDATORY: Structure your response strictly paper-by-paper. Use bold subheadings or bullet points.
MANDATORY: DO NOT include any meta-commentary, recommendations, "Bottom lines", or hypothetical examples. Stay analytical.

### CONTEXT ###
Original: {state['summary'][:2000]}
Related Research: {combined}

## 3. Data & Evidence
"""
    state["comp_data"] = llm(prompt, disable_failsafe=True)
    return state

def node_compare_results(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary'][:1500]}" for p in state["papers"]])
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Reviewer | TASK: 4. Results & Findings
CONSTRAINTS: What did they achieve? Compare the findings, observations, Accuracy rates, or categorical improvements against the related works.
MANDATORY: Structure your response strictly paper-by-paper. Use bold subheadings or bullet points.
MANDATORY: DO NOT include any meta-commentary, recommendations, "Bottom lines", or hypothetical examples. Stay analytical.

### CONTEXT ###
Original: {state['summary'][:2000]}
Related Research: {combined}

## 4. Results & Findings
"""
    state["comp_results"] = llm(prompt, disable_failsafe=True)
    return state

def node_compare_eval(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary'][:1500]}" for p in state["papers"]])
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Reviewer | TASK: 5. Evaluation Method
CONSTRAINTS: How did they validate results? Compare the validation strategies, metrics, experiments, or proofs used to confirm effectiveness.
MANDATORY: Structure your response strictly paper-by-paper. Use bold subheadings or bullet points.
MANDATORY: DO NOT include any meta-commentary, recommendations, "Bottom lines", or hypothetical examples. Stay analytical.

### CONTEXT ###
Original: {state['summary'][:2000]}
Related Research: {combined}

## 5. Evaluation Method
"""
    state["comp_eval"] = llm(prompt, disable_failsafe=True)
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
    raw = llm(prompt, disable_failsafe=True)
    try:
        raw_clean = re.sub(r'```(?:json)?', '', raw).strip()
        m = re.search(r'\[.*\]', raw_clean, re.DOTALL)
        if m:
            edits = json.loads(m.group())
            unique_edits = []
            seen_sections = set()
            for ed in edits:
                sec = ed.get("section", "General")
                if sec not in seen_sections:
                    unique_edits.append(ed)
                    seen_sections.add(sec)
            state["edits"] = unique_edits
        else:
            state["edits"] = [{"section": "JSON Formatting Error", "original": "The AI failed to format the response as a JSON array.", "rewritten": raw_clean}]
    except Exception as e:
        state["edits"] = [{"section": "JSON Parse Error", "original": f"Exception: {str(e)}", "rewritten": raw}]
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

@st.cache_resource
def build_graphs():
    g1 = StateGraph(PaperState)
    g1.add_node("summarize", node_summarize)
    g1.add_node("extract_topic", node_extract_topic)
    g1.set_entry_point("summarize")
    g1.add_edge("summarize", "extract_topic")
    g1.add_edge("extract_topic", END)

    g2 = StateGraph(PaperState)
    g2.add_node("arxiv_search", node_arxiv_search)
    g2.add_node("compare_problem", node_compare_problem)
    g2.add_node("compare_method", node_compare_method)
    g2.add_node("compare_data", node_compare_data)
    g2.add_node("compare_results", node_compare_results)
    g2.add_node("compare_eval", node_compare_eval)
    g2.set_entry_point("arxiv_search")
    g2.add_edge("arxiv_search", "compare_problem")
    g2.add_edge("compare_problem", "compare_method")
    g2.add_edge("compare_method", "compare_data")
    g2.add_edge("compare_data", "compare_results")
    g2.add_edge("compare_results", "compare_eval")
    g2.add_edge("compare_eval", END)

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
