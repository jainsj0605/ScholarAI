import re
import json
from typing import TypedDict, List, Optional
import streamlit as st
from langgraph.graph import StateGraph, END

from config import client, VISION_MODEL
from utils import llm, distill_context, retrieve, encode_image, store_figure_description, rerank_papers, deduplicate_chunks
from api_search import search_arxiv, search_crossref, search_openalex, search_semantic_scholar, fetch_arxiv_fulltext, fetch_crossref_fulltext

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
    # Perform 4 targeted semantic searches for full-paper coverage
    queries = [
        "abstract problem statement motivation",
        "methodology approach proposed method",
        "results experiments evaluation metrics",
        "conclusion limitations future work"
    ]
    all_chunks = []
    for q in queries:
        all_chunks.extend(retrieve(q, k=3))
    
    context_chunks = deduplicate_chunks(all_chunks)
    context_text = "\n\n".join(context_chunks)

    system_prompt = """You are a Senior Technical Auditor. Your goal is to provide a MASSIVE, technically exhaustive analysis of the research paper.

### RULES FOR SPECIFICITY (MANDATORY) ###
- NEVER use generic filler like "The authors propose", "This paper presents", or "In this work".
- ALWAYS name the EXACT proposed system, algorithm, or mathematical framework from the paper.
- ALWAYS cite EXACT numerical metrics (e.g., 94.3% accuracy, 2.1dB gain, 15ms latency).
- Use EXACT variable names and dataset names mentioned in the text.
- If a number or specific method is not found in the context, do not invent one.

### MANDATORY MATH FORMATTING ###
- Use ONLY LaTeX for mathematical symbols and equations.
- Use '$' for inline math (e.g., $E = mc^2$) and '$$' for block equations.
- Use raw strings for LaTeX: e.g., $2\\pi T_s \\epsilon$.
- NEVER use '\\[' or '\\]' markers.

### REQUIRED STRUCTURE ###
## TLDR
(One sentence summary: Must include specific problem + exact method name + top result WITH a number)

## Problem
(What specific technical problem does it solve? Use terms from the paper.)

## Method
(Deep dive into the EXACT proposed algorithm/framework. Name it explicitly.)

## Results
(List EXACT metrics: e.g., 94.3% accuracy on [dataset], outperforming [baseline] by 2.1%)

## Limitations
(Technical summary of findings and specific known constraints mentioned by authors)"""

    user_prompt = f"""<PAPER_CONTEXT_TO_ANALYZE>
{context_text}
</PAPER_CONTEXT_TO_ANALYZE>"""

    raw_response = llm(user_prompt, system_prompt=system_prompt)
    
    # --- CLEANING SAFETY NET ---
    marker = "## TLDR"
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
3. 200 WORDS OF SPATIAL CONTEXT (100 words directly above and 100 words directly below the figure in the PDF)

Use ALL three sources together to answer the following:

---
{context_block}
---

**Your Analysis Tasks:**
- **Figure Type**: What kind of figure is this? (e.g., Architecture diagram, Results graph, etc.)
- **Key Visual Data**: What specific numbers, labels, or patterns are visible in the image?
- **Author's Claim**: Based on the caption and the surrounding context (Above/Below), what is the author trying to prove?
- **Claim Validation**: Does the visual content actually support the text provided above and below the figure?
- **Research Significance**: Why is this figure important to the paper's contribution?

Be precise. Do not hallucinate data not visible in the figure."""

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
    """
    Engineering-Grade Multi-Engine Academic Search
    
    This node implements an ENGINEERING-GRADE SEARCH SYSTEM that:
    1. Searches across 4 academic databases (ArXiv, Semantic Scholar, OpenAlex, CrossRef)
    2. Prioritizes prestigious engineering venues (IEEE, Springer, Elsevier, Wiley)
    3. Applies strict quality filtering (100+ char abstracts)
    4. Intelligently deduplicates (keeps version with best abstract)
    5. Enriches with full-text content from ArXiv/CrossRef
    6. Re-ranks by semantic relevance + venue prestige
    
    The system ensures engineering papers like "Amplitude Modulation" from IEEE
    are prioritized over general noise.
    """
    query = state["topic"]
    if not query or "Error:" in query:
        state["papers"] = []
        return state
        
    # Search all sources - they filter for quality abstracts internally (Level 1)
    arxiv_p = search_arxiv(query)
    semantic_p = search_semantic_scholar(query)
    openalex_p = search_openalex(query)
    crossref_p = search_crossref(query)
    
    # Combine all results
    all_p = arxiv_p + semantic_p + openalex_p + crossref_p
    
    # LEVEL 2 FILTERING: Engineering-Grade Gatekeeper with Intelligent Deduplication
    unique = {}  # Use dict to track best version of each paper
    
    for p in all_p:
        # Special handling for papers flagged as needing full-text
        if p.get("needs_fulltext"):
            slug = re.sub(r'[^a-z0-9]', '', p['title'].lower())
            if slug not in unique:
                unique[slug] = p
            continue
        
        # RULE 1: Must have title and substantial abstract (100+ chars)
        if not p.get("title") or not p.get("summary") or len(p["summary"].strip()) < 100:
            continue
        
        # RULE 2: Skip generic placeholder text
        summary_lower = p["summary"].lower()
        if "not provided" in summary_lower or "metadata indicates" in summary_lower:
            continue
        
        # RULE 3: Skip "no abstract available" messages
        if "no abstract available" in summary_lower or "abstract not available" in summary_lower:
            continue
        
        # RULE 4: Intelligent Deduplication - Keep version with best abstract/venue
        slug = re.sub(r'[^a-z0-9]', '', p['title'].lower())
        if slug:
            if slug not in unique:
                unique[slug] = p
            else:
                # Keep the version with longer abstract OR higher venue score
                existing = unique[slug]
                existing_len = len(existing.get("summary", ""))
                new_len = len(p.get("summary", ""))
                existing_score = existing.get("venue_score", 0)
                new_score = p.get("venue_score", 0)
                
                # Prefer: 1) Higher venue score, 2) Longer abstract
                if new_score > existing_score or (new_score == existing_score and new_len > existing_len):
                    unique[slug] = p
    
    # Convert back to list
    unique_papers = list(unique.values())
    
    # ENGINEERING-GRADE RANKING: Combine semantic relevance + venue prestige
    # First, get semantic scores from rerank_papers
    ranked_papers = rerank_papers(state["summary"], unique_papers, top_k=15)  # Get more candidates
    
    # Apply venue weighting to boost prestigious engineering venues
    for paper in ranked_papers:
        venue_score = paper.get("venue_score", 1.0)
        # Boost papers from IEEE, Springer, Elsevier significantly
        if venue_score >= 8.0:
            # Move high-prestige papers to front (multiply by position boost)
            paper["final_score"] = venue_score * 2.0
        else:
            paper["final_score"] = venue_score
    
    # Re-sort by final score (venue prestige + semantic relevance)
    ranked_papers.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    
    # Take top 6 after venue-weighted ranking
    ranked_papers = ranked_papers[:6]
    
    # ENHANCEMENT: Fetch full text for ArXiv and CrossRef papers
    enriched_papers = []
    for paper in ranked_papers:
        # Try ArXiv full-text extraction
        if paper.get("venue") == "ArXiv" and paper.get("link"):
            full_text = fetch_arxiv_fulltext(paper["link"])
            if full_text:
                paper["full_content"] = full_text
                paper["summary"] = paper["summary"] + "\n\n[Extended Content from ArXiv Paper]: " + full_text[:3000]
        
        # Try CrossRef full-text extraction
        elif paper.get("needs_fulltext") or (paper.get("venue_raw") and any(x in paper.get("venue_raw", "").lower() for x in ["ieee", "springer", "elsevier"])):
            if paper.get("link"):
                full_text = fetch_crossref_fulltext(paper["link"])
                if full_text:
                    paper["full_content"] = full_text
                    if "[Abstract pending" in paper["summary"]:
                        paper["summary"] = full_text[:3000]
                    else:
                        paper["summary"] = paper["summary"] + "\n\n[Extended Content from DOI]: " + full_text[:3000]
                    paper["needs_fulltext"] = False
                elif paper.get("needs_fulltext"):
                    continue
        
        enriched_papers.append(paper)
    
    state["papers"] = enriched_papers
    return state


def node_compare_problem(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary']}" for p in state["papers"]])
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Auditor | TASK: 1. Problem & Objective Comparison
CONSTRAINTS: 
- Compare the primary technical objective of the original paper versus each related paper.
- MANDATORY: Use EXACT terminology from the abstracts.
- BANNED: "The authors address...", "This paper explores...".
- Each paper must have its own bold subheading.
- Focus on the specific research gap (e.g., "high latency in 5G handovers") rather than general themes.

### CONTEXT ###
Original Paper Summary:
{state['summary'][:2500]}

Related Research Papers:
{combined}

## 1. Problem & Objective
"""
    state["comp_problem"] = llm(prompt, disable_failsafe=True)
    return state

def node_compare_method(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary']}" for p in state["papers"]])
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Auditor | TASK: 2. Methodology & Approach Comparison
CONSTRAINTS: 
- Compare the exact algorithms, architectures, or mathematical models.
- MANDATORY: List specific variable names or algorithm names (e.g., "DeepSet-V2", "Adam Optimizer with 0.01 learning rate").
- BANNED: "They use a unique approach", "A machine learning model is utilized".
- Each paper must have its own bold subheading.

### CONTEXT ###
Original Paper Summary:
{state['summary'][:2500]}

Related Research Papers:
{combined}

## 2. Methodology & Approach
"""
    state["comp_method"] = llm(prompt, disable_failsafe=True)
    return state

def node_compare_data(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary']}" for p in state["papers"]])
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Auditor | TASK: 3. Data & Evidence Comparison
CONSTRAINTS: 
- Identify the exact datasets (e.g., "MNIST", "COCO", "Custom dataset of 5,000 ECG signals").
- MANDATORY: Mention sample sizes or simulation parameters.
- BANNED: "Publicly available data was used", "Simulation results are provided".

### CONTEXT ###
Original Paper Summary:
{state['summary'][:2500]}

Related Research Papers:
{combined}

## 3. Data & Evidence
"""
    state["comp_data"] = llm(prompt, disable_failsafe=True)
    return state

def node_compare_results(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary']}" for p in state["papers"]])
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Auditor | TASK: 4. Quantitative Results Comparison
CONSTRAINTS: 
- MANDATORY: Provide a Comparative Results Matrix (Table) at the end.
- MANDATORY: List EXACT numerical results (e.g., "accuracy: 98.2%", "RMSE: 0.12").
- BANNED: "Significant improvement was observed", "Better performance than baselines".
- Focus on extracting REAL numbers from each snippet.

### CONTEXT ###
Original Paper Summary:
{state['summary'][:2500]}

Related Research Papers:
{combined}

## 4. Results & Findings
"""
    state["comp_results"] = llm(prompt, disable_failsafe=True)
    return state

def node_compare_eval(state):
    combined = "\n\n".join([f"[{p['year']}] {p['title']}: {p['summary']}" for p in state["papers"]])
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Auditor | TASK: 5. Evaluation Method Comparison
CONSTRAINTS: 
- Compare the exact validation metrics (e.g., "F1-Score", "mAP", "BER").
- MANDATORY: Mention exactly what they were compared AGAINST (Baselines).
- BANNED: "Validated using standard metrics".

### CONTEXT ###
Original Paper Summary:
{state['summary'][:2500]}

Related Research Papers:
{combined}

## 5. Evaluation Method
"""
    state["comp_eval"] = llm(prompt, disable_failsafe=True)
    return state

def node_improve(state):
    # Perform 4 targeted searches for identify technical weaknesses
    queries = [
        "abstract introduction background",
        "methodology approach proposed solution",
        "results experiments analysis discussion",
        "conclusion limitations future scope"
    ]
    all_chunks = []
    for q in queries:
        all_chunks.extend(retrieve(q, k=3))
    
    context_chunks = deduplicate_chunks(all_chunks)
    context_text = "\n\n".join(context_chunks)

    # DISTILL comparison context to save tokens (capped at 1500 chars)
    comp_context = distill_context(state['comparison'])[:1500]
    
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Senior Technical Editor
TASK: Identify exactly 3-5 SPECIFIC weak sections in the paper that need technical improvement.

### RULES FOR SPECIFICITY (MANDATORY) ###
- For each weakness, you MUST quote the EXACT weak sentence from the paper using quotation marks.
- Explain precisely WHY that sentence is weak (e.g., "missing justification for parameter X", "generic claims without benchmark").
- Suggest a CONCRETE technical fix (e.g., "add ablation study comparing method X against baseline Y").
- DO NOT use generic feedback like "improve clarity" or "add more details".
- DO NOT fabricate weaknesses in sections that are technically strong.

### STRUCTURE ###
Each section heading (e.g. ## Methodology) must contain:
1. "Quoted weak text"
2. Reason it is weak.
3. Concrete, actionable fix.

### CONTEXT ###
[COMPARATIVE ANALYSIS GAPS]
{comp_context}

[PAPER SNIPPETS]
{context_text}

## Improvement Strategy
"""
    state["improvements"] = llm(prompt)
    return state

def node_rewrite(state):
    if "Error" in state["improvements"] or not state["improvements"]:
        state["edits"] = []
        return state

    # Perform targeted searches to get verbatim text for the identified sections
    queries = [
        "abstract introduction",
        "methodology proposed method",
        "results experiments metrics",
        "conclusion limitations"
    ]
    all_chunks = []
    for q in queries:
        all_chunks.extend(retrieve(q, k=3))
    
    context_chunks = deduplicate_chunks(all_chunks)
    context_text = "\n\n".join(context_chunks)

    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Author
TASK: Rewrite the weak sections identified by the editor to sound exactly like the original authors wrote them.

### RULES FOR AUTHENTICITY (MANDATORY) ###
- Use the EXACT technical terminology, variable names, and method names from the paper.
- Use REAL numbers and metrics found in the paper (DO NOT invent statistics).
- Ensure the tone matches the original professional scientific style.
- The rewritten text must be directly usable in the paper.
- BANNED: Placeholders like "[add result here]" or "[insert figure]".

### OUTPUT FORMAT (MANDATORY) ###
Return a VALID JSON array ONLY.
[
  {{
    "section": "Precise Section Name",
    "original": "EXACT 100-150 character snippet of the original text to locate it",
    "rewritten": "The complete improved technical text for the section"
  }}
]

### ANALYSIS OF WEAKNESSES ###
{state['improvements']}

### FULL PAPER VERBATIM CONTEXT ###
{context_text}

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
