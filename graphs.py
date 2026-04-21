import re
import json
from typing import TypedDict, List, Optional
import streamlit as st
from langgraph.graph import StateGraph, END

from config import client, VISION_MODEL, FAST_MODEL
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
    # Perform 4 targeted semantic searches for high-depth coverage
    queries = [
        "abstract problem statement motivation",
        "methodology approach proposed method",
        "results experiments evaluation metrics",
        "conclusion limitations future work"
    ]
    all_chunks = []
    for q in queries:
        all_chunks.extend(retrieve(q, k=5)) # Increased to k=5 for depth
    
    context_chunks = deduplicate_chunks(all_chunks)
    context_text = "\n\n".join(context_chunks)

    system_prompt = """You are a Senior Technical Auditor. Your goal is to provide an EXHAUSTIVE, technically dense analysis of the research paper.

### EXHAUSTIVE DEPTH RULES (MANDATORY) ###
- Each section (Problem, Method, Results) must be at least 2-3 detailed paragraphs long.
- Do not just summarize; provide a deep technical architectural walkthrough.
- BANNED: Never use generic filler like "The authors propose", "This paper presents", or "In this work".
- ALWAYS name the EXACT proposed system, algorithm, or mathematical framework from the paper.
- ALWAYS cite EXACT numerical metrics (e.g., 94.3% accuracy, 2.1dB gain, 15ms latency).
- Use EXACT variable names and dataset names mentioned in the text.

### MANDATORY MATH FORMATTING ###
- Use ONLY LaTeX for mathematical symbols and equations.
- Use '$' for inline math (e.g., $E = mc^2$) and '$$' for block equations.
- NEVER use markers like '\[', '\]', '\(', '\)', or '\begin{equation}'.
- If you need to write a formula, just use '$$' for display math.

### REQUIRED STRUCTURE ###
## Executive Summary
(Provide 2-3 detailed paragraphs highlighting the core innovation, novelty, and exact top-line result. Do NOT use the prefix 'TL;DR'.)

## Architecture & Methodology
(Deep dive into the EXACT proposed algorithm/framework. Provide at least 3 detailed paragraphs explaining the systems, backbones, and mathematical logic unit by unit.)

## Performance Analysis
(Exhaustive technical analysis of channel models, mathematical derivations, and theoretical performance bounds mentioned in the paper. Minimum 2 paragraphs.)

## Simulation & Results
(Exhaustive Benchmarking: List EVERY numerical metric, comparison baseline, and performance delta. Explain the significance of the data in multiple paragraphs.)

## Conclusion
(Technical summary of findings, specific known constraints, and future research directions.)

### FORMATTING RULES ###
- Use ONLY '## ' for section headers.
- BANNED: Never use horizontal separators like '---' or '***'.
- BANNED: Never use bolding inside the section headers."""

    user_prompt = f"""<PAPER_CONTEXT_TO_ANALYZE>
{context_text}
</PAPER_CONTEXT_TO_ANALYZE>"""

    raw_response = llm(user_prompt, system_prompt=system_prompt)
    
    # --- CLEANING SAFETY NET ---
    marker = "## Executive Summary"
    if marker in raw_response:
        cleaned = raw_response[raw_response.find(marker):].strip()
        # Hard Failsafe: Strip accidental TLDR or Summary prefixes
        cleaned = re.sub(r'(## Executive Summary\n*)\b(TL;DR|TLDR|Summary)\b\s*[-—:]*', r'\1', cleaned, flags=re.IGNORECASE)
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

        # AUTOMATIC CLEANING: Fix LaTeX formulas in Vision descriptions
        from utils import clean_math_output
        description = clean_math_output(description)

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
ROLE: Technical Reviewer | TASK: 1. Problem & Objective
CONSTRAINTS: What is the paper trying to solve? Compare the primary objective facing the original paper versus the related research.
MANDATORY: Structure your response strictly paper-by-paper. Use bold subheadings (e.g., **[Year] [Title]**).
MANDATORY: Extract EVERY available detail from the abstracts - research gaps, motivations, target applications, specific challenges addressed.
MANDATORY: If an abstract lacks specific problem statements, infer from the methodology and results described.
MANDATORY: DO NOT write "Not Reported" unless the abstract is completely uninformative. Extract what you can.

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
ROLE: Technical Reviewer | TASK: 2. Methodology & Approach
CONSTRAINTS: How is the problem solved? Compare the approaches used (e.g., Algorithms, Models, Frameworks, Experiments, Theoretical analysis).
MANDATORY: Structure your response strictly paper-by-paper. Use bold subheadings.
MANDATORY: Extract EVERY methodological detail from abstracts - algorithms, architectures, frameworks, mathematical models, experimental designs.
MANDATORY: Look for keywords like "propose", "develop", "design", "implement", "analyze", "model", "framework", "algorithm", "method".
MANDATORY: If methodology is implicit, infer from problem and results sections.

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
ROLE: Technical Reviewer | TASK: 3. Data & Evidence
CONSTRAINTS: What data/evidence is used? Compare datasets, simulations, case studies, experimental setups, or theoretical proofs.
MANDATORY: Structure your response strictly paper-by-paper. Use bold subheadings.
MANDATORY: Extract EVERY data-related detail - dataset names, simulation parameters, experimental conditions, sample sizes, test scenarios.
MANDATORY: Look for keywords like "dataset", "data", "simulation", "experiment", "test", "benchmark", "case study", "evaluation", "validate".
MANDATORY: If specific datasets aren't named, describe the type of data used (e.g., "satellite telemetry data", "clinical trials", "synthetic data").

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
ROLE: Technical Reviewer | TASK: 4. Results & Findings
CONSTRAINTS: What did they achieve? Compare findings, performance metrics, improvements, observations, or discoveries.
MANDATORY: Structure your response strictly paper-by-paper. Use bold subheadings.
MANDATORY: Extract EVERY quantitative and qualitative result - accuracy rates, error reductions, performance gains, efficiency improvements, novel findings.
MANDATORY: Look for keywords like "achieve", "improve", "reduce", "increase", "demonstrate", "show", "find", "result", "performance", "accuracy", "error", "%", "dB", "rate".
MANDATORY: Include specific numbers, percentages, or comparative statements (e.g., "outperforms", "better than", "reduces by").

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
ROLE: Technical Reviewer | TASK: 5. Evaluation Method
CONSTRAINTS: How did they validate results? Compare validation strategies, metrics, experimental protocols, or theoretical proofs.
MANDATORY: Structure your response strictly paper-by-paper. Use bold subheadings.
MANDATORY: Extract EVERY evaluation detail - metrics used (RMSE, accuracy, F1, BER, SNR, etc.), comparison baselines, validation methods, statistical tests.
MANDATORY: Look for keywords like "evaluate", "validate", "compare", "metric", "measure", "assess", "test", "benchmark", "baseline", "versus", "against".
MANDATORY: If explicit evaluation isn't described, infer from results section (e.g., "compared against X" implies comparative evaluation).

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
    # Perform 4 targeted searches to identify technical weaknesses with k=5 depth
    queries = [
        "abstract introduction background",
        "methodology approach proposed solution",
        "results experiments analysis discussion",
        "conclusion limitations future scope"
    ]
    all_chunks = []
    for q in queries:
        all_chunks.extend(retrieve(q, k=5))
    
    context_chunks = deduplicate_chunks(all_chunks)
    context_text = "\n\n".join(context_chunks)

    # DISTILL comparison context to save tokens (capped at 1500 chars)
    comp_context = distill_context(state['comparison'])[:1500]
    
    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Senior Technical Editor
TASK: Identify EXACTLY 5 DIFFERENT and SPECIFIC weak sections in the paper that need technical improvement. 

### RULES FOR SPECIFICITY (MANDATORY) ###
- You MUST find exactly 5 separate weaknesses. 
- For each weakness, you MUST quote the EXACT weak sentence from the paper using quotation marks.
- Explain precisely WHY that sentence is weak (e.g., "missing justification for parameter X", "generic claims without benchmark").
- Provide a LONG, CONCRETE technical fix (e.g., "add ablation study comparing method X against baseline Y").
- BANNED: Do not use generic feedback like "improve clarity" or "add more details".
- DO NOT combine improvements. List them as 1, 2, 3, 4, 5.

### RESEARCH CONTEXT (YOUR PAPER) ###
{context_text}

### COMPARATIVE ANALYSIS (STATE OF THE ART) ###
{comp_context}

### STRUCTURE ###
1. ## [Weakness 1 Name]
   - "Quoted weak text"
   - Technical reason: ...
   - Actionable fix: ...

2. ## [Weakness 2 Name]
   ... and so on for all 5.

## Improvement Strategy
"""
    state["improvements"] = llm(prompt)
    return state

def node_rewrite(state):
    if "Error" in state["improvements"] or not state["improvements"]:
        state["edits"] = []
        return state

    # Perform targeted searches to get verbatim text for the identified sections with k=5 depth
    queries = [
        "abstract introduction",
        "methodology proposed method",
        "results experiments metrics",
        "conclusion limitations"
    ]
    all_chunks = []
    for q in queries:
        all_chunks.extend(retrieve(q, k=5))
    
    context_chunks = deduplicate_chunks(all_chunks)
    context_text = "\n\n".join(context_chunks)

    prompt = f"""<<< SYSTEM INSTRUCTIONS >>>
ROLE: Technical Author
TASK: Rewrite the 5 weak sections identified by the editor to sound exactly like the original authors wrote them.

### RULES FOR AUTHENTICITY (MANDATORY) ###
- You MUST provide EXACTLY 5 objects in the JSON array. ONE FOR EACH IMPROVEMENT.
- For each rewritten section, provide at least 2-3 dense technical paragraphs.
- Use the EXACT technical terminology, variable names, and method names from the paper.
- Use REAL numbers and metrics found in the paper.
- BANNED: Placeholders like "[add result here]" or "[insert figure]".

### OUTPUT FORMAT (MANDATORY) ###
Return a VALID JSON array ONLY. Do not add any preamble or conversational text.
[
  {{
    "section": "Precise Section Name 1",
    "original": "EXACT 100-150 character snippet of the original text",
    "rewritten": "The complete improved technical text (at least 2 paragraphs)"
  }},
  ... (Repeat for all 5 sections)
]


### ANALYSIS OF WEAKNESSES ###
{state['improvements']}

### FULL PAPER VERBATIM CONTEXT ###
{context_text}

### OUTPUT ###
"""
    raw = llm(prompt, disable_failsafe=True, skip_cleanup=True)
    
    # DE-HALLUCINATION: Filter out repetitive bracket sequences (common model failure)
    # If a string contains more than 50 consecutive [ or ], truncate it.
    raw = re.sub(r'\[{50,}', ' [ ... (repetitive hallucination) ] ', raw)
    raw = re.sub(r'\]{50,}', ' [ ... (repetitive hallucination) ] ', raw)
    try:
        # Step 1: Aggressive Cleanup - isolate the array
        # This handles cases where the LLM adds text before/after the JSON
        start = raw.find('[')
        end = raw.rfind(']')
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON array structure found in response.")
        
        raw_json = raw[start:end+1]
        
        # Step 2: Fix common LLM JSON errors (unescaped newlines inside strings)
        # We replace actual newlines within the JSON block with spaces to prevent parser crashes
        # but we preserve the \n escape sequences.
        raw_json = raw_json.replace('\n', ' ').replace('\r', '')
        
        # Step 3: Try standard parse
        try:
            edits = json.loads(raw_json)
        except json.JSONDecodeError:
            # Step 4: Hyper-Robust Fallback - Regex-based object extraction
            # This finds individual { } objects even if the outer array is broken
            obj_pattern = r'\{[^{}]*?"section":.*?"rewritten":.*?\}'
            found_objs = re.findall(obj_pattern, raw_json, re.DOTALL)
            
            edits = []
            for obj_str in found_objs:
                try:
                    # Clean the individual object string and try to parse it
                    # This often fixes trailing commas or missing closing brackets
                    clean_obj = obj_str.strip()
                    if not clean_obj.endswith('}'): clean_obj += '}'
                    edits.append(json.loads(clean_obj))
                except:
                    continue
            
            if not edits:
                raise ValueError("Could not extract any valid objects from JSON.")

        # Step 5: Clean and Deduplicate
        unique_edits = []
        seen_sections = set()
        for ed in edits:
            if not isinstance(ed, dict): continue
            
            sec = ed.get("section", "Technical Improvement").strip()
            orig = ed.get("original", "See paper...").strip()
            rewr = ed.get("rewritten", "").strip()
            
            # AUTOMATIC CLEANING: Fix LaTeX formulas inside JSON fields
            from utils import clean_math_output
            orig = clean_math_output(orig)
            rewr = clean_math_output(rewr)
            
            if rewr and sec not in seen_sections:
                unique_edits.append({
                    "section": sec,
                    "original": orig,
                    "rewritten": rewr
                })
                seen_sections.add(sec)
        
        state["edits"] = unique_edits

    except Exception as e:
        # Step 6: Final Safety Net - Split by markers if JSON is totally destroyed
        # We try to at least provide some structure rather than a single block
        sections = re.split(r'\"section\":', raw)
        if len(sections) > 1:
            fallback_edits = []
            for s in sections[1:]:
                name_match = re.search(r'\"(.*?)\"', s)
                name = name_match.group(1) if name_match else "Improvement"
                content_match = re.search(r'\"rewritten\":\s*\"(.*?)\"', s, re.DOTALL)
                content = content_match.group(1) if content_match else s
                fallback_edits.append({
                    "section": name,
                    "original": "Partial match",
                    "rewritten": content.replace('\\n', '\n').strip()
                })
            state["edits"] = fallback_edits
        else:
            state["edits"] = [{
                "section": "Rewrite Outcome",
                "original": "The AI provided complex text that required manual splitting.",
                "rewritten": clean_math_output(raw)
            }]
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
