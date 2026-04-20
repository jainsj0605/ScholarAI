import os
import re
import base64
import tempfile
import fitz
import faiss
import numpy as np
import streamlit as st

from config import client, TEXT_MODEL, FALLBACK_MODEL, FAST_MODEL, MODEL_TIERS, embed_model, dimension

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def llm(prompt: str, system_prompt: str = None, model: str = None, max_chars: int = 100000, disable_failsafe: bool = False) -> str:
    # Truncate prompt to prevent 413 or TPM errors
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars] + "\n\n[Context truncated due to size limits...]"
        
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Tiered Fallback Logic (120B -> 70B -> 8B)
    # If a specific model is requested (e.g. 8B for light tasks), we try ONLY that model.
    # Otherwise, we try all tiers in order.
    tiers_to_try = [model] if model else MODEL_TIERS
    
    last_error = ""
    for current_model in tiers_to_try:
        try:
            res = client.chat.completions.create(
                model=current_model,
                messages=messages,
                max_tokens=4000,
                temperature=0.3
            )
            content = res.choices[0].message.content
            # Failsafe: Strip accidental instruction leakage
            content = re.sub(r'^<<< SYSTEM INSTRUCTIONS >>>', '', content, flags=re.MULTILINE).strip()
            content = re.sub(r'^### .* ###', '', content, flags=re.MULTILINE).strip()
            
            # Hard Failsafe: Ensure it ends on a full sentence
            if not disable_failsafe and not content.strip().endswith(('.', '!', '?', ']', '\"', '\'')):
                last_period = max(content.rfind('.'), content.rfind('!'), content.rfind('?'))
                if last_period != -1:
                    content = content[:last_period + 1] + "\n\n[Section complete]"
            return content
            
        except Exception as e:
            err_msg = str(e).lower()
            last_error = str(e)
            
            # Detection: TPM (Per Minute) vs TPD (Per Day)
            is_tpm_limit = ("429" in err_msg or "rate limit" in err_msg) and ("tpm" in err_msg or "per minute" in err_msg or "rpm" in err_msg)
            is_tpd_limit = "tokens per day" in err_msg or "daily limit" in err_msg or "tpd" in err_msg
            is_context_limit = "413" in err_msg or "context length" in err_msg
            
            if is_tpd_limit and not model:
                # Daily limit reached for this tier, definitely move to next
                continue
            elif is_tpm_limit and not model:
                # Per-minute limit reached. We could wait, but it's faster to move to the next tier 
                # which has its own separate per-minute bucket.
                continue
            elif is_context_limit:
                if len(messages[-1]["content"]) > 10000:
                    messages[-1]["content"] = messages[-1]["content"][:10000] + "..."
                continue
            else:
                # Other errors (401, 500, etc.)
                return f"Error: {last_error}"

    # If we are here, all attempted models failed
    if "tokens per day" in last_error.lower() or "tpd" in last_error.lower() or "daily" in last_error.lower():
        return "CRITICAL: All AI models have reached their daily token limits. Please try again in 24 hours or use a different API key."
    elif "rate limit" in last_error.lower() or "429" in last_error.lower():
        return "NOTICE: AI models are temporarily busy (Rate Limit). Please wait 60 seconds and try again."
    return f"Error (All Tiers Exhausted): {last_error}"

def distill_context(context: str) -> str:
    """Extracts critical technical points including Architecture, Optimization, and Innovation."""
    if not context or "Error" in context: return "No comparative data available."
    
    distilled = []
    
    # Mapping of headers to labels and character caps
    sections = [
        (r'## 1\. Problem & Objective(.*?)(?=##|$)', "PROBLEM GAPS", 1200),
        (r'## 2\. Methodology & Approach(.*?)(?=##|$)', "METHODOLOGY SHORTFALLS", 1200),
        (r'## 3\. Data & Evidence(.*?)(?=##|$)', "DATA DIFFERENCES", 1200),
        (r'## 4\. Results & Findings(.*?)(?=##|$)', "RESULTS COMPARISONS", 1200),
        (r'## 5\. Evaluation Method(.*?)(?=##|$)', "EVALUATION GAPS", 1200)
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

def _extract_caption(page, img_rect):
    """Finds the closest text line directly BELOW or ABOVE an image bounding box."""
    # Search 120pt above and below the image for caption text
    search_rect = fitz.Rect(img_rect.x0, img_rect.y0 - 120, img_rect.x1, img_rect.y1 + 120)
    words = page.get_text("words", clip=search_rect)
    
    # Prioritize words that start with "Fig" or "Figure"
    all_text = " ".join([w[4] for w in words])
    fig_match = re.search(r'(Fig\w*\.?\s*\d+.*)', all_text, re.IGNORECASE)
    if fig_match:
        return fig_match.group(1).strip()
        
    return all_text.strip()[:300]


def _get_surrounding_text(page, img_rect, window_words=100):
    """
    Extracts up to window_words appearing vertically ABOVE and BELOW the image rect.
    Uses PDF coordinates for precise context grouping.
    """
    words = page.get_text("words") # list of [x0, y0, x1, y1, "word", ...]
    
    # Sort words by their actual reading order (Y then X) to ensure clean context
    words.sort(key=lambda w: (w[1], w[0]))
    
    above_words = []
    below_words = []
    
    for w in words:
        word_text = w[4]
        word_y1 = w[3]
        word_y0 = w[1]
        
        # Word is effectively ABOVE the image if its bottom is above the image's top
        if word_y1 <= img_rect.y0:
            above_words.append(word_text)
        # Word is effectively BELOW the image if its top is below the image's bottom
        elif word_y0 >= img_rect.y1:
            below_words.append(word_text)
            
    context_above = " ".join(above_words[-window_words:]) # last 100 words above
    context_below = " ".join(below_words[:window_words])  # first 100 words below
    
    return f"[TEXT ABOVE FIGURE]: {context_above}\n\n[TEXT BELOW FIGURE]: {context_below}"


def parse_pdf(file_path):
    """Parses PDF and returns:
    - full text (str)
    - list of figure dicts, each containing:
        path, caption, page_num, context (surrounding text)
    """
    doc = fitz.open(file_path)
    full_text = ""
    page_texts = []
    for page in doc:
        pt = page.get_text()
        full_text += pt
        page_texts.append(pt)

    figures = []
    fig_index = 0
    
    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        page_text = page_texts[page_idx]
        
        # 1. Collect candidate regions from IMAGES
        candidate_rects = []
        for img in page.get_image_info():
            r = img.get("bbox")
            if r: candidate_rects.append(fitz.Rect(r))
            
        # 2. Collect candidate regions from DRAWINGS (Vector plots)
        # We group nearby drawings into combined rectangles
        drawings = page.get_drawings()
        for d in drawings:
            r = d.get("rect")
            if r and r.width > 50 and r.height > 50:
                candidate_rects.append(fitz.Rect(r))
        
        # 3. Merge overlapping or very close rectangles to avoid duplicate sub-component captures
        merged_rects = []
        if candidate_rects:
            candidate_rects.sort(key=lambda r: (r.y0, r.x0))
            current = candidate_rects[0]
            for i in range(1, len(candidate_rects)):
                nxt = candidate_rects[i]
                # Manually check if nxt is close to current (within 30 points)
                is_close = (nxt.x0 < current.x1 + 30 and nxt.x1 > current.x0 - 30 and
                            nxt.y0 < current.y1 + 30 and nxt.y1 > current.y0 - 30)
                
                if nxt.intersects(current) or is_close:
                    current = current | nxt # Union
                else:
                    merged_rects.append(current)
                    current = nxt
            merged_rects.append(current)

        # 4. Final Processing of Valid Regions
        for rect in merged_rects:
            # --- FILTER 1: Size check (150px minimum) ---
            if rect.width < 150 or rect.height < 150:
                continue

            # --- FILTER 2: Aspect ratio check ---
            if rect.width / max(rect.height, 1) > 6 or rect.height / max(rect.width, 1) > 6:
                continue

            # --- FILTER 3: Position check (Avoid headers/footers) ---
            if page_num == 1: continue # Skip title page logos
            page_h = page.rect.height
            if rect.y0 < 0.05 * page_h or rect.y1 > 0.95 * page_h:
                continue

            # --- CAPTURE: High-Res Snapshot (300 DPI) ---
            # Using Matrix(3,3) for 3x zoom capture
            pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(3, 3))
            img_path = os.path.join(tempfile.gettempdir(), f"fig_{page_num}_{int(rect.x0)}.png")
            pix.save(img_path)

            caption = _extract_caption(page, rect)
            context = _get_surrounding_text(page, rect)

            fig_index += 1
            figures.append({
                "path": img_path,
                "caption": caption,
                "page_num": page_num,
                "context": context,
                "figure_index": fig_index,
            })

    return full_text, figures

def chunk_text(text, size=500, overlap=0):
    """Splits text into fixed-size chunks with optional overlap."""
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunk = text[i:i + size]
        if len(chunk) > 50: # Skip tiny fragments
            chunks.append(chunk)
    return chunks

def deduplicate_chunks(chunks):
    """Removes duplicate chunks while preserving order."""
    seen = set()
    unique = []
    for c in chunks:
        if c not in seen:
            unique.append(c)
            seen.add(c)
    return unique

def rerank_papers(original_summary, papers, top_k=6):
    """Uses the embedding model to sort papers by semantic similarity to the original summary."""
    if not papers:
        return []
    
    # Compute embedding for original summary
    query_emb = embed_model.encode([original_summary])[0]
    
    # Compute embeddings for all candidate summaries
    candidates = [p.get("summary", "") for p in papers]
    cand_embs = embed_model.encode(candidates)
    
    # Calculate cosine similarity
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = cosine_similarity([query_emb], cand_embs)[0]
    
    # Pair similarities with papers and sort
    ranked = sorted(zip(papers, similarities), key=lambda x: x[1], reverse=True)
    
    # Return top K unique papers
    return [r[0] for r in ranked[:top_k]]

def store_embeddings(chunks):
    st.session_state.faiss_index = faiss.IndexFlatL2(dimension)
    st.session_state.documents = []
    embs = embed_model.encode(chunks)
    st.session_state.faiss_index.add(np.array(embs))
    st.session_state.documents.extend(chunks)

def store_figure_description(figure_index: int, description: str):
    """Embeds a figure's AI description and stores it in the shared FAISS index.
    This makes figures searchable via the Q&A RAG pipeline."""
    if not hasattr(st.session_state, 'faiss_index') or st.session_state.faiss_index is None:
        return  # No index yet, skip silently
    # Prefix so Q&A can identify this as figure-sourced context
    tagged = f"[Figure {figure_index} Analysis]: {description}"
    emb = embed_model.encode([tagged])
    st.session_state.faiss_index.add(np.array(emb))
    st.session_state.documents.append(tagged)

def retrieve(query, k=3):
    docs = st.session_state.documents
    idx = st.session_state.faiss_index
    if not docs: return []
    emb = embed_model.encode([query])
    k = min(k, len(docs))
    _, I = idx.search(np.array(emb), k)
    return [docs[i] for i in I[0] if i < len(docs)]
