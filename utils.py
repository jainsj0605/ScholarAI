import os
import re
import base64
import tempfile
import fitz
import faiss
import numpy as np
import streamlit as st

from config import client, TEXT_MODEL, FALLBACK_MODEL, embed_model, dimension

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def llm(prompt: str, system_prompt: str = None, model: str = TEXT_MODEL, max_chars: int = 24000, disable_failsafe: bool = False) -> str:
    # Truncate prompt to prevent 413 or TPM errors
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars] + "\n\n[Context truncated due to size limits...]"
        
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    current_model = model
    try:
        # Initial Attempt
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
        # Fallback Logic: Only switch if the primary hits a rate limit
        err_msg = str(e).lower()
        if "429" in err_msg or "limit" in err_msg or "413" in err_msg:
            try:
                # When falling back, use a smaller max_chars if it was a 413
                fallback_limit = 10000 if "413" in err_msg else max_chars
                res = client.chat.completions.create(
                    model=FALLBACK_MODEL,
                    messages=[{"role": "user", "content": prompt[:fallback_limit]}],
                    max_tokens=4000,
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
    """Finds the closest text line directly BELOW an image bounding box."""
    caption_lines = []
    # Search a region 80pt below the image for caption text
    search_rect = fitz.Rect(img_rect.x0, img_rect.y1, img_rect.x1, img_rect.y1 + 80)
    words = page.get_text("words", clip=search_rect)
    if words:
        caption_lines = [w[4] for w in words]  # word text is index 4
    return " ".join(caption_lines).strip()


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
        path, caption, page_num, context (surrounding text), bytes
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

        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)

            # Get image bounding rect on the page FIRST for position checks
            img_rect = None
            for item in page.get_image_info():
                if item.get("xref") == xref:
                    r = item.get("bbox")
                    if r:
                        img_rect = fitz.Rect(r)
                    break

            # --- FILTER 1: Size check (300x300 minimum) ---
            w = base_image.get("width", 0)
            h = base_image.get("height", 0)
            if w < 300 or h < 300:
                continue

            # --- FILTER 2: Aspect ratio check ---
            if w / max(h, 1) > 4 or h / max(w, 1) > 4:
                continue

            # --- FILTER 3: Position check ---
            # Skip title page completely
            if page_num == 1:
                continue
            # Skip if image sits in extreme top/bottom header & footer zones
            if img_rect:
                page_h = page.rect.height
                if img_rect.y0 < 0.08 * page_h or img_rect.y1 > 0.92 * page_h:
                    continue

            # --- FILTER 4: Caption Extraction (No mandatory labeling) ---
            # We no longer skip images that don't say "Figure"
            caption = _extract_caption(page, img_rect) if img_rect else ""

            # --- Only Genuine Figures Remain ---
            img_bytes = base_image["image"]
            img_path = os.path.join(tempfile.gettempdir(), f"temp_{xref}.png")
            with open(img_path, "wb") as f:
                f.write(img_bytes)

            if img_rect:
                context = _get_surrounding_text(page, img_rect)
            else:
                # Fallback if spatial context cannot be determined
                context = f"[PAGE CONTEXT]: {page.get_text()[:600]}..."

            fig_index += 1
            figures.append({
                "path": img_path,
                "caption": caption,
                "page_num": page_num,
                "context": context,
                "figure_index": fig_index,
            })

    return full_text, figures

def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

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
