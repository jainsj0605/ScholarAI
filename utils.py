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

def llm(prompt: str, model: str = TEXT_MODEL, max_chars: int = 24000, disable_failsafe: bool = False) -> str:
    # Truncate prompt to prevent 413 or TPM errors
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars] + "\n\n[Context truncated due to size limits...]"
        
    # Enforce Streamlit Math Rendering
    prompt += "\n\nMANDATORY MATH RULE: Use strictly '$$' for block equations and '$' for inline. NEVER use '\\[' or '\\]'."
    
    current_model = model
    try:
        # Initial Attempt
        res = client.chat.completions.create(
            model=current_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000, # Increased strictly to allow for heavy multi-paragraph generations
            temperature=0.3  # Slightly lower for more precision
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


def _get_surrounding_text(full_text, page_text, window_words=100):
    """Extracts up to window_words words before and after the page's text block."""
    # Locate this page's text block inside the full document text
    start_idx = full_text.find(page_text[:200])
    if start_idx == -1:
        # Fallback: just return the page text itself (first 600 chars)
        return page_text[:600]
    words = full_text.split()
    # Find approximate word index
    prefix = full_text[:start_idx]
    word_start = len(prefix.split())
    page_words = len(page_text.split())
    before = words[max(0, word_start - window_words): word_start]
    after = words[word_start + page_words: word_start + page_words + window_words]
    return " ".join(before) + " [...] " + " ".join(after)


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

            # Smart filtering: skip logos, banners, and tiny icons
            w = base_image.get("width", 0)
            h = base_image.get("height", 0)
            if w < 300 or h < 300:
                continue
            if w / max(h, 1) > 4 or h / max(w, 1) > 4:
                continue

            img_bytes = base_image["image"]
            img_path = os.path.join(tempfile.gettempdir(), f"temp_{xref}.png")
            with open(img_path, "wb") as f:
                f.write(img_bytes)

            # Get image bounding rect on the page
            img_rect = None
            for item in page.get_image_info():
                if item.get("xref") == xref:
                    r = item.get("bbox")
                    if r:
                        img_rect = fitz.Rect(r)
                    break

            # Extract caption and surrounding context
            caption = _extract_caption(page, img_rect) if img_rect else ""
            context = _get_surrounding_text(full_text, page_text)

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
