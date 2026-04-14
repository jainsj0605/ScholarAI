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
