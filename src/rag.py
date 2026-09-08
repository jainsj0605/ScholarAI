import os
from typing import List, Tuple
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K

class VectorStore:
    """Manages document chunk embeddings and FAISS vector indexing."""
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, dimension: int = EMBEDDING_DIM):
        self.dimension = dimension
        self.model_name = model_name
        self._embed_model = None
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents: List[str] = []

    @property
    def embed_model(self):
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(self.model_name)
        return self._embed_model

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []

    def add_chunks(self, chunks: List[str]):
        if not chunks:
            return
        embs = self.embed_model.encode(chunks)
        self.index.add(np.array(embs, dtype=np.float32))
        self.documents.extend(chunks)

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> List[str]:
        if not self.documents:
            return []
        emb = self.embed_model.encode([query])
        top_k = min(k, len(self.documents))
        _, indices = self.index.search(np.array(emb, dtype=np.float32), top_k)
        return [self.documents[i] for i in indices[0] if 0 <= i < len(self.documents)]

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.embed_model.encode(texts)

# Global singleton instance for application session
vector_store = VectorStore()

def parse_pdf(file_path: str) -> Tuple[str, List[str]]:
    """Extract full text and figures from a PDF document.
    
    Returns:
        text (str): Extracted plain text across all pages.
        images (List[str]): File paths of extracted figure images.
    """
    doc = fitz.open(file_path)
    text = ""
    images = []
    for i, page in enumerate(doc):
        text += page.get_text()
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_path = f"temp_{xref}.png"
            with open(img_path, "wb") as f:
                f.write(base_image["image"])
            images.append(img_path)
    doc.close()
    return text, images

def chunk_text(text: str, size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """Break text into sliding or fixed size chunks."""
    if not text:
        return []
    return [text[i:i + size] for i in range(0, len(text), size)]

def store_embeddings(chunks: List[str]):
    """Reset vector store and index given text chunks."""
    vector_store.clear()
    vector_store.add_chunks(chunks)

def retrieve(query: str, k: int = DEFAULT_TOP_K) -> List[str]:
    """Retrieve top-k semantically relevant chunks for a query."""
    return vector_store.retrieve(query, k=k)
