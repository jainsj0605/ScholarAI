import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# API Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Server Configuration
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "t")
PORT = int(os.environ.get("PORT", 5000))

# LLM Models
# Primary model with automatic fallback on 429 rate limit
TEXT_MODELS = [
    "openai/gpt-oss-120b",        # primary
    "llama-3.3-70b-versatile",    # fallback
]
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Vector Search / Embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_CHUNK_SIZE = 500
DEFAULT_TOP_K = 3

# PDF Generator Layout
PAGE_W = 595
PAGE_H = 842
MARGIN = 55
TW = PAGE_W - 2 * MARGIN

# Prestigious Engineering Publishers for Search Reranking Bonus
ENGINEERING_PUBLISHERS = {
    "ieee", "springer", "elsevier", "wiley", "acm", "nature", "taylor",
    "iet", "inspec", "iospress", "hindawi", "mdpi", "sage", "emerald"
}

# Upload Path
UPLOAD_PDF_PATH = "paper.pdf"
