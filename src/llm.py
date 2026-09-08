import base64
import time
from groq import Groq
from src.config import GROQ_API_KEY, TEXT_MODELS, VISION_MODEL

# Initialize Groq client
_client = None

def get_client() -> Groq:
    """Lazy initialize Groq client."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Please set it in your .env file.")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client

def encode_image(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def llm(prompt: str) -> str:
    """Call Groq LLM with automatic model fallback on rate-limit (429)."""
    client = get_client()
    for model in TEXT_MODELS:
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048
            )
            return res.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                print(f"[Rate limit] {model} exhausted, falling back to next model...")
                time.sleep(2)
                continue
            return f"Error: {e}"
    return "Error: All models are currently rate-limited. Please wait a few minutes and try again."

def analyze_figure(img_path: str) -> str:
    """Analyze a research paper figure using the vision model."""
    client = get_client()
    try:
        b64 = encode_image(img_path)
        res = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this research paper figure using markdown:\n"
                            "- **Type**: What kind of figure?\n"
                            "- **Key Insights**: What does it show?\n"
                            "- **Importance**: Why is it significant?"
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    }
                ]
            }]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"Vision error: {e}"
