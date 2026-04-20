import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
print(f"Using API Key: {api_key[:10]}...")

try:
    client = Groq(api_key=api_key)
    models = client.models.list()
    print("Available Models:")
    for m in models.data:
        print(f" - {m.id}")
except Exception as e:
    print(f"Error: {e}")
