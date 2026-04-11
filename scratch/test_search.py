import os, re, requests, json
from urllib.parse import quote_plus
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

# Sample summary for RAG paper (2312.10997)
summary = """
This paper introduces Retrieval-Augmented Generation (RAG), a model that combines pre-trained parametric (seq2seq) and non-parametric (dense vector search over Wikipedia) memory. RAG models generate text by retrieving relevant documents and then using them as context for a generator. The approach is tested on knowledge-intensive NLP tasks like Open-Domain Q&A, showing state-of-the-art results.
"""

def llm(prompt):
    res = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content

# 1. Topic Extraction
prompt = f"Extract the main research topic (3-4 essential terms) from this summary. Return ONLY the keywords separated by spaces. No quotes, no preamble, and no symbols.\n\nSummary:\n{summary}"
topic = llm(prompt).strip().replace('"', '').replace("'", "")
print(f"Extracted Topic: '{topic}'")

# 2. Search logic (Legacy)
def perform_search(q_text, sort_by_date=True):
    words = [w for w in re.split(r'\s+', q_text) if w]
    if not words: return []
    q = "+AND+".join([f"all:{quote_plus(w)}" for w in words])
    url = f"https://export.arxiv.org/api/query?search_query={q}&start=0&max_results=5"
    if sort_by_date: url += "&sortBy=submittedDate&sortOrder=descending"
    else: url += "&sortBy=relevance"
    print(f"Searching: {url}")
    res = requests.get(url)
    # Just print titles for now
    titles = re.findall(r'<title>(.*?)</title>', res.text, re.DOTALL)
    return [t.strip().replace('\n', ' ') for t in titles[1:]] # Skip the feed title

print("Results (Date Sort):", perform_search(topic, True))
print("Results (Relevance Sort):", perform_search(topic, False))
