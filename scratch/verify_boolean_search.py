import requests
import re
from urllib.parse import quote_plus

def perform_search(q_text, sort_by_date=True):
    # Format as strict boolean: all:term1+AND+all:term2
    words = [w for w in re.split(r'\s+', q_text) if w]
    if not words: return []
    
    # Build strict Boolean query
    q = "+AND+".join([f"all:{quote_plus(w)}" for w in words])
    
    url = f"https://export.arxiv.org/api/query?search_query={q}&start=0&max_results=5"
    if sort_by_date:
        url += "&sortBy=submittedDate&sortOrder=descending"
    else:
        url += "&sortBy=relevance"
        
    print(f"DEBUG URL: {url}")
    res = requests.get(url, timeout=25)
    return re.findall('<entry>.*?<title>(.*?)</title>', res.text, re.DOTALL)

# Test the exact problematic query
query = "retrieval augmented generation large language models"
print(f"Testing Query: {query}")
titles = perform_search(query)
if not titles:
    print("No strict results, trying relevance fallback...")
    titles = perform_search(query, sort_by_date=False)

print("\nRESULTS FOUND:")
for t in titles:
    print(f"- {t.strip()}")
