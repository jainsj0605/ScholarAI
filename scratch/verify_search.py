import requests
import re
from urllib.parse import quote_plus

def search_arxiv_new(query):
    # Basic cleaning
    cleaned_query = re.sub(r'^(Topic|Keywords):\s*', '', query, flags=re.IGNORECASE).strip()
    cleaned_query = cleaned_query.strip('"\'')
    
    if not cleaned_query:
        return []

    q = quote_plus(cleaned_query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending"
    print(f"DEBUG: Searching URL: {url}")
    
    try:
        res = requests.get(url, timeout=10)
        papers = []
        if res.status_code == 200:
            # More robust parsing using regex
            entries = re.findall(r'<entry>(.*?)</entry>', res.text, re.DOTALL)
            print(f"DEBUG: Entries found: {len(entries)}")
            for entry in entries:
                title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                published_match = re.search(r'<published>(.*?)</published>', entry, re.DOTALL)
                
                if title_match and summary_match:
                    title = re.sub(r'\s+', ' ', title_match.group(1)).strip()
                    summary = re.sub(r'\s+', ' ', summary_match.group(1)).strip()
                    year = published_match.group(1)[:4] if published_match else ""
                    papers.append({"title": title, "summary": summary, "year": year})
        return papers
    except Exception as e:
        return [{"title": "Search failed", "summary": str(e), "year": ""}]

# Test cases including special characters
test_queries = [
    "Machine Learning",
    "Keywords: Deep Learning",
    'Topic: "Large Language Models"',
    "AI & Robotics", # Contains &
    "Quantum #Computing", # Contains #
]

for query in test_queries:
    print(f"\n--- Testing Query: '{query}' ---")
    results = search_arxiv_new(query)
    print(f"Results count: {len(results)}")
    if results:
        print(f"First Result Title: {results[0]['title']}")
