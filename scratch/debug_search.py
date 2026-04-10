import requests
import os

def search_arxiv(query):
    q = query.replace(" ", "+")
    url = f"http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending"
    print(f"DEBUG: Searching URL: {url}")
    try:
        res = requests.get(url, timeout=10)
        papers = []
        if res.status_code == 200:
            print(f"DEBUG: Status 200, Content Length: {len(res.text)}")
            entries = res.text.split("<entry>")
            print(f"DEBUG: Number of potential entries found: {len(entries)-1}")
            for entry in entries[1:]:
                title   = entry.split("<title>")[1].split("</title>")[0].strip()
                summary = entry.split("<summary>")[1].split("</summary>")[0].strip()
                year    = entry.split("<published>")[1].split("</published>")[0][:4] if "<published>" in entry else ""
                papers.append({"title": title, "summary": summary, "year": year})
        return papers
    except Exception as e:
        return [{"title": "Search failed", "summary": str(e), "year": ""}]

# Test cases
test_queries = [
    "Machine Learning",
    "Deep Learning in Medical Imaging",
    'Artificial Intelligence "Natural Language Processing"',
    "a" * 100, # Very long query
    "", # Empty query
    "quantum computing @#$%", # Special characters
]

for query in test_queries:
    print(f"\n--- Testing Query: '{query}' ---")
    results = search_arxiv(query)
    print(f"Results count: {len(results)}")
    if results:
        print(f"First Result Title: {results[0]['title']}")
