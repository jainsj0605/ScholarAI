import requests
from urllib.parse import quote
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

def search_arxiv(query):
    if not query or len(query.strip()) < 3:
        print("Query too short")
        return []
        
    q = quote(query.strip())
    # Test with both http and https, though we use https in app.py
    url = f"https://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending"
    
    print(f"Testing URL: {url}")
    
    # Configure retries
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    try:
        # 5s connect timeout, 15s read timeout for test
        res = session.get(url, timeout=(5, 15))
        papers = []
        print(f"Status Code: {res.status_code}")
        if res.status_code == 200:
            entries = res.text.split("<entry>")
            print(f"Entries found: {len(entries)-1}")
            if len(entries) > 1:
                for entry in entries[1:]:
                    try:
                        title   = entry.split("<title>")[1].split("</title>")[0].strip()
                        # summary = entry.split("<summary>")[1].split("</summary>")[0].strip()
                        year    = entry.split("<published>")[1].split("</published>")[0][:4] if "<published>" in entry else ""
                        papers.append({"title": title, "year": year})
                    except (IndexError, ValueError):
                        continue
        return papers
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    queries = ["Machine Learning in Healthcare", "Quantum Computing Algorithms", "asdfghjkl12345"]
    for q in queries:
        print(f"\n--- Searching for: {q} ---")
        results = search_arxiv(q)
        for p in results:
            print(f"- [{p['year']}] {p['title']}")
