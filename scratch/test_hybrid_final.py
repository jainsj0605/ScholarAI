import requests
import re
from urllib.parse import quote_plus

def search_crossref(query):
    url = f"https://api.crossref.org/works?query={quote_plus(query)}&rows=5"
    try:
        res = requests.get(url, timeout=12)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("message", {}).get("items", []):
                papers.append({
                    "title": item.get("title", ["Untitled"])[0],
                    "venue": item.get("container-title", ["CrossRef"])[0]
                })
            return papers
    except: pass
    return []

def search_openalex(query):
    url = f"https://api.openalex.org/works?search={quote_plus(query)}&limit=5"
    try:
        res = requests.get(url, timeout=12)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("results", []):
                papers.append({
                    "title": item.get("display_name", "Untitled"),
                    "venue": (item.get("primary_location") or {}).get("source", {}).get("display_name", "OpenAlex")
                })
            return papers
    except: pass
    return []

# Test
query = "amplitude modulation"
print(f"Testing Hybrid Discovery for: {query}")
cr = search_crossref(query)
oa = search_openalex(query)

print("\nCROSSREF RESULTS:")
for p in cr: print(f"- {p['title']} [{p['venue']}]")

print("\nOPENALEX RESULTS:")
for p in oa: print(f"- {p['title']} [{p['venue']}]")
