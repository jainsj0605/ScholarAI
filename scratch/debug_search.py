import requests
import re
from urllib.parse import quote_plus

def search_arxiv(query):
    # Basic cleaning - handle "Topic:", smart quotes, and labels
    cleaned_query = re.sub(r'^(Topic|Keywords|Search):\s*', '', query, flags=re.IGNORECASE).strip()
    cleaned_query = re.sub(r'^[“"‘\']*(.*?)[”"’\']*$', r'\1', cleaned_query).strip()
    
    if not cleaned_query: return []

    def perform_search(q_text, sort_by_date=True):
        words = [w for w in re.split(r'\s+', q_text) if w]
        if not words: return []
        q = "+AND+".join([f"all:{quote_plus(w)}" for w in words])
        
        url = f"https://export.arxiv.org/api/query?search_query={q}&start=0&max_results=5"
        if sort_by_date: url += "&sortBy=submittedDate&sortOrder=descending"
        else: url += "&sortBy=relevance"
            
        try:
            res = requests.get(url, timeout=25)
            papers = []
            if res.status_code == 200:
                entries = re.findall(r'<entry>(.*?)</entry>', res.text, re.DOTALL)
                for entry in entries:
                    t_m = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                    if t_m:
                        papers.append({"title": re.sub(r'\s+', ' ', t_m.group(1)).strip()})
            return papers
        except: return []

    words = [w for w in re.split(r'\s+', cleaned_query) if w]
    print(f"Keywords: {words}")
    
    # Tier 1: Iterative AND (Date Sorted)
    for count in [len(words), 3, 2]:
        if count > len(words): continue
        print(f"Trying Iterative AND (Date) with {count} words...")
        results = perform_search(" ".join(words[:count]), sort_by_date=True)
        if results: 
            print(f"Match found in Tier 1 (Count: {count})")
            return results
        
    # Tier 2: Iterative AND (Relevance Sorted)
    for count in [len(words), 2]:
        if count > len(words): continue
        print(f"Trying Iterative AND (Relevance) with {count} words...")
        results = perform_search(" ".join(words[:count]), sort_by_date=False)
        if results: 
            print(f"Match found in Tier 2 (Count: {count})")
            return results
    
    return []

# Test
query = "Amplitude Modulation ModulationIndex Overmodulation"
print(f"Final Test for: {query}")
results = search_arxiv(query)
print("\nRESULTS FOUND:")
for r in results:
    print(f"- {r['title']}")
