import requests
import re
from urllib.parse import quote_plus

# ArXiv category mapping
ARXIV_CATEGORIES = {
    "cs.AI": "Artificial Intelligence", "cs.CV": "Computer Vision",
    "cs.LG": "Machine Learning", "cs.CL": "Computation and Language",
    "cs.NE": "Neural/Evolutionary Computing", "cs.RO": "Robotics",
    "stat.ML": "Machine Learning", "cs.CR": "Cryptography",
    "cs.IR": "Information Retrieval", "cs.SE": "Software Engineering",
    "cs.DC": "Distributed Computing", "cs.HC": "Human-Computer Interaction",
}

def clean_query(query):
    # Remove metadata prefixes and quotes
    cleaned = re.sub(r'^(Topic|Keywords|Search):\s*', '', query, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'^[“"‘\']*(.*?)[”"’\']*$', r'\1', cleaned).strip()
    # List of common stop-words to remove
    stop_words = {'a', 'an', 'the', 'and', 'or', 'of', 'for', 'in', 'on', 'at', 'by', 'with', 'about', 'to', 'from'}
    words = [w for w in re.split(r'\s+', cleaned) if w.lower() not in stop_words]
    return " ".join(words)

def search_semantic_scholar(query):
    # Increase limit slightly for re-ranking
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&limit=10&fields=title,abstract,year,url,venue"
    try:
        res = requests.get(url, timeout=12)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("data", []):
                title = item.get("title", "Untitled")
                summary = item.get("abstract") or f"This Semantic Scholar record for '{title}' covers research related to the technical keywords. While a full abstract was not provided in the primary feed, the metadata indicates high technical relevance."
                papers.append({
                    "title": title,
                    "summary": summary,
                    "year": str(item.get("year", "")),
                    "link": item.get("url", ""),
                    "venue": item.get("venue") or "Semantic Scholar"
                })
            return papers
    except: pass
    return []

def search_openalex(query):
    url = f"https://api.openalex.org/works?search={requests.utils.quote(query)}&limit=8"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("results", []):
                title = item.get("display_name", "Untitled")
                p_dict = {
                    "title": title,
                    "year": str(item.get("publication_year", "")),
                    "link": item.get("doi") or f"https://openalex.org/{item.get('id').split('/')[-1]}",
                    "venue": (item.get("primary_location") or {}).get("source", {}).get("display_name", "OpenAlex")
                }
                
                # Reconstruct OpenAlex inverted index abstract
                idx = item.get("abstract_inverted_index")
                if idx:
                    words = []
                    for word, pos in idx.items():
                        for p in pos: words.append((p, word))
                    p_dict["summary"] = " ".join([w[1] for w in sorted(words)])[:1500]
                else:
                    p_dict["summary"] = f"Abstract for '{title}' is not provided via the OpenAlex API in this technical record. Please check the linked repository or publication page for the full text."
                    
                papers.append(p_dict)
            return papers
    except: pass
    return []

def search_crossref(query):
    url = f"https://api.crossref.org/works?query={requests.utils.quote(query)}&rows=8"
    try:
        res = requests.get(url, timeout=12)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("message", {}).get("items", []):
                title = item.get("title", ["Untitled"])[0]
                papers.append({
                    "title": title,
                    "summary": f"This research record for '{title}' was found via CrossRef. While the specific abstract metadata was not provided in the search response, the paper is indexed under the relevant academic venue for evaluation.",
                    "year": str(item.get("published-print", {}).get("date-parts", [[""]])[0][0]),
                    "link": item.get("URL", ""),
                    "venue": item.get("container-title", ["CrossRef"])[0]
                })
            return papers
    except: pass
    return []

def search_arxiv(query):
    cleaned_query = clean_query(query)
    if not cleaned_query: return []

    def perform_search(q_text):
        words = [w for w in re.split(r'\s+', q_text) if w]
        if not words: return []
        
        # Search in title OR abstract specifically
        q_parts = [f"(ti:{quote_plus(w)}+OR+abs:{quote_plus(w)})" for w in words]
        q = "+AND+".join(q_parts)
        
        url = f"https://export.arxiv.org/api/query?search_query={q}&start=0&max_results=10&sortBy=relevance"
        try:
            res = requests.get(url, timeout=25)
            papers = []
            if res.status_code == 200:
                entries = re.findall(r'<entry>(.*?)</entry>', res.text, re.DOTALL)
                for entry in entries:
                    t_m = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                    s_m = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                    id_m = re.search(r'<id>(.*?)</id>', entry, re.DOTALL)
                    p_m = re.search(r'<published>(.*?)</published>', entry, re.DOTALL)
                    if t_m and s_m:
                        papers.append({
                            "title": re.sub(r'\s+', ' ', t_m.group(1)).strip(),
                            "summary": re.sub(r'\s+', ' ', s_m.group(1)).strip(),
                            "year": p_m.group(1)[:4] if p_m else "",
                            "link": id_m.group(1).strip() if id_m else "",
                            "venue": "ArXiv"
                        })
            return papers
        except: return []

    words = [w for w in re.split(r'\s+', cleaned_query) if w]
    # Try searching with all keywords, then fall back to top 3 if too restrictive
    for count in [len(words), 3]:
        if count > len(words): continue
        results = perform_search(" ".join(words[:count]))
        if results: return results
    return perform_search(cleaned_query)
