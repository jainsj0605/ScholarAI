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

def get_domain_name(cat_code):
    if not cat_code: return "Research"
    if cat_code in ARXIV_CATEGORIES: return ARXIV_CATEGORIES[cat_code]
    prefix = cat_code.split('.')[0]
    if prefix == "cs": return "Computer Science"
    if prefix == "stat": return "Statistics"
    if "physics" in cat_code or cat_code.startswith("hep-"): return "Physics"
    if "math" in cat_code: return "Mathematics"
    return cat_code.upper()

def search_semantic_scholar(query):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&limit=5&fields=title,abstract,year,url,venue"
    try:
        res = requests.get(url, timeout=12)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("data", []):
                papers.append({
                    "title": item.get("title", "Untitled"),
                    "summary": item.get("abstract") or "No abstract available.",
                    "year": str(item.get("year", "")),
                    "link": item.get("url", ""),
                    "venue": item.get("venue") or "Semantic Scholar"
                })
            return papers
    except: pass
    return []

def search_openalex(query):
    url = f"https://api.openalex.org/works?search={requests.utils.quote(query)}&limit=5"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("results", []):
                p_dict = {
                    "title": item.get("display_name", "Untitled"),
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
                    p_dict["summary"] = "No abstract available."
                    
                papers.append(p_dict)
            return papers
    except: pass
    return []

def search_crossref(query):
    url = f"https://api.crossref.org/works?query={requests.utils.quote(query)}&rows=5"
    try:
        res = requests.get(url, timeout=12)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("message", {}).get("items", []):
                papers.append({
                    "title": item.get("title", ["Untitled"])[0],
                    "summary": "Engineering research record found in CrossRef. No abstract available via metadata API.",
                    "year": str(item.get("published-print", {}).get("date-parts", [[""]])[0][0]),
                    "link": item.get("URL", ""),
                    "venue": item.get("container-title", ["CrossRef"])[0]
                })
            return papers
    except: pass
    return []

def search_arxiv(query):
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
    for count in [len(words), 3, 2]:
        if count > len(words): continue
        results = perform_search(" ".join(words[:count]), sort_by_date=True)
        if results: return results
    return perform_search(cleaned_query, sort_by_date=False)
