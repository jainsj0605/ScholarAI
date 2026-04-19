import requests
import re
import time
import concurrent.futures
from urllib.parse import quote_plus

def clean_query(query):
    """Step 2: Sanitizes the AI-generated topic into high-density keywords."""
    if not query: return ""
    # Remove common preamble patterns (keywords:, topics:, etc)
    cleaned = re.sub(r'^(keywords?|topics?|terms?|search?|query?)\s*:\s*', '', query, flags=re.IGNORECASE)
    # Remove smart quotes and normal quotes
    cleaned = cleaned.replace('\u201c', '').replace('\u201d', '').replace('"', '').replace("'", "")
    # Remove generic characters that break Boolean logic
    cleaned = re.sub(r'[\[\](){}]', '', cleaned)
    return cleaned.strip()

# Venue name normalization map — long institutional names to short badge labels
_VENUE_MAP = [
    ("institute of electrical and electronics engineers", "IEEE"),
    ("ieee",           "IEEE"),
    ("springer",       "Springer"),
    ("elsevier",       "Elsevier"),
    ("wiley",          "Wiley"),
    ("acm",            "ACM"),
    ("nature",         "Nature"),
    ("taylor",         "Taylor & Francis"),
    ("iet ",           "IET"),
    ("hindawi",        "Hindawi"),
    ("mdpi",           "MDPI"),
    ("sage",           "SAGE"),
    ("emerald",        "Emerald"),
    ("informs",        "INFORMS"),
    ("american physical", "APS"),
    ("royal society",  "Royal Society"),
    ("iopscience",     "IOP"),
    ("oxford",         "Oxford Academic"),
    ("cambridge",      "Cambridge UP"),
    ("plos",           "PLOS"),
    ("frontiers",      "Frontiers"),
    ("crossref",       "CrossRef"),
    ("semantic scholar", "Semantic Scholar"),
    ("openalex",       "OpenAlex"),
]

def normalize_venue(venue: str) -> str:
    """Shorten long publisher names to a clean badge label."""
    if not venue:
        return "Academic Source"
    v = venue.strip()
    vl = v.lower()
    for pattern, label in _VENUE_MAP:
        if pattern in vl:
            return label
    # Truncate very long names
    return v if len(v) <= 30 else v[:27] + "..."

def search_arxiv(query, sort_by="submittedDate"):
    """Step 3: Tiered Search for ArXiv with Boolean logic and Sorting."""
    if not query: return []
    # query is already formatted as all:K1+AND+all:K2 from the tiered graph node
    url = f"https://export.arxiv.org/api/query?search_query={query}&start=0&max_results=10"
    
    if sort_by == "submittedDate":
        url += "&sortBy=submittedDate&sortOrder=descending"
    else:
        url += "&sortBy=relevance&sortOrder=descending"

    try:
        res = requests.get(url, timeout=15)
        papers = []
        if res.status_code == 200:
            entries = re.findall(r'<entry>(.*?)</entry>', res.text, re.DOTALL)
            for entry in entries:
                t_m = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                s_m = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                id_m = re.search(r'<id>(.*?)</id>', entry, re.DOTALL)
                p_m = re.search(r'<published>(.*?)</published>', entry, re.DOTALL)
                
                if t_m and s_m and id_m:
                    abs_text = re.sub(r'\s+', ' ', s_m.group(1)).strip()
                    papers.append({
                        "title": re.sub(r'\s+', ' ', t_m.group(1)).strip(),
                        "summary": abs_text,
                        "has_abstract": len(abs_text) > 50,
                        "year": p_m.group(1)[:4] if p_m else "",
                        "link": id_m.group(1).strip(),
                        "venue": "ArXiv"
                    })
        return papers
    except:
        return []

def search_semantic_scholar(query):
    cleaned = clean_query(query)
    q_encoded = quote_plus(cleaned)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q_encoded}&limit=10&fields=title,abstract,year,url,venue"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("data", []):
                if not item.get("title") or not item.get("url"): continue
                abstract = item.get("abstract") or ""
                papers.append({
                    "title": item.get("title"),
                    "summary": abstract,
                    "has_abstract": len(abstract) > 50,
                    "year": str(item.get("year", "")),
                    "link": item.get("url"),
                    "venue": normalize_venue(item.get("venue") or "Semantic Scholar")
                })
            return papers
    except: pass
    return []

def search_openalex(query):
    """Step 4: Parallel Search on OpenAlex with Inverted Index Reconstruction."""
    cleaned = clean_query(query)
    words = cleaned.split()
    q_encoded = "+".join(words)
    url = f"https://api.openalex.org/works?search={q_encoded}&filter=has_abstract:true&per_page=10&mailto=admin@scholarai.app"
    try:
        res = requests.get(url, timeout=18)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("results", []):
                title = item.get("display_name")
                link = item.get("doi") or f"https://openalex.org/{item.get('id').split('/')[-1]}"
                if not title or not link: continue
                
                # Reconstruct abstract from inverted index
                abstract = ""
                idx = item.get("abstract_inverted_index")
                if idx:
                    word_positions = {}
                    for word, positions in idx.items():
                        for pos in positions: word_positions[pos] = word
                    sorted_words = [word_positions[i] for i in sorted(word_positions.keys())]
                    abstract = " ".join(sorted_words)
                
                papers.append({
                    "title": title,
                    "summary": abstract,
                    "has_abstract": len(abstract) > 50,
                    "year": str(item.get("publication_year", "")),
                    "link": link,
                    "venue": normalize_venue((item.get("primary_location") or {}).get("source", {}).get("display_name", "OpenAlex"))
                })
            return papers
    except: pass
    return []

def search_crossref(query):
    cleaned = clean_query(query)
    q_encoded = quote_plus(cleaned)
    url = f"https://api.crossref.org/works?query={q_encoded}&rows=10"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("message", {}).get("items", []):
                title_list = item.get("title", [])
                link = item.get("URL")
                if not title_list or not link: continue
                
                # Crossref abstracts are often missing or in JATS XML
                abstract = item.get("abstract", "")
                abstract = re.sub(r'<[^>]+>', '', abstract) # simple tag strip
                
                # Extract DOI for abstract enrichment later
                doi = item.get("DOI", "")
                
                papers.append({
                    "title": title_list[0],
                    "summary": abstract,
                    "has_abstract": len(abstract) > 50,
                    "year": str(item.get("published-print", {}).get("date-parts", [[""]])[0][0]),
                    "link": link,
                    "doi": doi,
                    "venue": normalize_venue(item.get("container-title", ["CrossRef"])[0])
                })
            return papers
    except: pass
    return []


def _fetch_abstract_by_doi(doi):
    """Try to fetch an abstract from Semantic Scholar using a DOI."""
    if not doi:
        return None
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=abstract"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            abstract = res.json().get("abstract", "")
            if abstract and len(abstract) > 50:
                return abstract
    except:
        pass
    return None


def _fetch_abstract_by_title(title):
    """Fallback: search Semantic Scholar by title to find the abstract."""
    if not title:
        return None
    try:
        q = quote_plus(title)
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit=1&fields=abstract,title"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json().get("data", [])
            if data:
                abstract = data[0].get("abstract", "")
                if abstract and len(abstract) > 50:
                    return abstract
    except:
        pass
    return None


def enrich_missing_abstracts(papers):
    """For papers without abstracts, try to fetch from Semantic Scholar.
    
    Uses DOI-based lookup first (fast, precise), then falls back to
    title-based search. Runs in parallel for speed.
    """
    papers_needing_enrichment = [
        (i, p) for i, p in enumerate(papers) if not p.get("has_abstract")
    ]
    
    if not papers_needing_enrichment:
        return papers  # all papers already have abstracts
    
    def enrich_one(idx_paper):
        idx, paper = idx_paper
        # Try DOI first (precise match)
        doi = paper.get("doi", "")
        abstract = _fetch_abstract_by_doi(doi)
        
        # Fallback to title search
        if not abstract:
            abstract = _fetch_abstract_by_title(paper.get("title", ""))
        
        return idx, abstract
    
    # Run enrichment in parallel (max 4 at a time to respect rate limits)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(enrich_one, ip) for ip in papers_needing_enrichment]
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, abstract = future.result()
                if abstract:
                    papers[idx]["summary"] = abstract
                    papers[idx]["has_abstract"] = True
                    papers[idx]["enriched"] = True  # mark as enriched
            except:
                pass
    
    return papers
