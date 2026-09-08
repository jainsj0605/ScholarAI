import re
import requests
import concurrent.futures
from urllib.parse import quote_plus
import numpy as np

from src.config import ENGINEERING_PUBLISHERS
from src.rag import vector_store

def clean_query(query: str) -> str:
    """Sanitize topic or search keywords for Boolean and API compatibility."""
    if not query:
        return ""
    # Remove common preamble patterns
    cleaned = re.sub(r'^(keywords?|topics?|terms?|search?|query?)\s*:\s*', '', query, flags=re.IGNORECASE)
    # Remove smart and normal quotes
    cleaned = cleaned.replace('\u201c', '').replace('\u201d', '').replace('"', '').replace("'", "")
    # Remove brackets that break Boolean parsing
    cleaned = re.sub(r'[\[\](){}]', '', cleaned)
    return cleaned.strip()

# Normalization lookup table for publishers & academic venues
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
    return v if len(v) <= 30 else v[:27] + "..."

def search_arxiv(query: str, sort_by: str = "submittedDate") -> list:
    """Query ArXiv API with boolean keywords and sorting."""
    if not query:
        return []
    url = f"https://export.arxiv.org/api/query?search_query={query}&start=0&max_results=15"
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
                        "has_abstract": len(abs_text) > 20,
                        "year": p_m.group(1)[:4] if p_m else "",
                        "link": id_m.group(1).strip(),
                        "venue": "ArXiv"
                    })
        return papers
    except Exception:
        return []

def search_semantic_scholar(query: str) -> list:
    """Query Semantic Scholar Graph API (supports optional SEMANTIC_SCHOLAR_API_KEY)."""
    import os
    cleaned = clean_query(query)
    q_encoded = quote_plus(cleaned)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q_encoded}&limit=15&fields=title,abstract,year,url,venue"
    try:
        headers = {"User-Agent": "ScholarAI/1.0 (mailto:admin@scholarai.app)"}
        s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if s2_key:
            headers["x-api-key"] = s2_key

        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("data", []):
                if not item.get("title") or not item.get("url"):
                    continue
                abstract = item.get("abstract") or ""
                papers.append({
                    "title": item.get("title"),
                    "summary": abstract,
                    "has_abstract": len(abstract) > 20,
                    "year": str(item.get("year", "")),
                    "link": item.get("url"),
                    "venue": normalize_venue(item.get("venue") or "Semantic Scholar")
                })
            return papers
    except Exception:
        pass
    return []

def search_openalex(query: str) -> list:
    """Query OpenAlex API and reconstruct abstracts from inverted index."""
    cleaned = clean_query(query)
    words = cleaned.split()
    q_encoded = "+".join(words)
    url = f"https://api.openalex.org/works?search={q_encoded}&filter=has_abstract:true&per_page=15&mailto=admin@scholarai.app"
    try:
        headers = {"User-Agent": "ScholarAI/1.0 (mailto:admin@scholarai.app)"}
        res = requests.get(url, headers=headers, timeout=18)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("results", []):
                title = item.get("display_name")
                link = item.get("doi") or f"https://openalex.org/{item.get('id').split('/')[-1]}"
                if not title or not link:
                    continue

                abstract = ""
                idx = item.get("abstract_inverted_index")
                if idx:
                    word_positions = {}
                    for word, positions in idx.items():
                        for pos in positions:
                            word_positions[pos] = word
                    sorted_words = [word_positions[i] for i in sorted(word_positions.keys())]
                    abstract = " ".join(sorted_words)

                primary_loc = item.get("primary_location") or {}
                source = primary_loc.get("source") or {}
                venue_name = source.get("display_name") or "OpenAlex"

                papers.append({
                    "title": title,
                    "summary": abstract,
                    "has_abstract": len(abstract) > 20,
                    "year": str(item.get("publication_year", "")),
                    "link": link,
                    "venue": normalize_venue(venue_name)
                })
            return papers
    except Exception:
        pass
    return []

def search_crossref(query: str) -> list:
    """Query CrossRef API."""
    cleaned = clean_query(query)
    q_encoded = quote_plus(cleaned)
    url = f"https://api.crossref.org/works?query={q_encoded}&rows=15"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("message", {}).get("items", []):
                title_list = item.get("title", [])
                link = item.get("URL")
                if not title_list or not link:
                    continue

                abstract = item.get("abstract", "")
                abstract = re.sub(r'<[^>]+>', '', abstract)
                doi = item.get("DOI", "")

                papers.append({
                    "title": title_list[0],
                    "summary": abstract,
                    "has_abstract": len(abstract) > 20,
                    "year": str(item.get("published-print", {}).get("date-parts", [[""]])[0][0]),
                    "link": link,
                    "doi": doi,
                    "venue": normalize_venue(item.get("container-title", ["CrossRef"])[0])
                })
            return papers
    except Exception:
        pass
    return []

def _fetch_abstract_by_doi(doi: str):
    """Fetch abstract from Semantic Scholar using DOI."""
    if not doi:
        return None
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=abstract"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            abstract = res.json().get("abstract", "")
            if abstract and len(abstract) > 50:
                return abstract
    except Exception:
        pass
    return None

def _fetch_abstract_by_title(title: str):
    """Fallback: search Semantic Scholar by title for paper abstract."""
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
    except Exception:
        pass
    return None

def enrich_missing_abstracts(papers: list) -> list:
    """Enrich papers that lack abstracts via parallel Semantic Scholar lookups."""
    papers_needing_enrichment = [
        (i, p) for i, p in enumerate(papers) if not p.get("has_abstract")
    ]
    if not papers_needing_enrichment:
        return papers

    def enrich_one(idx_paper):
        idx, paper = idx_paper
        doi = paper.get("doi", "")
        abstract = _fetch_abstract_by_doi(doi)
        if not abstract:
            abstract = _fetch_abstract_by_title(paper.get("title", ""))
        return idx, abstract

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(enrich_one, ip) for ip in papers_needing_enrichment]
        for future in concurrent.futures.as_completed(futures):
            try:
                idx, abstract = future.result()
                if abstract:
                    papers[idx]["summary"] = abstract
                    papers[idx]["has_abstract"] = True
                    papers[idx]["enriched"] = True
            except Exception:
                pass
    return papers

def _engineering_bonus(paper: dict) -> float:
    """Return +0.10 relevance boost if published by a prestigious engineering venue."""
    venue = paper.get("venue", "").lower()
    if any(pub in venue for pub in ENGINEERING_PUBLISHERS):
        return 0.10
    return 0.0

def semantic_rerank(query_summary: str, candidate_list: list) -> list:
    """Sort papers by cosine similarity to the paper summary + engineering publisher bonus."""
    if not candidate_list or not query_summary:
        return candidate_list
    try:
        safe_candidates = [p for p in candidate_list if p.get('title')]
        if not safe_candidates:
            return []

        texts = [f"{p['title']} {p.get('summary', '')}" for p in safe_candidates]
        query_emb = vector_store.encode([query_summary])[0]
        candidate_embs = vector_store.encode(texts)

        norm_q = np.linalg.norm(query_emb)
        for i, p in enumerate(safe_candidates):
            emb = candidate_embs[i]
            norm_e = np.linalg.norm(emb)
            base_score = np.dot(query_emb, emb) / (norm_q * norm_e) if (norm_q > 0 and norm_e > 0) else 0.0
            p["relevance_score"] = float(base_score) + _engineering_bonus(p)

        safe_candidates.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        top_score = safe_candidates[0].get("relevance_score", 0)
        dynamic_threshold = max(0.15, min(0.6, top_score * 0.75))

        filtered = [p for p in safe_candidates if p.get("relevance_score", 0) >= dynamic_threshold]
        if len(filtered) < 5:
            return safe_candidates[:5]
        return filtered
    except Exception as e:
        print(f"Reranking error: {e}")
        return candidate_list[:6]
