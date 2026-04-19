import requests
import re
from urllib.parse import quote_plus
import fitz  # PyMuPDF
import tempfile
import os
from bs4 import BeautifulSoup

# ArXiv category mapping
ARXIV_CATEGORIES = {
    "cs.AI": "Artificial Intelligence", "cs.CV": "Computer Vision",
    "cs.LG": "Machine Learning", "cs.CL": "Computation and Language",
    "cs.NE": "Neural/Evolutionary Computing", "cs.RO": "Robotics",
    "stat.ML": "Machine Learning", "cs.CR": "Cryptography",
    "cs.IR": "Information Retrieval", "cs.SE": "Software Engineering",
    "cs.DC": "Distributed Computing", "cs.HC": "Human-Computer Interaction",
}

# ENGINEERING-GRADE VENUE WEIGHTING
# Prestigious engineering publishers get higher relevance scores
PRESTIGIOUS_VENUES = {
    # IEEE Family (highest priority for engineering)
    "ieee": 10.0,
    "institute of electrical and electronics engineers": 10.0,
    "ieee transactions": 10.0,
    "ieee communications": 10.0,
    "ieee journal": 10.0,
    
    # Springer (high priority)
    "springer": 8.0,
    "springer nature": 8.0,
    
    # Elsevier (high priority)
    "elsevier": 8.0,
    
    # Wiley (high priority)
    "wiley": 7.0,
    "john wiley": 7.0,
    
    # ACM (high priority for CS/Engineering)
    "acm": 7.0,
    "association for computing machinery": 7.0,
    
    # Nature (high priority)
    "nature": 9.0,
    
    # Science (high priority)
    "science": 9.0,
    "aaas": 9.0,
    
    # ArXiv (good for recent work)
    "arxiv": 6.0,
    
    # Other reputable publishers
    "taylor & francis": 6.0,
    "sage": 5.0,
    "mdpi": 5.0,
}

# Venue normalization for clean display
VENUE_NORMALIZATION = {
    "institute of electrical and electronics engineers": "IEEE",
    "ieee transactions on communications": "IEEE Trans. Communications",
    "ieee transactions on wireless communications": "IEEE Trans. Wireless",
    "ieee transactions on signal processing": "IEEE Trans. Signal Processing",
    "ieee transactions on information theory": "IEEE Trans. Information Theory",
    "ieee communications letters": "IEEE Commun. Letters",
    "ieee journal on selected areas in communications": "IEEE JSAC",
    "springer nature": "Springer",
    "elsevier bv": "Elsevier",
    "john wiley & sons": "Wiley",
    "wiley-blackwell": "Wiley",
    "association for computing machinery": "ACM",
    "association for computing machinery (acm)": "ACM",
}


def normalize_venue(venue):
    """
    Normalize venue names for clean display.
    
    Converts long publisher names to short, recognizable forms.
    Example: "Institute of Electrical and Electronics Engineers" -> "IEEE"
    """
    if not venue:
        return "Unknown"
    
    venue_lower = venue.lower().strip()
    
    # Check exact matches first
    for key, normalized in VENUE_NORMALIZATION.items():
        if key in venue_lower:
            return normalized
    
    # Check for IEEE patterns
    if "ieee" in venue_lower:
        if "transactions" in venue_lower or "trans." in venue_lower:
            # Keep the specific transaction name
            return venue.replace("Institute of Electrical and Electronics Engineers", "IEEE")
        return "IEEE"
    
    # Check for other patterns
    if "springer" in venue_lower:
        return "Springer"
    if "elsevier" in venue_lower:
        return "Elsevier"
    if "wiley" in venue_lower:
        return "Wiley"
    if "acm" in venue_lower:
        return "ACM"
    if "nature" in venue_lower:
        return "Nature"
    if "science" in venue_lower:
        return "Science"
    
    # Return original if no match (but capitalize properly)
    return venue[:50]  # Limit length


def calculate_venue_score(venue):
    """
    Calculate relevance bonus based on venue prestige.
    
    Engineering papers from IEEE, Springer, Elsevier get higher scores.
    This ensures technical papers are prioritized over general noise.
    
    Returns:
        float: Relevance bonus (0.0 to 10.0)
    """
    if not venue:
        return 0.0
    
    venue_lower = venue.lower().strip()
    
    # Check for prestigious venues
    for key, score in PRESTIGIOUS_VENUES.items():
        if key in venue_lower:
            return score
    
    # Default score for unknown venues
    return 1.0


def fetch_arxiv_fulltext(arxiv_url, max_chars=8000):
    """
    Attempts to download and extract text from an ArXiv PDF.
    Returns extended content if successful, otherwise returns None.
    """
    try:
        # Extract ArXiv ID from URL (e.g., http://arxiv.org/abs/2301.12345v1 -> 2301.12345)
        arxiv_id_match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', arxiv_url)
        if not arxiv_id_match:
            return None
        
        arxiv_id = arxiv_id_match.group(1)
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        # Download PDF with timeout
        response = requests.get(pdf_url, timeout=30, stream=True)
        if response.status_code != 200:
            return None
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        # Extract text from first few pages (Introduction, Method sections)
        doc = fitz.open(tmp_path)
        text_parts = []
        total_chars = 0
        
        # Extract from first 4-5 pages (usually covers intro, method, and some results)
        for page_num in range(min(5, len(doc))):
            page = doc[page_num]
            page_text = page.get_text()
            text_parts.append(page_text)
            total_chars += len(page_text)
            
            if total_chars >= max_chars:
                break
        
        doc.close()
        os.unlink(tmp_path)  # Clean up temp file
        
        full_text = "\n".join(text_parts)[:max_chars]
        return full_text if len(full_text) > 500 else None
        
    except Exception as e:
        return None


def clean_query(query):
    # Remove metadata prefixes and quotes
    cleaned = re.sub(r'^(Topic|Keywords|Search):\s*', '', query, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r'^[“"‘\']*(.*?)[”"’\']*$', r'\1', cleaned).strip()
    # List of common stop-words to remove
    stop_words = {'a', 'an', 'the', 'and', 'or', 'of', 'for', 'in', 'on', 'at', 'by', 'with', 'about', 'to', 'from'}
    words = [w for w in re.split(r'\s+', cleaned) if w.lower() not in stop_words]
    return " ".join(words)

def search_semantic_scholar(query):
    """
    Search Semantic Scholar API with strict abstract filtering and venue scoring.
    
    Only returns papers with valid, substantial abstracts (100+ characters).
    Papers from prestigious venues (IEEE, Springer, Elsevier) get higher scores.
    """
    # Increase limit to get more candidates with real abstracts
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(query)}&limit=20&fields=title,abstract,year,url,venue"
    try:
        res = requests.get(url, timeout=12)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("data", []):
                title = item.get("title", "Untitled")
                abstract = item.get("abstract")
                venue = item.get("venue") or "Semantic Scholar"
                
                # STRICT ACADEMIC FILTERING: Immediately discard papers without valid abstracts
                if not abstract or len(abstract.strip()) < 100:
                    continue
                
                # Check for "no abstract available" messages
                if "no abstract available" in abstract.lower() or "abstract not available" in abstract.lower():
                    continue
                
                # Calculate venue prestige score
                venue_score = calculate_venue_score(venue)
                normalized_venue = normalize_venue(venue)
                
                papers.append({
                    "title": title,
                    "summary": abstract,
                    "year": str(item.get("year", "")),
                    "link": item.get("url", ""),
                    "venue": normalized_venue,
                    "venue_raw": venue,  # Keep original for deduplication
                    "venue_score": venue_score  # For ranking
                })
            return papers
    except: pass
    return []

def reassemble_openalex_abstract(inverted_index):
    """
    Abstract Re-Assembly Script for OpenAlex Inverted Index
    
    OpenAlex stores abstracts in an inverted index format where:
    - Keys are words
    - Values are lists of positions where the word appears
    
    Example: {"hello": [0], "world": [1]} -> "hello world"
    
    This function decodes the complex inverted index and reconstructs
    the abstract text chronologically so the AI can read it properly.
    
    Args:
        inverted_index (dict): OpenAlex abstract_inverted_index
        
    Returns:
        str: Reconstructed abstract text in proper reading order
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    
    try:
        # Create list of (position, word) tuples
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        
        # Sort by position to get chronological order
        word_positions.sort(key=lambda x: x[0])
        
        # Reconstruct the abstract
        reconstructed = " ".join([word for pos, word in word_positions])
        
        return reconstructed
    except Exception as e:
        return ""


def search_openalex(query):
    """
    Search OpenAlex API with automatic abstract reconstruction and venue scoring.
    
    OpenAlex uses an inverted index format for abstracts. This function
    automatically decodes and reassembles the abstract text chronologically.
    Papers from prestigious venues get higher relevance scores.
    """
    url = f"https://api.openalex.org/works?search={requests.utils.quote(query)}&limit=15"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("results", []):
                title = item.get("display_name", "Untitled")
                venue = (item.get("primary_location") or {}).get("source", {}).get("display_name", "OpenAlex")
                
                # Decode OpenAlex's inverted index abstract format
                inverted_idx = item.get("abstract_inverted_index")
                if inverted_idx:
                    # Use the re-assembly script to decode the abstract
                    reconstructed = reassemble_openalex_abstract(inverted_idx)
                    
                    # STRICT ACADEMIC FILTERING: Only include papers with substantial abstracts
                    if len(reconstructed.strip()) >= 100:
                        # Calculate venue prestige score
                        venue_score = calculate_venue_score(venue)
                        normalized_venue = normalize_venue(venue)
                        
                        papers.append({
                            "title": title,
                            "summary": reconstructed[:1500],
                            "year": str(item.get("publication_year", "")),
                            "link": item.get("doi") or f"https://openalex.org/{item.get('id').split('/')[-1]}",
                            "venue": normalized_venue,
                            "venue_raw": venue,
                            "venue_score": venue_score
                        })
            return papers
    except: pass
    return []

def fetch_crossref_fulltext(doi_url, max_chars=8000):
    """
    Attempts to fetch full-text content from CrossRef DOI links.
    
    CrossRef provides DOI links that may lead to:
    1. Open access PDFs
    2. Publisher HTML pages with full text
    3. Repository links with accessible content
    
    Returns extended content if successful, otherwise returns None.
    """
    if not doi_url:
        return None
    
    try:
        # Try to resolve DOI and fetch content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        
        # First, try to get the DOI itself which might be a PDF
        response = requests.get(doi_url, headers=headers, timeout=15, allow_redirects=True)
        
        # Check if we got a PDF
        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' in content_type:
            # Save and extract PDF text
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
            
            try:
                doc = fitz.open(tmp_path)
                text_parts = []
                total_chars = 0
                
                # Extract from first 4-5 pages
                for page_num in range(min(5, len(doc))):
                    page = doc[page_num]
                    page_text = page.get_text()
                    text_parts.append(page_text)
                    total_chars += len(page_text)
                    
                    if total_chars >= max_chars:
                        break
                
                doc.close()
                os.unlink(tmp_path)
                
                full_text = "\n".join(text_parts)[:max_chars]
                return full_text if len(full_text) > 500 else None
            except:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                return None
        
        # If not PDF, try to extract text from HTML
        elif 'text/html' in content_type:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Try to find main content areas (common patterns)
            content_areas = soup.find_all(['article', 'main', 'div'], 
                                         class_=re.compile(r'(abstract|content|article|body|text)', re.I))
            
            if content_areas:
                text = ' '.join([area.get_text(separator=' ', strip=True) for area in content_areas])
                text = re.sub(r'\s+', ' ', text).strip()[:max_chars]
                return text if len(text) > 500 else None
        
        return None
        
    except Exception as e:
        return None


def search_crossref(query):
    """
    Search CrossRef API with strict abstract filtering, full-text fetching, and venue scoring.
    
    For papers with DOI links, attempts to fetch full-text content to enrich
    the abstract. Papers from prestigious venues get higher relevance scores.
    """
    url = f"https://api.crossref.org/works?query={requests.utils.quote(query)}&rows=20"
    try:
        res = requests.get(url, timeout=12)
        if res.status_code == 200:
            data = res.json()
            papers = []
            for item in data.get("message", {}).get("items", []):
                title = item.get("title", ["Untitled"])[0]
                abstract = item.get("abstract", "")
                doi_url = item.get("URL", "") or item.get("DOI", "")
                venue = item.get("container-title", ["CrossRef"])[0]
                
                # If DOI exists, convert to URL format
                if doi_url and not doi_url.startswith("http"):
                    doi_url = f"https://doi.org/{doi_url}"
                
                # STRICT ACADEMIC FILTERING: Check if we have valid abstract
                has_valid_abstract = abstract and len(abstract.strip()) >= 100
                
                # Skip "no abstract available" messages
                if abstract and ("no abstract available" in abstract.lower() or 
                               "abstract not available" in abstract.lower()):
                    has_valid_abstract = False
                
                # Only include papers with valid abstracts OR DOI links for fetching
                if has_valid_abstract or doi_url:
                    # Calculate venue prestige score
                    venue_score = calculate_venue_score(venue)
                    normalized_venue = normalize_venue(venue)
                    
                    papers.append({
                        "title": title,
                        "summary": abstract if has_valid_abstract else f"[Abstract pending - will fetch from DOI] {title}",
                        "year": str(item.get("published-print", {}).get("date-parts", [[""]])[0][0]) or 
                               str(item.get("published-online", {}).get("date-parts", [[""]])[0][0]),
                        "link": doi_url,
                        "venue": normalized_venue,
                        "venue_raw": venue,
                        "venue_score": venue_score,
                        "needs_fulltext": not has_valid_abstract
                    })
            return papers
    except: pass
    return []

def search_arxiv(query):
    """
    Search ArXiv API with strict abstract filtering and venue scoring.
    
    ArXiv provides full abstracts by default. This function ensures
    only papers with substantial content are returned.
    """
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
                        summary = re.sub(r'\s+', ' ', s_m.group(1)).strip()
                        
                        # STRICT ACADEMIC FILTERING: Ensure abstract is substantial
                        if len(summary) < 100:
                            continue
                        
                        arxiv_id = id_m.group(1).strip() if id_m else ""
                        venue = "ArXiv"
                        venue_score = calculate_venue_score(venue)
                        
                        papers.append({
                            "title": re.sub(r'\s+', ' ', t_m.group(1)).strip(),
                            "summary": summary,
                            "year": p_m.group(1)[:4] if p_m else "",
                            "link": arxiv_id,
                            "venue": venue,
                            "venue_raw": venue,
                            "venue_score": venue_score,
                            "arxiv_id": arxiv_id
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
