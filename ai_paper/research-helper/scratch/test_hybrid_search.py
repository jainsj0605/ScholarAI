import sys
import os

# Porting check
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api_search import search_crossref, search_arxiv, search_openalex

def test_engineering_query():
    query = "Amplitude Modulation"
    print(f"Testing Query: {query}")
    
    print("\n--- Testing Crossref ---")
    cr = search_crossref(query)
    print(f"Found: {len(cr)}")
    for p in cr[:2]:
        print(f" - [{p['venue']}] {p['title']}")
        
    print("\n--- Testing ArXiv (Boolean) ---")
    ar = search_arxiv("all:Amplitude+AND+all:Modulation")
    print(f"Found: {len(ar)}")
    for p in ar[:2]:
        print(f" - [{p['venue']}] {p['title']}")

if __name__ == "__main__":
    test_engineering_query()
