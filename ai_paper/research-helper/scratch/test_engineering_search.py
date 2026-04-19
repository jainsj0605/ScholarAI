import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api_search import search_crossref, search_semantic_scholar, search_openalex, normalize_venue

def test():
    query = "Amplitude Modulation"
    print(f"\n=== Engineering-Grade Hybrid Search Test: '{query}' ===\n")

    print("--- Crossref (Engineering/IEEE/Springer) ---")
    cr = search_crossref(query)
    has_abstract = [p for p in cr if p.get("has_abstract")]
    print(f"Total: {len(cr)} | With abstract: {len(has_abstract)}")
    for p in cr[:4]:
        status = "✅" if p.get("has_abstract") else "❌ (no abstract)"
        print(f"  {status} [{p['venue']}] {p['title'][:70]}")

    print("\n--- Semantic Scholar ---")
    s2 = search_semantic_scholar(query)
    has_abstract = [p for p in s2 if p.get("has_abstract")]
    print(f"Total: {len(s2)} | With abstract: {len(has_abstract)}")
    for p in s2[:4]:
        status = "✅" if p.get("has_abstract") else "❌ (no abstract)"
        print(f"  {status} [{p['venue']}] {p['title'][:70]}")

    print("\n--- Venue Normalization Test ---")
    test_venues = [
        "Institute of Electrical and Electronics Engineers",
        "IEEE Transactions on Communications",
        "Springer Nature",
        "Journal of Elsevier Publishing",
        "Wiley Online Library",
    ]
    for v in test_venues:
        print(f"  '{v[:40]}...' → '{normalize_venue(v)}'")

if __name__ == "__main__":
    test()
