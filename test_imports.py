"""
Test script to verify all imports work correctly
"""

print("Testing imports...")

try:
    print("1. Importing from api_search...")
    from api_search import (
        search_arxiv, 
        search_crossref, 
        search_openalex, 
        search_semantic_scholar, 
        fetch_arxiv_fulltext, 
        fetch_crossref_fulltext,
        normalize_venue,
        calculate_venue_score
    )
    print("   ✅ api_search imports successful")
except Exception as e:
    print(f"   ❌ api_search import failed: {e}")
    exit(1)

try:
    print("2. Importing from graphs...")
    from graphs import build_graphs, analyze_single_image
    print("   ✅ graphs imports successful")
except Exception as e:
    print(f"   ❌ graphs import failed: {e}")
    exit(1)

try:
    print("3. Importing from utils...")
    from utils import parse_pdf, chunk_text, store_embeddings
    print("   ✅ utils imports successful")
except Exception as e:
    print(f"   ❌ utils import failed: {e}")
    exit(1)

try:
    print("4. Importing from config...")
    from config import client, TEXT_MODEL
    print("   ✅ config imports successful")
except Exception as e:
    print(f"   ❌ config import failed: {e}")
    exit(1)

print("\n✅ All imports successful!")
print("\nTesting venue functions...")

# Test venue normalization
test_venues = [
    "Institute of Electrical and Electronics Engineers",
    "IEEE Transactions on Communications",
    "Springer Nature",
    "Elsevier BV"
]

for venue in test_venues:
    normalized = normalize_venue(venue)
    score = calculate_venue_score(venue)
    print(f"  {venue[:40]:40} → {normalized:20} (score: {score})")

print("\n✅ All tests passed!")
