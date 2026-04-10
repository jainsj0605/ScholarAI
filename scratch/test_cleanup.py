import re

def clean_query(query):
    # Basic cleaning - handle "Topic:", smart quotes, and labels
    cleaned_query = re.sub(r'^(Topic|Keywords|Search):\s*', '', query, flags=re.IGNORECASE).strip()
    # Handle both standard and "smart" quotes “ ” ‘ ’
    cleaned_query = re.sub(r'^[“"‘\']*(.*?)[”"’\']*$', r'\1', cleaned_query).strip()
    return cleaned_query

test_cases = [
    'Topic: "Machine Learning"',
    '“Retrieval Augmented Generation”',
    'keywords: ‘Quantum Computing’',
    '"Doppler prediction error power control LEO satellite uplink"'
]

for tc in test_cases:
    print(f"Original: {tc} -> Cleaned: {clean_query(tc)}")

# Fallback test simulation
def fallback_sim(query):
    cleaned = clean_query(query)
    words = cleaned.split()
    if len(words) > 3:
        fallback = " ".join(words[:3])
        return fallback
    return cleaned

long_query = "Doppler prediction error power control LEO satellite uplink"
print(f"\nLong Query: {long_query}")
print(f"Fallback: {fallback_sim(long_query)}")
