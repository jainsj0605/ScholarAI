import requests
import json

def test_openalex(query):
    url = f"https://api.openalex.org/works?search={requests.utils.quote(query)}&limit=10"
    print(f"Testing OpenAlex: {url}")
    try:
        res = requests.get(url, timeout=15)
        data = res.json()
        print(f"Results Found: {len(data.get('results', []))}")
        for item in data.get("results", []):
            print(f"- {item.get('display_name')}")
    except Exception as e:
        print(f"Error: {e}")

test_openalex("amplitude modulation")
