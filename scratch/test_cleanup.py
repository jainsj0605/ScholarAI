def mock_cleaner(raw_response):
    marker = "## Executive Summary"
    if marker in raw_response:
        return raw_response[raw_response.find(marker):].strip()
    return raw_response

# Test 1: Echoed text before summary
test1 = "This paper is about LEO networks... ## Executive Summary\nAnalysis starts here."
print(f"Test 1 Results:\n'{mock_cleaner(test1)}'")

# Test 2: Clean response
test2 = "## Executive Summary\nOnly clean text."
print(f"Test 2 Results:\n'{mock_cleaner(test2)}'")

# Test 1 should have removed the first sentence.
