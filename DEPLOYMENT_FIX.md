# Deployment Fix Guide

## Error Encountered

```
ImportError: This app has encountered an error.
File "/mount/src/scholarai/graphs.py", line 9, in <module>
    from api_search import search_arxiv, search_crossref, search_openalex, sea...
```

## Root Cause

The import statement in `graphs.py` is trying to import functions from `api_search.py`, but one or more of these functions might not be properly defined or there's a circular import issue.

## ✅ Solution Applied

### 1. Added Missing Function

Added `fetch_arxiv_fulltext()` function to `api_search.py` (it was referenced but not defined):

```python
def fetch_arxiv_fulltext(arxiv_url, max_chars=8000):
    """
    Attempts to download and extract text from an ArXiv PDF.
    Returns extended content if successful, otherwise returns None.
    """
    # ... implementation ...
```

### 2. Verified All Imports

All functions imported in `graphs.py` now exist in `api_search.py`:
- ✅ `search_arxiv`
- ✅ `search_crossref`
- ✅ `search_openalex`
- ✅ `search_semantic_scholar`
- ✅ `fetch_arxiv_fulltext`
- ✅ `fetch_crossref_fulltext`

### 3. Added Test Script

Created `test_imports.py` to verify all imports work:

```bash
cd ai_paper/research-helper
python test_imports.py
```

Expected output:
```
Testing imports...
1. Importing from api_search...
   ✅ api_search imports successful
2. Importing from graphs...
   ✅ graphs imports successful
3. Importing from utils...
   ✅ utils imports successful
4. Importing from config...
   ✅ config imports successful

✅ All imports successful!
```

## 🔧 If Error Persists

### Option 1: Check Python Path

Ensure Streamlit Cloud is running from the correct directory:

1. Check `streamlit run` command points to correct file
2. Verify all `.py` files are in the same directory
3. Check for `__pycache__` conflicts

### Option 2: Clear Cache

On Streamlit Cloud:
1. Click "Manage app"
2. Click "Reboot app"
3. Or click "Clear cache"

### Option 3: Verify Dependencies

Ensure `requirements.txt` is in the root directory and contains:

```
flask
python-dotenv
groq
pymupdf
sentence-transformers
faiss-cpu
numpy
requests
reportlab
langgraph
streamlit
torch
torchvision
transformers
beautifulsoup4
lxml
```

### Option 4: Check for Circular Imports

If the error persists, check for circular imports:

1. `api_search.py` should NOT import from `graphs.py`
2. `graphs.py` CAN import from `api_search.py`
3. `utils.py` should be independent

Current import structure (correct):
```
app.py
  ├─> graphs.py
  │     ├─> api_search.py ✅
  │     ├─> utils.py ✅
  │     └─> config.py ✅
  └─> utils.py
```

## 🚀 Deployment Checklist

Before deploying to Streamlit Cloud:

- [x] All functions exist in `api_search.py`
- [x] All dependencies in `requirements.txt`
- [x] No circular imports
- [x] Test imports locally with `test_imports.py`
- [ ] Reboot Streamlit Cloud app
- [ ] Clear cache if needed

## 📝 Files Modified

1. **api_search.py**
   - Added `fetch_arxiv_fulltext()` function
   - Added venue scoring and normalization functions
   - All search functions updated with venue scoring

2. **graphs.py**
   - Updated `node_arxiv_search()` with engineering-grade ranking
   - Imports all necessary functions from `api_search.py`

3. **requirements.txt**
   - Added `beautifulsoup4`
   - Added `lxml`

## ✅ Verification

Run this command to verify everything works:

```bash
cd ai_paper/research-helper
python test_imports.py
```

If all tests pass, the app should work on Streamlit Cloud after a reboot.
