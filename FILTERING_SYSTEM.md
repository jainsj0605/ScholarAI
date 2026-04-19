# Strict Academic Filtering System

## Overview

ScholarAI implements a **two-level strict academic filtering system** that ensures only high-quality research papers with substantial abstracts are used for comparative analysis.

## Problem Solved

Academic search APIs often return papers with:
- Missing abstracts
- Generic placeholder text ("no abstract available")
- Incomplete metadata
- Inverted index formats (OpenAlex) that need decoding

These low-quality results lead to "No specific data or evidence" messages in comparative analysis.

## Solution: Two-Level Filtering

### Level 1: Source-Level Filtering (api_search.py)

Each search function filters papers **at the source** before returning results:

#### 1. OpenAlex Abstract Re-Assembly Script

**Problem**: OpenAlex stores abstracts in an inverted index format:
```json
{
  "hello": [0, 5],
  "world": [1],
  "research": [2, 6]
}
```

**Solution**: `reassemble_openalex_abstract()` function
- Decodes the complex inverted index
- Reconstructs abstract text chronologically
- Ensures AI can read it properly

```python
def reassemble_openalex_abstract(inverted_index):
    """
    Decodes OpenAlex's inverted index and glues the abstract
    text back together chronologically.
    """
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    
    word_positions.sort(key=lambda x: x[0])
    return " ".join([word for pos, word in word_positions])
```

**Filtering Rules**:
- ✅ Only include if reconstructed abstract ≥ 100 characters
- ❌ Discard papers without inverted index data

#### 2. Semantic Scholar Strict Filtering

```python
def search_semantic_scholar(query):
    """
    Only returns papers with valid, substantial abstracts (100+ characters).
    Papers without abstracts are immediately discarded.
    """
```

**Filtering Rules**:
- ✅ Abstract must exist and be ≥ 100 characters
- ❌ Discard if abstract contains "no abstract available"
- ❌ Discard if abstract contains "abstract not available"
- ❌ Immediately move to next paper if invalid

#### 3. CrossRef Strict Filtering

```python
def search_crossref(query):
    """
    Only returns papers with valid, substantial abstracts (100+ characters).
    Papers without abstracts are immediately discarded.
    """
```

**Filtering Rules**:
- ✅ Abstract must exist and be ≥ 100 characters
- ❌ Discard if abstract contains "no abstract available"
- ❌ Discard if abstract contains "abstract not available"
- ❌ Immediately move to next paper if invalid

#### 4. ArXiv Strict Filtering

```python
def search_arxiv(query):
    """
    ArXiv provides full abstracts by default.
    Ensures only papers with substantial content are returned.
    """
```

**Filtering Rules**:
- ✅ Abstract must be ≥ 100 characters
- ❌ Discard papers with short abstracts
- ✅ Store arxiv_id for potential full-text download

### Level 2: Application-Level Gatekeeper (graphs.py)

The `node_arxiv_search()` function applies **additional strict filtering** after collecting results from all sources:

```python
def node_arxiv_search(state):
    """
    Multi-Engine Academic Search with Strict Quality Filtering
    
    Implements a STRICT ACADEMIC GATEKEEPER that:
    1. Searches across 4 academic databases
    2. Immediately discards papers without valid abstracts
    3. Filters out generic placeholder text
    4. Enriches ArXiv papers with full-text content
    5. Re-ranks results by semantic relevance
    """
```

**Filtering Rules**:

1. **RULE 1**: Must have title and substantial abstract (100+ chars)
   ```python
   if not p.get("title") or not p.get("summary") or len(p["summary"].strip()) < 100:
       continue  # Immediately discard and pull next paper
   ```

2. **RULE 2**: Skip generic placeholder text
   ```python
   if "not provided" in summary_lower or "metadata indicates" in summary_lower:
       continue  # Immediately discard and pull next paper
   ```

3. **RULE 3**: Skip "no abstract available" messages
   ```python
   if "no abstract available" in summary_lower or "abstract not available" in summary_lower:
       continue  # Immediately discard and pull next paper
   ```

4. **RULE 4**: Deduplicate by title
   ```python
   slug = re.sub(r'[^a-z0-9]', '', p['title'].lower())
   if slug and slug not in seen:
       unique.append(p)
       seen.add(slug)
   ```

## Filtering Flow Diagram

```
User Query
    ↓
┌─────────────────────────────────────────┐
│  LEVEL 1: Source-Level Filtering        │
├─────────────────────────────────────────┤
│  ArXiv API          → Filter (100+ chars)│
│  Semantic Scholar   → Filter (100+ chars)│
│  OpenAlex          → Reassemble + Filter │
│  CrossRef          → Filter (100+ chars) │
└─────────────────────────────────────────┘
    ↓
Combined Results (all with valid abstracts)
    ↓
┌─────────────────────────────────────────┐
│  LEVEL 2: Application Gatekeeper        │
├─────────────────────────────────────────┤
│  RULE 1: Check length (100+ chars)      │
│  RULE 2: No placeholder text            │
│  RULE 3: No "not available" messages    │
│  RULE 4: Deduplicate by title           │
└─────────────────────────────────────────┘
    ↓
Unique, High-Quality Papers
    ↓
Semantic Re-Ranking (top 6 most relevant)
    ↓
ArXiv Full-Text Enrichment (if available)
    ↓
Final Papers for Comparative Analysis
```

## Benefits

### Before Filtering System
```
[2019] Impact of Motion-Induced Antenna Pointing Errors
No specific data or evidence is provided in the abstract, as the paper 
is indexed under a relevant academic venue for evaluation.
```

### After Filtering System
```
**[2019] Impact of Motion-Induced Antenna Pointing Errors**
- **Data/Evidence Used**: Mobile satellite communication scenarios with 
  motion-induced antenna pointing errors and Doppler effects
- **Scale/Scope**: Various pointing error magnitudes and satellite 
  velocity conditions analyzed through simulation
- **Source**: Theoretical analysis validated through Monte Carlo 
  simulation of LEO satellite communication links
```

## Key Statistics

- **Minimum Abstract Length**: 100 characters (strictly enforced)
- **Search Sources**: 4 academic databases
- **Filtering Levels**: 2 (source + application)
- **Filtering Rules**: 4 strict rules at application level
- **Final Papers**: Top 6 most relevant (after semantic re-ranking)
- **ArXiv Enhancement**: Full-text extraction (first 4-5 pages, up to 8000 chars)

## Implementation Files

1. **api_search.py**: Source-level filtering for all 4 search engines
   - `reassemble_openalex_abstract()`: OpenAlex inverted index decoder
   - `search_semantic_scholar()`: Semantic Scholar with strict filtering
   - `search_openalex()`: OpenAlex with re-assembly + filtering
   - `search_crossref()`: CrossRef with strict filtering
   - `search_arxiv()`: ArXiv with strict filtering

2. **graphs.py**: Application-level gatekeeper
   - `node_arxiv_search()`: Multi-engine search with 4-rule filtering
   - ArXiv full-text enrichment integration

## Testing the System

To verify the filtering system is working:

1. **Check Paper Count**: Should see 6 high-quality papers (or fewer if not enough quality papers found)
2. **Check Abstract Length**: All papers should have substantial abstracts
3. **Check for Placeholders**: No "not provided" or "no abstract available" messages
4. **Check Year Display**: All papers show **[YEAR]** prominently
5. **Check Details**: Comparative analysis should have specific technical details

## Maintenance

When adding new search sources:
1. Implement source-level filtering (100+ char minimum)
2. Check for "no abstract available" messages
3. Handle API-specific abstract formats (like OpenAlex inverted index)
4. Add to `node_arxiv_search()` source list
5. Update priority order if needed
