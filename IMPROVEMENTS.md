# Comparative Analysis Improvements

## Problem Identified
The comparative analysis was showing "Not Reported" or "No specific data or evidence" for many papers because:
1. APIs were returning papers without abstracts
2. Generic placeholder text was being used instead of real abstracts
3. The LLM couldn't extract technical details from placeholder text

## Solutions Implemented

### 1. Stricter Abstract Quality Filtering
- **Semantic Scholar**: Increased limit to 20 papers, only keep papers with abstracts ≥100 characters
- **OpenAlex**: Increased limit to 15 papers, only include papers with reconstructed abstracts ≥100 characters
- **CrossRef**: Increased limit to 15 papers, only include if real abstracts exist (≥100 chars)
- **ArXiv**: Unchanged (already provides full abstracts)

### 2. Removed Generic Placeholder Text
- Eliminated fallback messages like "This research record was found via CrossRef..."
- Papers without real abstracts are now excluded entirely
- Added filter to skip papers with "not provided" or "metadata indicates" in summaries

### 3. Enhanced Search Quality
- Prioritized sources: ArXiv > Semantic Scholar > OpenAlex > CrossRef
- Stricter filtering in `node_arxiv_search`: minimum 100 chars, no placeholders
- Better deduplication to avoid showing the same paper multiple times

### 4. Improved Comparison Prompts
Enhanced all 5 comparison nodes with:
- **More context**: Increased summary length from 1500 to full abstract
- **Explicit extraction instructions**: Added keywords to look for in each category
- **Inference guidance**: Instructed LLM to infer details when explicit statements are missing
- **Reduced "Not Reported"**: Only use when abstract is completely uninformative

#### Specific Improvements by Section:

**Problem & Objective**
- Extract research gaps, motivations, target applications
- Infer from methodology/results if problem statement is implicit

**Methodology & Approach**
- Look for: "propose", "develop", "design", "implement", "analyze"
- Extract algorithms, architectures, frameworks, models

**Data & Evidence**
- Look for: "dataset", "simulation", "experiment", "benchmark"
- Describe data types even if specific names aren't mentioned

**Results & Findings**
- Look for: "achieve", "improve", "reduce", "%", "dB", "rate"
- Extract all quantitative metrics and comparative statements

**Evaluation Method**
- Look for: "evaluate", "validate", "metric", "baseline", "versus"
- Infer evaluation from results section if not explicit

## Expected Outcomes
- Fewer "Not Reported" messages in comparative analysis
- More detailed technical comparisons with specific data points
- Higher quality related papers with substantial abstracts
- Better extraction of implicit information from abstracts
