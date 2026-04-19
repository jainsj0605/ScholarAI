from utils import rerank_papers
import json

def test_reranking():
    summary = "This paper presents a novel approach to Time Series Forecasting using Transformer architectures with a focus on long-term dependencies and positional encoding."
    
    # Mock papers from different sources
    meta_papers = [
        {"title": "Deep Learning for CV", "summary": "A paper about convolutional neural networks for image classification in medical imaging.", "venue": "CrossRef"},
        {"title": "Time-LLM: Time Series Forecasting", "summary": "Recent state of the art results in time series prediction using Large Language Models and transformers.", "venue": "ArXiv"},
        {"title": "Climate Change Analysis", "summary": "A study on the global impact of rising temperatures in the arctic regions.", "venue": "OpenAlex"},
        {"title": "Forecasting with Transformers", "summary": "Detailed abstract for Time Series Forecasting using Transformer architectures which is exactly what we want.", "venue": "Semantic Scholar"},
        {"title": "Quantum Computing", "summary": "Research on qubit stability and quantum error correction codes.", "venue": "ArXiv"}
    ]
    
    print("Original Summary:", summary)
    print("\nRanking papers...")
    ranked = rerank_papers(summary, meta_papers)
    
    print("\nTop 3 Ranked Papers:")
    for i, p in enumerate(ranked[:3]):
        print(f"{i+1}. {p['title']} [{p['venue']}]")

if __name__ == "__main__":
    test_reranking()
