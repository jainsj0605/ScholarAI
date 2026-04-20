import os
import sys
from unittest.mock import MagicMock

# Mock Streamlit secrets and environment before any imports
import streamlit as st
st.secrets = {"GROQ_API_KEY": "test_key"}
st.session_state = MagicMock()

# Add current dir to path to import utils
sys.path.append(os.getcwd())
import utils

def test_extraction(pdf_path):
    print(f"Testing extraction on: {pdf_path}")
    try:
        text, figures = utils.parse_pdf(pdf_path)
        print(f"Total Figures Detected: {len(figures)}")
        for i, fig in enumerate(figures):
            print(f"--- Fig {i+1} ---")
            print(f"Page: {fig['page_num']}")
            print(f"Caption: {fig['caption']}")
            print(f"Path: {fig['path']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test on the sample paper only for speed
    test_extraction("paper.pdf")
