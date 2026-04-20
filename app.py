import tempfile
import re
import streamlit as st

from utils import parse_pdf, chunk_text, store_embeddings
from graphs import build_graphs, analyze_single_image

upload_graph, compare_graph, improve_graph, qa_graph = build_graphs()

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="ScholarAI", layout="wide")

def check_limits(state_dict):
    """Checks for the CRITICAL daily limit message or temporary NOTICE in any state field."""
    for val in state_dict.values():
        if not isinstance(val, str): continue
        
        if "CRITICAL: All AI models have reached their daily token limits" in val:
            st.error("🚫 **Daily Token Limit Reached**")
            st.info("The Groq Free Tier has reached its daily ceiling for all available models (120B, 70B, and 8B).", icon="ℹ️")
            st.warning("Please try again in 24 hours or sign in with a different API key to continue your research.", icon="⚠️")
            st.stop()
            
        elif "NOTICE: AI models are temporarily busy" in val:
            st.warning("⏳ **AI Models Busy (Rate Limit)**")
            st.info("You are submitting requests too quickly for the free tier. Please wait **60 seconds** and click the button again.", icon="⏱️")
            st.stop()

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .paper-card {
        background: #1a1f2e; border: 1px solid #2a3347; border-radius: 12px;
        padding: 20px; margin-bottom: 16px;
    }
    .domain-badge {
        background: #2a3a1e; color: #4caf82; padding: 2px 10px;
        border-radius: 20px; font-size: 0.75rem; display: inline-block;
    }
    .year-badge {
        background: #1e2a3a; color: #7c9ef8; padding: 2px 10px;
        border-radius: 20px; font-size: 0.75rem; display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

st.title("Research Paper Helper")
st.caption("Upload a PDF → Get AI Summary, Q&A, Multi-Engine Comparison, and Improvements")

# Initialize session state
for key in ["summary", "topic", "papers", "comparison", 
            "comp_problem", "comp_method", "comp_data", "comp_results", "comp_eval",
            "improvements", "edits", "text", "images", "chunks", "qa_history"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["topic", "papers", "edits",
                                                "images", "chunks", "qa_history"] else ""

if "vision_dict" not in st.session_state:
    st.session_state.vision_dict = {}

# --- SIDEBAR: Upload ---
with st.sidebar:
    st.header("📂 Data Ingestion")
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type="pdf")
    
    if st.button("🔄 Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
        
    st.divider()

    if uploaded_file and st.button("Analyze Paper", type="primary", use_container_width=True):
        with st.spinner("Parsing PDF & running AI analysis..."):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(uploaded_file.read())
            tmp.close()

            text, images = parse_pdf(tmp.name)
            chunks = chunk_text(text)
            store_embeddings(chunks)

            init = {
                "text": text, "images": images, "chunks": chunks,
                "summary": "", "vision": [], "topic": [],
                "papers": [], "comparison": "", "improvements": "",
                "edits": [], "query": "", "answer": "", "error": None
            }
            result = upload_graph.invoke(init)
            check_limits(result)

            st.session_state.text = text
            st.session_state.images = images
            st.session_state.chunks = chunks
            st.session_state.summary = result["summary"]
            st.session_state.topic = result["topic"]
            st.session_state.pdf_path = tmp.name
            st.session_state.vision_dict = {}  # Clear previous paper's figure analysis

        st.success("Analysis complete!")

    if st.session_state.topic:
        topic_str = st.session_state.topic if isinstance(st.session_state.topic, str) else ", ".join(st.session_state.topic)
        st.info(f"**Topic:** {topic_str}")

# --- MAIN TABS ---
tab1, tab5, tab2, tab3, tab4 = st.tabs(["Summary", "Figures", "Q&A", "Compare", "Improve"])

# --- TAB 1: Summary ---
with tab1:
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        st.info("👈 Upload a PDF in the sidebar to get started.")

# --- TAB 2: Q&A ---
with tab2:
    if not st.session_state.text:
        st.info("Upload a paper first to ask questions.")
    else:
        query = st.chat_input("Ask anything about the paper...")
        for item in st.session_state.qa_history:
            with st.chat_message("user"):
                st.write(item["q"])
            with st.chat_message("assistant"):
                st.markdown(item["a"])

        if query:
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    init = {
                        "text": "", "images": [], "chunks": [],
                        "summary": "", "vision": [], "topic": [],
                        "papers": [], "comparison": "", "improvements": "",
                        "edits": [], "query": query, "answer": "", "error": None
                    }
                    result = qa_graph.invoke(init)
                    check_limits(result)
                    st.markdown(result["answer"])
                    st.session_state.qa_history.append({"q": query, "a": result["answer"]})

# --- TAB 3: Compare ---
with tab3:
    if not st.session_state.summary:
        st.info("Upload a paper first.")
    else:
        if st.button("Run Comparative Study", type="primary"):
            with st.spinner("Searching multi-engine academic sources... (30-60s)"):
                init = {
                    "text": "", "images": [], "chunks": [],
                    "summary": st.session_state.summary, "vision": [],
                    "topic": st.session_state.topic,
                    "papers": [], "comparison": "", "improvements": "",
                    "edits": [], "query": "", "answer": "", "error": None
                }
                result = compare_graph.invoke(init)
                check_limits(result)
                st.session_state.papers = result["papers"]
                st.session_state.comp_problem = result["comp_problem"]
                st.session_state.comp_method = result["comp_method"]
                st.session_state.comp_data = result["comp_data"]
                st.session_state.comp_results = result["comp_results"]
                st.session_state.comp_eval = result["comp_eval"]

        if st.session_state.papers:
            st.subheader("📚 Related Papers Found")
            for p in st.session_state.papers:
                st.markdown(f"""
<div class="paper-card">
    <span class="year-badge">{p['year']}</span>
    <span class="domain-badge">{p.get('venue', 'Academic Source')}</span>
    <br><strong style="color:#7c9ef8">{p['title']}</strong>
    <p style="color:#999;font-size:0.85rem">{p['summary'][:300]}...</p>
    {'<a href="' + p["link"] + '" target="_blank">View Source →</a>' if p.get("link") else ''}
</div>""", unsafe_allow_html=True)

        if st.session_state.comp_problem:
            st.divider()
            st.subheader("📊 Comparative Analysis")
            st.markdown("### 1. Problem & Objective")
            st.markdown(re.sub(r'^\s*(?:#+\s*)?1[^\n]*\n*', '', st.session_state.comp_problem, flags=re.IGNORECASE).strip())
            
        if st.session_state.comp_method:
            st.markdown("### 2. Methodology & Approach")
            st.markdown(re.sub(r'^\s*(?:#+\s*)?2[^\n]*\n*', '', st.session_state.comp_method, flags=re.IGNORECASE).strip())
            
        if st.session_state.comp_data:
            st.markdown("### 3. Data & Evidence")
            st.markdown(re.sub(r'^\s*(?:#+\s*)?3[^\n]*\n*', '', st.session_state.comp_data, flags=re.IGNORECASE).strip())
            
        if st.session_state.comp_results:
            st.markdown("### 4. Results & Findings")
            st.markdown(re.sub(r'^\s*(?:#+\s*)?4[^\n]*\n*', '', st.session_state.comp_results, flags=re.IGNORECASE).strip())
            
        if st.session_state.comp_eval:
            st.markdown("### 5. Evaluation Method")
            st.markdown(re.sub(r'^\s*(?:#+\s*)?5[^\n]*\n*', '', st.session_state.comp_eval, flags=re.IGNORECASE).strip())

# --- TAB 4: Improve ---
with tab4:
    if not st.session_state.comp_problem:
        st.info("Run the Comparative Study first (Compare tab).")
    else:
        if st.button("Analyze & Rewrite Sections", type="primary"):
            with st.spinner("Identifying weak sections & generating rewrites..."):
                # Combine modular results for the improvement engine
                combined_comparison = f"""
                {st.session_state.comp_problem}
                {st.session_state.comp_method}
                {st.session_state.comp_data}
                {st.session_state.comp_results}
                {st.session_state.comp_eval}
                """
                init = {
                    "text": st.session_state.text, "images": [], "chunks": [],
                    "summary": st.session_state.summary, "vision": [],
                    "topic": st.session_state.topic,
                    "papers": [], "comparison": combined_comparison,
                    "improvements": "", "edits": [],
                    "query": "", "answer": "", "error": None
                }
                result = improve_graph.invoke(init)
                check_limits(result)
                st.session_state.improvements = result["improvements"]
                st.session_state.edits = result["edits"]

        if st.session_state.improvements:
            st.markdown("---")
            st.subheader("📋 Improvement Analysis")
            st.caption("The following weaknesses were identified in the paper with specific evidence and recommended fixes:")
            if "Error" in st.session_state.improvements:
                st.error(st.session_state.improvements)
            else:
                st.markdown(st.session_state.improvements)

        if st.session_state.edits:
            st.markdown("---")
            st.subheader(f"✏️ Rewritten Sections ({len(st.session_state.edits)} improvements)")
            st.caption("Each weak section has been rewritten to match the original authors' technical style:")
            for idx, ed in enumerate(st.session_state.edits):
                title = ed.get('section', 'Unknown Section')
                with st.expander(f"#{idx+1} — {title}", expanded=False):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🔴 ORIGINAL (Weak)**")
                        orig = ed.get("original", "")
                        st.info(orig[:600] + ("..." if len(orig) > 600 else ""))
                    with col2:
                        st.markdown("**🟢 REWRITTEN (Improved)**")
                        st.success(ed.get("rewritten", ""))
        elif st.session_state.improvements and "Error" not in st.session_state.improvements:
            pass

# --- TAB 5: Figures ---
with tab5:
    if not st.session_state.images:
        st.info("Upload a paper with figures to analyze them.")
    else:
        # Analyze All Button
        if len(st.session_state.vision_dict) < len(st.session_state.images):
            if st.button("Analyze All Figures", type="primary"):
                for i, fig_data in enumerate(st.session_state.images):
                    if i not in st.session_state.vision_dict:
                        fig_dict = fig_data if isinstance(fig_data, dict) else {"path": fig_data, "caption": "", "page_num": "?", "context": "", "figure_index": i+1}
                        with st.spinner(f"Analyzing Figure {i+1}...") as sp:
                            st.session_state.vision_dict[i] = analyze_single_image(fig_dict)
                st.rerun()
                
        st.divider()
        
        # Display Images
        for i, fig_data in enumerate(st.session_state.images):
            fig_dict = fig_data if isinstance(fig_data, dict) else {"path": fig_data, "caption": "", "page_num": "?", "context": "", "figure_index": i+1}
            st.subheader(f"Figure {i+1}")
            
            # Show the context and image
            if fig_dict.get("caption"):
                st.caption(f"**Extracted Caption:** {fig_dict['caption']}")
            
            
            st.image(fig_dict["path"], width=600)
            
            if i in st.session_state.vision_dict:
                st.markdown(st.session_state.vision_dict[i])
            else:
                if st.button(f"Analyze Figure {i+1}", key=f"btn_fig_{i}"):
                    with st.spinner(f"Analyzing Figure {i+1}..."):
                        st.session_state.vision_dict[i] = analyze_single_image(fig_dict)
                    st.rerun()
            st.divider()
