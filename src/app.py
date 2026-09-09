import os
try:
    import pymupdf as fitz
except ImportError:
    import fitz
from flask import Flask, request, jsonify, send_file, render_template

from src.config import UPLOAD_PDF_PATH
from src.rag import parse_pdf, chunk_text, store_embeddings, vector_store
from src.workflows import (
    PaperState,
    upload_graph,
    compare_graph,
    improve_graph,
    qa_graph
)
from src.pdf_builder import build_analysis_pdf, build_edited_original_pdf

# Initialize Flask application
app = Flask(__name__, template_folder="templates")

# Warm up embedding model at startup so first request doesn't block on model init
try:
    _ = vector_store.embed_model
except Exception as _init_err:
    print(f"[Warning] Model pre-warm notice: {_init_err}")

# =========================
# WEB PAGE ROUTES
# =========================
@app.route("/")
def home():
    return render_template("upload.html", page="upload")

@app.route("/qa")
def page_qa():
    return render_template("qa.html", page="qa")

@app.route("/compare")
def page_compare():
    return render_template("compare.html", page="compare")

@app.route("/improve")
def page_improve():
    return render_template("improve.html", page="improve")

@app.route("/download")
def page_download():
    return render_template("download.html", page="download")

# =========================
# API ENDPOINTS
# =========================
@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part provided"}), 400
        file = request.files["file"]
        if not file or not file.filename:
            return jsonify({"error": "No file selected"}), 400

        file.save(UPLOAD_PDF_PATH)
        text, images = parse_pdf(UPLOAD_PDF_PATH)
        chunks = chunk_text(text)
        store_embeddings(chunks)

        init_state: PaperState = {
            "text": text,
            "images": images,
            "chunks": chunks,
            "summary": "",
            "vision": [],
            "topic": "",
            "papers": [],
            "comparison": "",
            "improvements": "",
            "edits": [],
            "query": "",
            "answer": "",
            "error": None
        }
        result = upload_graph.invoke(init_state)
        return jsonify({
            "summary": result["summary"],
            "vision": result["vision"],
            "topic": result["topic"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json() or {}
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "No query provided"}), 400

        init_state: PaperState = {
            "text": "",
            "images": [],
            "chunks": [],
            "summary": "",
            "vision": [],
            "topic": "",
            "papers": [],
            "comparison": "",
            "improvements": "",
            "edits": [],
            "query": query,
            "answer": "",
            "error": None
        }
        result = qa_graph.invoke(init_state)
        return jsonify({"answer": result["answer"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compare", methods=["POST"])
def compare():
    try:
        data = request.get_json() or {}
        summary = data.get("summary", "")
        topic = data.get("topic", "")

        init_state: PaperState = {
            "text": "",
            "images": [],
            "chunks": [],
            "summary": summary,
            "vision": [],
            "topic": topic,
            "papers": [],
            "comparison": "",
            "improvements": "",
            "edits": [],
            "query": "",
            "answer": "",
            "error": None
        }
        result = compare_graph.invoke(init_state)
        return jsonify({
            "papers": result["papers"],
            "comparison": result["comparison"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/improve", methods=["POST"])
def improve():
    try:
        data = request.get_json() or {}
        summary = data.get("summary", "")
        comparison = data.get("comparison", "")

        full_text = ""
        if os.path.exists(UPLOAD_PDF_PATH):
            doc = fitz.open(UPLOAD_PDF_PATH)
            for page in doc:
                full_text += page.get_text()
            doc.close()

        init_state: PaperState = {
            "text": full_text,
            "images": [],
            "chunks": [],
            "summary": summary,
            "vision": [],
            "topic": "",
            "papers": [],
            "comparison": comparison,
            "improvements": "",
            "edits": [],
            "query": "",
            "answer": "",
            "error": None
        }
        result = improve_graph.invoke(init_state)
        return jsonify({
            "improvements": result["improvements"],
            "edits": result["edits"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate-pdf", methods=["POST"])
def generate_pdf():
    """Build and download the complete analysis PDF report with appendix."""
    try:
        data = request.get_json() or {}
        summary = data.get("summary", "")
        vision = data.get("vision", [])
        comparison = data.get("comparison", "")
        improvements = data.get("improvements", "")
        edits = data.get("edits", [])
        papers = data.get("papers", [])
        topic = data.get("topic", "Research Paper")
        qa_text = data.get("qa_text", "")

        if not os.path.exists(UPLOAD_PDF_PATH):
            return jsonify({"error": "Original PDF not found on server. Please upload a paper first."}), 400

        buf = build_analysis_pdf(
            UPLOAD_PDF_PATH, summary, vision, comparison,
            improvements, edits, papers, topic, qa_text
        )
        return send_file(
            buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="paper_full_analysis.pdf"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download-original", methods=["POST"])
def download_original():
    """Build and download the original PDF with only in-place rewrites applied."""
    try:
        data = request.get_json() or {}
        edits = data.get("edits", [])

        if not os.path.exists(UPLOAD_PDF_PATH):
            return jsonify({"error": "Original PDF not found on server. Please upload a paper first."}), 400

        buf = build_edited_original_pdf(UPLOAD_PDF_PATH, edits)
        return send_file(
            buf,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="paper_edited.pdf"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    from src.config import FLASK_DEBUG, PORT
    app.run(debug=FLASK_DEBUG, port=PORT)
