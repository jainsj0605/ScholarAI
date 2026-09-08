"""ScholarAI Entrypoint Runner."""
from src.config import FLASK_DEBUG, PORT
from src.app import app

if __name__ == "__main__":
    print(f"Starting ScholarAI on http://127.0.0.1:{PORT} (Debug={FLASK_DEBUG})")
    app.run(host="0.0.0.0", port=PORT, debug=FLASK_DEBUG)
