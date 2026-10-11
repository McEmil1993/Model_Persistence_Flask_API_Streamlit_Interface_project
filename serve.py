"""Production server (use this for demos and deployment instead of `python app.py`)."""
from waitress import serve

from app import app

if __name__ == "__main__":
    print("Serving on http://127.0.0.1:8000  (Ctrl+C to stop)")
    serve(app, host="127.0.0.1", port=8000)
