"""Python-only runner: start the Streamlit app."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    subprocess.check_call(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "streamlit_app.py")]
    )
