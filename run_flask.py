"""Python-only runner: start the Flask API."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    subprocess.check_call([sys.executable, str(ROOT / "app.py")])
