"""Python-only setup: install deps and train the sample model."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
    run([sys.executable, str(ROOT / "train_model.py")])


if __name__ == "__main__":
    main()
