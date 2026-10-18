import sys
from pathlib import Path

# Rende importabile `lib` quando pytest parte dalla cartella del progetto.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
