"""Percorsi condivisi dello spazio di lavoro tennis.

Unico posto in cui è scritto dove stanno i dati: le analisi importano da qui
invece di costruire path relativi (che si rompono a seconda della cartella da
cui si lancia il notebook).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ANALYSES_DIR = PROJECT_ROOT / "analyses"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def analysis_dir(name: str) -> Path:
    """Cartella di una singola analisi, es. analysis_dir("serve-dominance")."""
    return ANALYSES_DIR / name


def processed_path(name: str) -> Path:
    """Path di un dataset derivato in data/processed/ (creando la cartella)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DIR / name
