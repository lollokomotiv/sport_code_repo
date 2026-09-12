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


def analysis_dir(matchup: str, name: str | None = None) -> Path:
    """Cartella di un matchup o di una sua analisi.

    analysis_dir("alcaraz-zverev") -> analyses/alcaraz-zverev/
    analysis_dir("alcaraz-zverev", "smorzate") -> analyses/alcaraz-zverev/smorzate/
    """
    return ANALYSES_DIR / matchup if name is None else ANALYSES_DIR / matchup / name


def processed_path(name: str) -> Path:
    """Path di un dataset derivato in data/processed/ (creando la cartella)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DIR / name
