"""Accesso ai dati SkillCorner per il workbench.

Attenzione: questo modulo esiste **solo per il workbench**. Nella submission i
dati vanno caricati con `kloppy.skillcorner.load_open_data()`, perché il
regolamento vieta file di dati grossi nel repo e richiede che il notebook giri
su una macchina pulita. Vedi `plans/02-costruire-la-submission.md`.

Perché non basta kloppy anche qui
---------------------------------
Kloppy non espone il campo `is_detected`, che distingue le posizioni *osservate*
da quelle *estrapolate*: `frame.players_data[p].other_data` è vuoto e `to_df()`
restituisce solo x, y, distanza e velocità. Siccome l'estrapolazione è il limite
principale di questi dati, per quantificarla bisogna leggere il JSONL grezzo.

Due sorgenti, stessa interfaccia
--------------------------------
I file di tracking sono su Git LFS (~90 MB l'uno). Se il clone locale non è stato
popolato con `git lfs pull`, sul disco ci sono solo puntatori da 133 byte. Le
funzioni qui sotto se ne accorgono e ripiegano sullo streaming da GitHub, così i
notebook girano comunque.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd

# Clone locale di https://github.com/SkillCorner/opendata
OPENDATA = Path.home() / "Documents/Projects/sport_data/skillcorner-opendata"
DATA = OPENDATA / "data"
MATCHES = DATA / "matches"

# I file LFS non scaricati sono puntatori di poche centinaia di byte.
_LFS_POINTER_MAX_BYTES = 1024

_REMOTE = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data"


def list_matches() -> list[int]:
    """Gli id delle partite disponibili nel clone locale."""
    return sorted(int(p.name) for p in MATCHES.iterdir() if p.name.isdigit())


def match_dir(match_id: int | str) -> Path:
    return MATCHES / str(match_id)


def is_lfs_pointer(path: Path) -> bool:
    """True se il file è un puntatore LFS invece del contenuto vero."""
    return path.stat().st_size <= _LFS_POINTER_MAX_BYTES


def tracking_available_locally(match_id: int | str) -> bool:
    path = match_dir(match_id) / f"{match_id}_tracking_extrapolated.jsonl"
    return path.exists() and not is_lfs_pointer(path)


def iter_tracking(match_id: int | str, limit: int | None = None) -> Iterator[dict]:
    """Frame di tracking grezzi, uno per riga del JSONL.

    Legge dal clone locale se il file è stato scaricato via LFS, altrimenti
    streamma da GitHub senza tenere in memoria l'intero file.

    Ogni frame ha: frame, timestamp, period, ball_data, possession,
    image_corners_projection, player_data. Ogni elemento di `player_data` ha
    x, y, player_id, **is_detected**.
    """
    path = match_dir(match_id) / f"{match_id}_tracking_extrapolated.jsonl"

    if path.exists() and not is_lfs_pointer(path):
        with path.open() as fh:
            for i, line in enumerate(fh):
                if limit is not None and i >= limit:
                    return
                if line.strip():
                    yield json.loads(line)
        return

    # Fallback: streaming da GitHub (il file locale è un puntatore LFS).
    import urllib.request

    url = f"{_REMOTE}/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"
    with urllib.request.urlopen(url) as resp:
        for i, raw in enumerate(resp):
            if limit is not None and i >= limit:
                return
            line = raw.decode("utf-8").strip()
            if line:
                yield json.loads(line)


def tracking_long(match_id: int | str, limit: int | None = None) -> pd.DataFrame:
    """Tracking in formato lungo: una riga per (frame, giocatore).

    Colonne: frame, timestamp, period, player_id, x, y, is_detected.
    A 10 fps una partita intera sono ~1,2 milioni di righe: usa `limit`
    durante lo sviluppo.
    """
    rows = []
    for fr in iter_tracking(match_id, limit=limit):
        for p in fr["player_data"]:
            rows.append(
                (
                    fr["frame"],
                    fr["timestamp"],
                    fr["period"],
                    p["player_id"],
                    p["x"],
                    p["y"],
                    p["is_detected"],
                )
            )
    return pd.DataFrame(
        rows,
        columns=["frame", "timestamp", "period", "player_id", "x", "y", "is_detected"],
    )


def load_match_meta(match_id: int | str) -> dict:
    """Metadati della partita: formazioni, minuti, arbitro, dimensioni campo."""
    with (match_dir(match_id) / f"{match_id}_match.json").open() as fh:
        return json.load(fh)


def load_dynamic_events(match_id: int | str) -> pd.DataFrame:
    return pd.read_csv(match_dir(match_id) / f"{match_id}_dynamic_events.csv",
                       low_memory=False)


def load_phases_of_play(match_id: int | str) -> pd.DataFrame:
    return pd.read_csv(match_dir(match_id) / f"{match_id}_phases_of_play.csv")


def load_aggregate(kind: str) -> pd.DataFrame:
    """Aggregati stagionali. kind: 'physical' | 'obr' | 'passing'."""
    names = {
        "physical": "aus1league_physicalaggregates_20242025.csv",
        "obr": "aus1league_obraggregates_20242025.csv",
        "passing": "aus1league_passingaggregates_20242025.csv",
    }
    return pd.read_csv(DATA / "aggregates" / names[kind])
