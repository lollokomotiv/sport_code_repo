"""Costruisce il dataset per il modello Post-Shot xG.

Popolazione: tiri NON da rigore che sono finiti IN PORTA (Goal, Saved, Saved to Post),
su tutti i match StatsBomb disponibili — i dati 360 non servono.

Due accortezze metodologiche, entrambe importanti:

1. `end_x` NON e' una feature. StatsBomb registra dove il pallone e' finito: per un gol
   e' la linea di porta (x=120), per una parata e' il punto in cui il portiere l'ha preso
   (mediana 117.8). Usarla darebbe al modello un leak quasi perfetto.

2. Per lo stesso motivo `end_y` e `end_z` grezzi non sono confrontabili fra gol e parate.
   Vengono proiettati lungo la traiettoria fino a x=120, cioe' stimando *dove sarebbe
   andato* il pallone — che e' esattamente la domanda a cui il PSxG risponde.
   Assunzione: traiettoria rettilinea in 3D dal piede del tiratore (z=0) al punto finale.
   E' un'approssimazione: il volo reale e' parabolico, ma la correzione media e' di ~2 metri.

Uso:
    python3 psxg/build_dataset.py                  # tutti i match
    python3 psxg/build_dataset.py --limit 100      # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "inference"))

# Riuso delle funzioni geometriche gia' usate per addestrare il modello xG:
# una sola implementazione, nessuna quarta copia da tenere allineata.
from score_match import distance_to_goal, shot_angle  # noqa: E402

DATA_ROOT = Path("/Users/lorenzoguercio/Documents/Projects/sport_data/open-data/data")
OUT_PATH = PROJECT_ROOT / "datasets" / "psxg_shots.csv"

GOAL_X, GOAL_Y = 120.0, 40.0
POST_LEFT, POST_RIGHT = 36.0, 44.0
CROSSBAR = 2.67
ON_TARGET = {"Goal", "Saved", "Saved to Post"}


def project_to_goal_line(loc: list, end: list) -> tuple[float, float, bool]:
    """Proietta la traiettoria del tiro fino alla linea di porta (x=120).

    Restituisce (y, z, proiettato). `proiettato` e' False quando il pallone
    aveva gia' raggiunto la linea: in quel caso le coordinate sono quelle reali.
    """
    x0, y0 = loc[0], loc[1]
    x1, y1, z1 = end[0], end[1], end[2]
    if x1 >= GOAL_X - 1e-6 or x1 <= x0:
        return y1, z1, False
    t = (GOAL_X - x0) / (x1 - x0)
    return y0 + t * (y1 - y0), z1 * t, True


def build_rows(events: list, match_id: str) -> list[dict]:
    rows = []
    for e in events:
        if e.get("type", {}).get("name") != "Shot":
            continue
        s = e.get("shot", {})
        if s.get("type", {}).get("name") == "Penalty":
            continue
        if s.get("outcome", {}).get("name") not in ON_TARGET:
            continue

        loc = e.get("location")
        end = s.get("end_location")
        if not (isinstance(loc, list) and len(loc) >= 2):
            continue
        if not (isinstance(end, list) and len(end) == 3):
            continue

        x, y = loc[0], loc[1]
        end_y, end_z, was_projected = project_to_goal_line(loc, end)

        rows.append({
            # identificativi
            "match_id": match_id,
            "event_id": e.get("id"),
            "team": e.get("team", {}).get("name"),
            "player": e.get("player", {}).get("name"),
            "minute": e.get("minute"),

            # target
            "goal": int(s.get("outcome", {}).get("name") == "Goal"),

            # piazzamento (post-tiro) — il cuore del modello
            "end_y": end_y,
            "end_z": end_z,
            "dist_from_center": abs(end_y - GOAL_Y),
            "dist_from_post": min(abs(end_y - POST_LEFT), abs(end_y - POST_RIGHT)),
            "dist_from_crossbar": CROSSBAR - end_z,
            "inside_frame": int(POST_LEFT <= end_y <= POST_RIGHT and 0 <= end_z <= CROSSBAR),
            "was_projected": int(was_projected),

            # contesto (pre-tiro) — stessa convenzione del modello xG
            "shot_x": x,
            "shot_y": y,
            "distance": distance_to_goal(x, y),
            "angle": shot_angle(x, y),
            "body_part": s.get("body_part", {}).get("name"),
            "shot_type": s.get("type", {}).get("name"),
            "shot_technique": s.get("technique", {}).get("name"),
            "play_pattern": e.get("play_pattern", {}).get("name"),
            # Flag: convertiti in 0/1 alla fonte. StatsBomb usa True/assente, ma
            # lasciare dei None qui significherebbe far decidere all'imputer cosa
            # vuol dire "assente" — ed e' esattamente il tipo di ambiguita' che ha
            # gia' prodotto un disallineamento fra training e inferenza nel modello xA.
            "first_time": int(bool(s.get("first_time"))),
            "one_on_one": int(bool(s.get("one_on_one"))),
            "under_pressure": int(bool(e.get("under_pressure"))),
            "aerial_won": int(bool(s.get("aerial_won"))),
            "deflected": int(bool(s.get("deflected"))),
            "open_goal": int(bool(s.get("open_goal"))),

            # benchmark per il confronto
            "statsbomb_xg": s.get("statsbomb_xg"),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="Usa solo i primi N match (smoke test)")
    ap.add_argument("--out", default=str(OUT_PATH), help="File CSV di destinazione")
    args = ap.parse_args()

    match_ids = sorted(p.stem for p in (DATA_ROOT / "events").glob("*.json"))
    if args.limit:
        match_ids = match_ids[:args.limit]

    rows, skipped = [], []
    for i, mid in enumerate(match_ids, 1):
        try:
            events = json.loads((DATA_ROOT / "events" / f"{mid}.json").read_text())
        except Exception as exc:
            skipped.append((mid, str(exc)[:60]))
            continue
        rows.extend(build_rows(events, mid))
        if i % 500 == 0:
            print(f"  {i}/{len(match_ids)} match — {len(rows):,} tiri", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nMatch processati: {len(match_ids) - len(skipped):,}/{len(match_ids):,}")
    if skipped:
        print(f"File illeggibili: {len(skipped)}")
        for mid, err in skipped[:5]:
            print(f"  - {mid}: {err}")
    print(f"Tiri in porta:    {len(df):,}")
    print(f"Gol:              {df['goal'].sum():,} ({df['goal'].mean()*100:.1f}%)")
    print(f"Salvato in:       {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
