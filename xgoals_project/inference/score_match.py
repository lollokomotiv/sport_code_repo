#!/usr/bin/env python3
"""Score matches with an xG model and compare to realized goals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import joblib

GOAL_X = 120
GOAL_Y = 40
GOAL_WIDTH = 7.32
LEFT_POST = (GOAL_X, GOAL_Y - GOAL_WIDTH / 2)
RIGHT_POST = (GOAL_X, GOAL_Y + GOAL_WIDTH / 2)

OPEN_PLAY_PATTERNS = {"Regular Play", "From Counter", "From Keeper", "Other"}
OPEN_PLAY_EXCLUDED_SHOT_TYPES = {"Penalty", "Free Kick"}
ALLOWED_SHOT_SCOPES = {"open_play", "all_non_penalty"}

REQUIRED_COLUMNS = [
    "distance",
    "angle",
    "body_part",
    "shot_type",
    "shot_technique",
    "play_pattern",
    "first_time",
    "one_on_one",
    "under_pressure",
    "nearest_defender_dist",
    "keeper_dist_to_shot",
    "keeper_dist_to_goal",
    "keeper_present",
    "n_defenders_within_1m",
    "n_defenders_within_2m",
    "n_defenders_within_3m",
    "n_players_in_cone_to_goal",
    "visible_area_size",
]

# Percorsi di default (coerenti con il notebook).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODELS_DIR / "xg_model_360.joblib"
DEFAULT_HOLDOUT_PATH = DEFAULT_MODELS_DIR / "holdout_match_ids.json"
# Data root di default usato nel notebook. Modifica se sposti i dati.
DEFAULT_DATA_ROOT = Path("/Users/lorenzoguercio/Documents/Projects/sport_data/open-data/data")


def distance_to_goal(x: float, y: float) -> float:
    return math.hypot(GOAL_X - x, GOAL_Y - y)


def shot_angle(x: float, y: float) -> float:
    half_width = GOAL_WIDTH / 2
    left_post = (GOAL_X, GOAL_Y - half_width)
    right_post = (GOAL_X, GOAL_Y + half_width)
    a = math.hypot(left_post[0] - x, left_post[1] - y)
    b = math.hypot(right_post[0] - x, right_post[1] - y)
    c = GOAL_WIDTH
    cos_angle = (a * a + b * b - c * c) / (2 * a * b) if a > 0 and b > 0 else 1.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.acos(cos_angle)


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def polygon_area(coords: List[Tuple[float, float]]) -> Optional[float]:
    if not coords or len(coords) < 3:
        return None
    area = 0.0
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def point_in_triangle(pt, a, b, c) -> bool:
    px, py = pt
    ax, ay = a
    bx, by = b
    cx, cy = c

    v0x, v0y = cx - ax, cy - ay
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = px - ax, py - ay

    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y

    denom = dot00 * dot11 - dot01 * dot01
    if denom == 0:
        return False
    inv = 1 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return (u >= 0) and (v >= 0) and (u + v <= 1)


def is_scoring_shot(event: dict, shot_scope: str) -> bool:
    if event.get("type", {}).get("name") != "Shot":
        return False
    shot_type = event.get("shot", {}).get("type", {}).get("name")
    play_pattern = event.get("play_pattern", {}).get("name")

    # Penalty escluso in tutti gli scope.
    if shot_type == "Penalty":
        return False

    if shot_scope == "open_play":
        if play_pattern not in OPEN_PLAY_PATTERNS:
            return False
        if shot_type in OPEN_PLAY_EXCLUDED_SHOT_TYPES:
            return False
    elif shot_scope != "all_non_penalty":
        raise ValueError(f"shot_scope non valido: {shot_scope}. Usa uno tra {sorted(ALLOWED_SHOT_SCOPES)}")

    return True


def is_open_play_shot(event: dict) -> bool:
    """Compatibilita' retro: mantiene il vecchio comportamento open-play."""
    if not is_scoring_shot(event, shot_scope="open_play"):
        return False
    return True


def load_events(events_dir: Path, match_id: str) -> List[dict]:
    path = events_dir / f"{match_id}.json"
    return json.loads(path.read_text())


def load_three_sixty(three_sixty_dir: Path, match_id: str) -> Dict[str, dict]:
    path = three_sixty_dir / f"{match_id}.json"
    frames = json.loads(path.read_text())
    return {f.get("event_uuid"): f for f in frames if f.get("event_uuid")}


def load_holdout_match_ids(path: Path) -> List[str]:
    """Carica la lista dei match holdout (creata dal notebook)."""
    if not path.exists():
        raise FileNotFoundError(f"Holdout file non trovato: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Il file holdout_match_ids.json deve contenere una lista di match_id.")
    return [str(mid) for mid in data]


def build_shot_breakdown(
    events: List[dict], frames_by_event: Dict[str, dict], match_id: str, shot_scope: str
) -> pd.DataFrame:
    """Conteggi debug per capire cosa entra (o no) nello scope di scoring."""
    teams = sorted(
        {
            e.get("team", {}).get("name")
            for e in events
            if e.get("team", {}).get("name")
        }
    )
    counts = {
        team: {
            "match_id": match_id,
            "team": team,
            "shots_total": 0,
            "shots_in_scope": 0,
            "shots_in_scope_360": 0,
            "goals_in_scope": 0,
        }
        for team in teams
    }

    for e in events:
        team = e.get("team", {}).get("name")
        if not team:
            continue
        if team not in counts:
            counts[team] = {
                "match_id": match_id,
                "team": team,
                "shots_total": 0,
                "shots_in_scope": 0,
                "shots_in_scope_360": 0,
                "goals_in_scope": 0,
            }

        if e.get("type", {}).get("name") != "Shot":
            continue

        counts[team]["shots_total"] += 1
        if is_scoring_shot(e, shot_scope=shot_scope):
            counts[team]["shots_in_scope"] += 1
            if e.get("shot", {}).get("outcome", {}).get("name") == "Goal":
                counts[team]["goals_in_scope"] += 1
            if e.get("id") in frames_by_event:
                counts[team]["shots_in_scope_360"] += 1

    rows = list(counts.values())
    if not rows:
        return pd.DataFrame(
            columns=["match_id", "team", "shots_total", "shots_in_scope", "shots_in_scope_360", "goals_in_scope"]
        )
    return pd.DataFrame(rows).sort_values(["match_id", "team"])


def debug_shot_breakdown(data_root: Path, match_id: str, shot_scope: str) -> pd.DataFrame:
    """Breakdown debug per un singolo match."""
    events_dir = data_root / "events"
    three_sixty_dir = data_root / "three-sixty"
    events = load_events(events_dir, match_id)
    frames_by_event = load_three_sixty(three_sixty_dir, match_id)
    return build_shot_breakdown(events, frames_by_event, match_id, shot_scope=shot_scope)


def extract_360_features(frame: dict, shot_x: float, shot_y: float) -> dict:
    ff = frame.get("freeze_frame") or []
    shot_pt = (shot_x, shot_y)

    opponents = [p for p in ff if p.get("teammate") is False]
    opp_locs = [p.get("location") for p in opponents if isinstance(p.get("location"), list)]
    opp_locs = [(p[0], p[1]) for p in opp_locs if len(p) >= 2]

    keepers = [p for p in ff if p.get("keeper") is True]
    keeper_loc = None
    if keepers:
        loc = keepers[0].get("location")
        if isinstance(loc, list) and len(loc) >= 2:
            keeper_loc = (loc[0], loc[1])

    nearest_def_dist = min([dist(shot_pt, p) for p in opp_locs], default=None)

    def within(radius: float) -> int:
        return sum(1 for p in opp_locs if dist(shot_pt, p) <= radius)

    visible_area = frame.get("visible_area")
    va_size = None
    if isinstance(visible_area, list):
        va_coords = [(visible_area[i], visible_area[i + 1]) for i in range(0, len(visible_area), 2)]
        va_size = polygon_area(va_coords)

    return {
        "nearest_defender_dist": nearest_def_dist,
        "keeper_dist_to_shot": dist(shot_pt, keeper_loc) if keeper_loc else None,
        "keeper_dist_to_goal": dist((GOAL_X, GOAL_Y), keeper_loc) if keeper_loc else None,
        "keeper_present": 1 if keeper_loc else 0,
        "n_defenders_within_1m": within(1),
        "n_defenders_within_2m": within(2),
        "n_defenders_within_3m": within(3),
        "n_players_in_cone_to_goal": sum(
            1 for p in opp_locs if point_in_triangle(p, shot_pt, LEFT_POST, RIGHT_POST)
        ),
        "visible_area_size": va_size,
    }


def build_shot_rows(
    events: List[dict],
    frames_by_event: Dict[str, dict],
    match_id: str,
    require_360: bool,
    shot_scope: str,
) -> List[dict]:
    rows = []
    for e in events:
        if not is_scoring_shot(e, shot_scope=shot_scope):
            continue

        event_id = e.get("id")
        frame = frames_by_event.get(event_id)
        if require_360 and not frame:
            continue

        loc = e.get("location") or []
        if len(loc) < 2:
            continue
        x, y = loc[0], loc[1]

        row = {
            "match_id": match_id,
            "event_id": event_id,
            "team": e.get("team", {}).get("name"),
            "player": e.get("player", {}).get("name"),
            "shot_x": x,
            "shot_y": y,
            "distance": distance_to_goal(x, y),
            "angle": shot_angle(x, y),
            "body_part": e.get("shot", {}).get("body_part", {}).get("name"),
            "first_time": e.get("shot", {}).get("first_time"),
            "one_on_one": e.get("shot", {}).get("one_on_one"),
            "under_pressure": e.get("under_pressure"),
            "shot_type": e.get("shot", {}).get("type", {}).get("name"),
            "shot_technique": e.get("shot", {}).get("technique", {}).get("name"),
            "play_pattern": e.get("play_pattern", {}).get("name"),
            "goal": 1 if e.get("shot", {}).get("outcome", {}).get("name") == "Goal" else 0,
        }

        if frame:
            row.update(extract_360_features(frame, x, y))
        else:
            # Keep columns aligned for baseline models.
            row.update({col: None for col in REQUIRED_COLUMNS if col not in row})

        rows.append(row)

    return rows


def score_match(
    model_path: Path,
    data_root: Path,
    match_id: str,
    require_360: bool,
    shot_scope: str = "open_play",
) -> pd.DataFrame:
    model = joblib.load(model_path)

    events_dir = data_root / "events"
    three_sixty_dir = data_root / "three-sixty"

    events = load_events(events_dir, match_id)
    frames_by_event = load_three_sixty(three_sixty_dir, match_id)

    rows = build_shot_rows(events, frames_by_event, match_id, require_360, shot_scope=shot_scope)
    shots = pd.DataFrame(rows)

    for col in REQUIRED_COLUMNS:
        if col not in shots.columns:
            shots[col] = None

    proba = model.predict_proba(shots)[:, 1]
    shots["xg"] = proba
    return shots


def score_matches(
    model_path: Path,
    data_root: Path,
    match_ids: List[str],
    require_360: bool,
    shot_scope: str = "open_play",
) -> pd.DataFrame:
    """Score multipli match con lo stesso modello."""
    all_shots = []
    for mid in match_ids:
        shots = score_match(
            model_path=model_path,
            data_root=data_root,
            match_id=mid,
            require_360=require_360,
            shot_scope=shot_scope,
        )
        all_shots.append(shots)

    if not all_shots:
        return pd.DataFrame()

    return pd.concat(all_shots, ignore_index=True)


def summarize(shots: pd.DataFrame) -> pd.DataFrame:
    return (
        shots.groupby("team", dropna=False)
        .agg(
            shots=("event_id", "count"),
            xg=("xg", "sum"),
            goals=("goal", "sum"),
        )
        .reset_index()
        .sort_values("xg", ascending=False)
    )


def summarize_by_match(shots: pd.DataFrame) -> pd.DataFrame:
    """Summary per match e per team (utile per holdout)."""
    return (
        shots.groupby(["match_id", "team"], dropna=False)
        .agg(
            shots=("event_id", "count"),
            xg=("xg", "sum"),
            goals=("goal", "sum"),
        )
        .reset_index()
        .sort_values(["match_id", "xg"], ascending=[True, False])
    )


def score_holdout_matches(
    model_path: Path = DEFAULT_MODEL_PATH,
    data_root: Path = DEFAULT_DATA_ROOT,
    holdout_path: Path = DEFAULT_HOLDOUT_PATH,
    require_360: bool = True,
    shot_scope: str = "open_play",
) -> pd.DataFrame:
    """Score dei match holdout (salvati dal notebook) senza usare la CLI."""
    holdout_match_ids = load_holdout_match_ids(holdout_path)
    if not holdout_match_ids:
        raise ValueError("Lista holdout vuota: nulla da score.")
    return score_matches(
        model_path=model_path,
        data_root=data_root,
        match_ids=holdout_match_ids,
        require_360=require_360,
        shot_scope=shot_scope,
    )


def run_holdout_scoring(
    model_path: Path = DEFAULT_MODEL_PATH,
    data_root: Path = DEFAULT_DATA_ROOT,
    holdout_path: Path = DEFAULT_HOLDOUT_PATH,
    require_360: bool = True,
    debug_breakdown: bool = False,
    shot_scope: str = "open_play",
) -> None:
    """Wrapper comodo: score holdout + stampa summary (equivalente al comando CLI)."""
    holdout_match_ids = load_holdout_match_ids(holdout_path)
    shots = score_matches(
        model_path=model_path,
        data_root=data_root,
        match_ids=holdout_match_ids,
        require_360=require_360,
        shot_scope=shot_scope,
    )

    if debug_breakdown:
        debug_rows = [debug_shot_breakdown(data_root, mid, shot_scope=shot_scope) for mid in holdout_match_ids]
        debug_df = pd.concat(debug_rows, ignore_index=True) if debug_rows else pd.DataFrame()
        print(f"\nDebug shot breakdown per team (scope={shot_scope}):")
        if debug_df.empty:
            print("Nessun dato disponibile.")
        else:
            print(debug_df.to_string(index=False))

    summary_by_match = summarize_by_match(shots)
    print(f"\nHoldout summary (scope={shot_scope}):")
    print(summary_by_match.to_string(index=False))

    total_xg = shots["xg"].sum()
    total_goals = shots["goal"].sum()
    print(f"\nTotale holdout xG ({shot_scope}): {total_xg:.3f}")
    print(f"Totale holdout goals ({shot_scope}): {total_goals}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        help="Path completo al file joblib del modello",
    )
    parser.add_argument(
        "--model-name",
        help="Nome modello nella models-dir (es: xg_model_360_no_penalty o xg_model360_no_penalty.joblib)",
    )
    parser.add_argument("--data-root", required=True, help="StatsBomb data root (contains events/ and three-sixty/)")
    parser.add_argument("--match-id", help="Match id to score (ignored if --score-holdout)")
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="Cartella modelli (default: <project>/models)",
    )
    parser.add_argument(
        "--holdout-path",
        default=str(DEFAULT_HOLDOUT_PATH),
        help="File JSON con match holdout (default: models/holdout_match_ids.json)",
    )
    parser.add_argument(
        "--score-holdout",
        action="store_true",
        help="Score dei match holdout salvati dal notebook",
    )
    parser.add_argument(
        "--require-360",
        action="store_true",
        help="Skip shots without 360 frames (recommended for 360 model)",
    )
    parser.add_argument(
        "--debug-shot-breakdown",
        action="store_true",
        help="Stampa per team: shots_total, shots_in_scope, shots_in_scope_360, goals_in_scope",
    )
    parser.add_argument(
        "--shot-scope",
        default="open_play",
        choices=sorted(ALLOWED_SHOT_SCOPES),
        help="Scope dei tiri da includere nello scoring",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    data_root = Path(args.data_root)

    if args.model_path and args.model_name:
        parser.error("Usa solo uno tra --model-path e --model-name.")

    if args.model_name:
        model_filename = args.model_name if args.model_name.endswith(".joblib") else f"{args.model_name}.joblib"
        model_path = models_dir / model_filename
    elif args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = models_dir / DEFAULT_MODEL_PATH.name

    holdout_path = Path(args.holdout_path)
    if args.holdout_path == str(DEFAULT_HOLDOUT_PATH) and args.models_dir:
        holdout_path = models_dir / DEFAULT_HOLDOUT_PATH.name

    if args.score_holdout:
        # Carico i match holdout dal file creato nel notebook.
        holdout_match_ids = load_holdout_match_ids(holdout_path)
        if not holdout_match_ids:
            raise ValueError("Lista holdout vuota: nulla da score.")

        shots = score_matches(
            model_path=model_path,
            data_root=data_root,
            match_ids=holdout_match_ids,
            require_360=args.require_360,
            shot_scope=args.shot_scope,
        )

        if args.debug_shot_breakdown:
            debug_rows = [debug_shot_breakdown(data_root, mid, shot_scope=args.shot_scope) for mid in holdout_match_ids]
            debug_df = pd.concat(debug_rows, ignore_index=True) if debug_rows else pd.DataFrame()
            print(f"\nDebug shot breakdown per team (scope={args.shot_scope}):")
            if debug_df.empty:
                print("Nessun dato disponibile.")
            else:
                print(debug_df.to_string(index=False))

        summary_by_match = summarize_by_match(shots)
        print(f"\nHoldout summary (scope={args.shot_scope}):")
        print(summary_by_match.to_string(index=False))

        total_xg = shots["xg"].sum()
        total_goals = shots["goal"].sum()
        print(f"\nTotale holdout xG ({args.shot_scope}): {total_xg:.3f}")
        print(f"Totale holdout goals ({args.shot_scope}): {total_goals}")
        return

    if not args.match_id:
        parser.error("Devi fornire --match-id oppure usare --score-holdout.")

    shots = score_match(
        model_path=model_path,
        data_root=data_root,
        match_id=args.match_id,
        require_360=args.require_360,
        shot_scope=args.shot_scope,
    )

    if args.debug_shot_breakdown:
        debug_df = debug_shot_breakdown(data_root, args.match_id, shot_scope=args.shot_scope)
        print(f"\nDebug shot breakdown per team (scope={args.shot_scope}):")
        if debug_df.empty:
            print("Nessun dato disponibile.")
        else:
            print(debug_df.to_string(index=False))

    summary = summarize(shots)
    print(f"\nMatch summary (scope={args.shot_scope}):")
    print(summary.to_string(index=False))

    total_xg = shots["xg"].sum()
    total_goals = shots["goal"].sum()
    print(f"\nTotal xG ({args.shot_scope}): {total_xg:.3f}")
    print(f"Total goals ({args.shot_scope}): {total_goals}")


if __name__ == "__main__":
    main()
