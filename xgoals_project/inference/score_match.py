#!/usr/bin/env python3
"""Score a single match with an xG model and compare to open-play goals."""

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
EXCLUDE_SHOT_TYPES = {"Penalty", "Free Kick"}

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


def is_open_play_shot(event: dict) -> bool:
    if event.get("type", {}).get("name") != "Shot":
        return False
    if event.get("play_pattern", {}).get("name") not in OPEN_PLAY_PATTERNS:
        return False
    if event.get("shot", {}).get("type", {}).get("name") in EXCLUDE_SHOT_TYPES:
        return False
    return True


def load_events(events_dir: Path, match_id: str) -> List[dict]:
    path = events_dir / f"{match_id}.json"
    return json.loads(path.read_text())


def load_three_sixty(three_sixty_dir: Path, match_id: str) -> Dict[str, dict]:
    path = three_sixty_dir / f"{match_id}.json"
    frames = json.loads(path.read_text())
    return {f.get("event_uuid"): f for f in frames if f.get("event_uuid")}


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
) -> List[dict]:
    rows = []
    for e in events:
        if not is_open_play_shot(e):
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
) -> pd.DataFrame:
    model = joblib.load(model_path)

    events_dir = data_root / "events"
    three_sixty_dir = data_root / "three-sixty"

    events = load_events(events_dir, match_id)
    frames_by_event = load_three_sixty(three_sixty_dir, match_id)

    rows = build_shot_rows(events, frames_by_event, match_id, require_360)
    shots = pd.DataFrame(rows)

    for col in REQUIRED_COLUMNS:
        if col not in shots.columns:
            shots[col] = None

    proba = model.predict_proba(shots)[:, 1]
    shots["xg"] = proba
    return shots


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to joblib model file")
    parser.add_argument("--data-root", required=True, help="StatsBomb data root (contains events/ and three-sixty/)")
    parser.add_argument("--match-id", required=True, help="Match id to score")
    parser.add_argument(
        "--require-360",
        action="store_true",
        help="Skip shots without 360 frames (recommended for 360 model)",
    )
    args = parser.parse_args()

    shots = score_match(
        model_path=Path(args.model_path),
        data_root=Path(args.data_root),
        match_id=args.match_id,
        require_360=args.require_360,
    )

    summary = summarize(shots)
    print("\nMatch summary (open play):")
    print(summary.to_string(index=False))

    total_xg = shots["xg"].sum()
    total_goals = shots["goal"].sum()
    print(f"\nTotal open-play xG: {total_xg:.3f}")
    print(f"Total open-play goals: {total_goals}")


if __name__ == "__main__":
    main()
