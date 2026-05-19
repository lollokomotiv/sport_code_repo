#!/usr/bin/env python3
"""Score matches with an xG model and compare to realized goals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
import joblib
from mplsoccer import VerticalPitch

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

DEFAULT_PSHOT_MODEL_PATH = DEFAULT_MODELS_DIR / "xa_pshot_360.joblib"
DEFAULT_ZONE_XG_PATH     = DEFAULT_MODELS_DIR / "xa_zone_xg_lookup.joblib"
DEFAULT_XA_HOLDOUT_PATH  = DEFAULT_MODELS_DIR / "xa_holdout_match_ids.json"

INCOMPLETE_PASS_OUTCOMES = {"Incomplete", "Out", "Pass Offside", "Unknown"}

PASS_FEATURE_COLUMNS = [
    "receiver_dist_goal", "receiver_angle_goal", "passer_dist_goal",
    "pass_length", "pass_forward", "cross_field_y",
    "cross", "switch", "through_ball",
    "height", "body_part", "play_pattern", "technique",
    "nearest_def_to_receiver", "n_def_2m_receiver", "n_def_3m_receiver",
    "n_def_5m_receiver", "n_def_between_recv_goal", "n_teammates_5m_receiver",
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


def plot_shot_map(
    shots: pd.DataFrame,
    title: str = "Shot Map",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Pitch plot verticale con un punto per ogni tiro.

    Dimensione punto  = xG (più grande = più pericoloso)
    Colore punto      = team (due colori distinti)
    Contorno dorato   = gol segnato
    Annotazione       = giocatore + xG al passaggio del mouse non disponibile,
                        ma i top tiri (xG > 0.3) vengono annotati sul plot.
    """
    pitch = VerticalPitch(
        pitch_type="statsbomb",
        half=True,
        pitch_color="#1a1a2e",
        line_color="#e0e0e0",
        linewidth=1,
        goal_type="box",
    )

    fig, ax = pitch.draw(figsize=(8, 7))
    fig.patch.set_facecolor("#1a1a2e")

    teams = shots["team"].dropna().unique()
    palette = ["#00d4ff", "#ff6b6b"]
    team_colors = {team: palette[i % len(palette)] for i, team in enumerate(sorted(teams))}

    for _, row in shots.iterrows():
        x, y = row["shot_x"], row["shot_y"]
        xg = row.get("xg", 0.05)
        is_goal = row.get("goal", 0) == 1
        color = team_colors.get(row.get("team"), "#ffffff")

        size = max(50, xg * 800)

        # Gol: contorno dorato spesso
        if is_goal:
            pitch.scatter(
                x, y, ax=ax,
                s=size * 1.6,
                color="gold",
                edgecolors="white",
                linewidth=1.5,
                zorder=4,
                alpha=0.95,
            )

        pitch.scatter(
            x, y, ax=ax,
            s=size,
            color=color,
            edgecolors="white" if not is_goal else "#1a1a2e",
            linewidth=0.8,
            alpha=0.85,
            zorder=5,
        )

        # Annota i tiri più pericolosi (xG > 0.3)
        if xg > 0.3:
            player = row.get("player", "")
            label = f"{player.split()[-1] if player else ''}\n{xg:.2f}"
            ax.annotate(
                label,
                xy=(y, x),
                fontsize=6,
                color="white",
                ha="center",
                va="bottom",
                xytext=(0, 6),
                textcoords="offset points",
                path_effects=[pe.withStroke(linewidth=2, foreground="#1a1a2e")],
                zorder=6,
            )

    # Legenda squadre
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=t)
        for t, c in team_colors.items()
    ]
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gold",
               markersize=10, label="Goal", markeredgecolor="white", markeredgewidth=1)
    )
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(legend_elements),
        frameon=False,
        fontsize=8,
        labelcolor="white",
        bbox_to_anchor=(0.5, -0.04),
    )

    # Titolo e stats xG
    summary = (
        shots.groupby("team")
        .agg(shots_n=("event_id", "count"), xg_sum=("xg", "sum"), goals=("goal", "sum"))
        .reset_index()
        .sort_values("xg_sum", ascending=False)
    )
    stats_lines = [
        f"{r['team']}: {r['shots_n']} tiri | xG {r['xg_sum']:.2f} | Gol {int(r['goals'])}"
        for _, r in summary.iterrows()
    ]
    subtitle = "  —  ".join(stats_lines)

    fig.suptitle(title, color="white", fontsize=13, fontweight="bold", y=1.01)
    ax.set_title(subtitle, color="#aaaaaa", fontsize=7.5, pad=6)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Shot map salvata in: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# xA — Expected Assists
# xA(key_pass)     = P_shot(pass_features) × xG_model(shot_features)
# xA(non_key_pass) = P_shot(pass_features) × zone_xg(end_location)
# ─────────────────────────────────────────────────────────────────────────────

def _pass_geometry(start_loc: List, end_loc: List) -> Optional[dict]:
    if not (isinstance(start_loc, list) and isinstance(end_loc, list)):
        return None
    sx, sy = start_loc[0], start_loc[1]
    ex, ey = end_loc[0], end_loc[1]
    if sx < 60:
        sx, sy = 120 - sx, 80 - sy
        ex, ey = 120 - ex, 80 - ey
    recv_dist  = math.hypot(ex - GOAL_X, ey - GOAL_Y)
    angle_den  = (ex - GOAL_X)**2 + (ey - GOAL_Y)**2 - (GOAL_WIDTH / 2)**2
    recv_angle = max(0.0, math.atan2(GOAL_WIDTH * abs(ey - GOAL_Y), angle_den)) if angle_den != 0 else 0.0
    return {
        "start_x_norm": sx, "start_y_norm": sy,
        "end_x_norm":   ex, "end_y_norm":   ey,
        "pass_length":         math.hypot(ex - sx, ey - sy),
        "pass_forward":        ex - sx,
        "receiver_dist_goal":  recv_dist,
        "receiver_angle_goal": recv_angle,
        "passer_dist_goal":    math.hypot(sx - GOAL_X, sy - GOAL_Y),
        "cross_field_y":       abs(ey - GOAL_Y),
    }


def _pass_meta(pass_data: dict) -> dict:
    return {
        "height":       pass_data.get("height", {}).get("name"),
        "body_part":    pass_data.get("body_part", {}).get("name"),
        "technique":    pass_data.get("technique", {}).get("name"),
        "cross":        bool(pass_data.get("cross")),
        "switch":       bool(pass_data.get("switch")),
        "through_ball": bool(pass_data.get("through_ball")),
    }


def _pass_360_features(frame: dict, end_x: float, end_y: float) -> dict:
    ff = frame.get("freeze_frame") or []
    opponents = [p for p in ff if p.get("teammate") is False and not p.get("keeper")]
    opp_locs  = [
        (p["location"][0], p["location"][1]) for p in opponents
        if isinstance(p.get("location"), list) and len(p["location"]) >= 2
    ]
    if opp_locs:
        dists       = [dist((end_x, end_y), loc) for loc in opp_locs]
        nearest_def = min(dists)
        n_def_2m    = sum(1 for d in dists if d <= 2)
        n_def_3m    = sum(1 for d in dists if d <= 3)
        n_def_5m    = sum(1 for d in dists if d <= 5)
    else:
        nearest_def = None
        n_def_2m = n_def_3m = n_def_5m = 0
    n_def_recv_goal = sum(
        1 for l in opp_locs
        if l[0] > end_x and abs(l[1] - GOAL_Y) <= (GOAL_WIDTH / 2 + 3)
    )
    teammates = [p for p in ff if p.get("teammate") is True and not p.get("actor")]
    tm_locs   = [
        (p["location"][0], p["location"][1]) for p in teammates
        if isinstance(p.get("location"), list) and len(p["location"]) >= 2
    ]
    n_tm_5m = sum(1 for l in tm_locs if dist((end_x, end_y), l) <= 5)
    return {
        "nearest_def_to_receiver":  nearest_def,
        "n_def_2m_receiver":        n_def_2m,
        "n_def_3m_receiver":        n_def_3m,
        "n_def_5m_receiver":        n_def_5m,
        "n_def_between_recv_goal":  n_def_recv_goal,
        "n_teammates_5m_receiver":  n_tm_5m,
    }


def _zone_xg_lookup(end_x: float, end_y: float, zone_data: dict) -> float:
    import bisect
    x_bins      = zone_data["x_bins"]
    y_bins      = zone_data["y_bins"]
    zone_xg_map = zone_data["zone_xg_map"]
    global_avg  = zone_data["global_avg_xg"]
    if end_x < 60:
        end_x, end_y = 120 - end_x, 80 - end_y
    d      = math.hypot(end_x - GOAL_X, end_y - GOAL_Y)
    aden   = (end_x - GOAL_X)**2 + (end_y - GOAL_Y)**2 - (GOAL_WIDTH / 2)**2
    angle  = max(0.0, math.atan2(GOAL_WIDTH * abs(end_y - GOAL_Y), aden)) if aden != 0 else 0.0
    x_val  = max(float(x_bins[0]),  GOAL_X - d)
    y_val  = min(float(y_bins[-1]) - 0.01, GOAL_Y + angle * 10)
    xi = max(0, min(bisect.bisect_right(x_bins, x_val) - 1, len(x_bins) - 2))
    yi = max(0, min(bisect.bisect_right(y_bins, y_val) - 1, len(y_bins) - 2))
    return zone_xg_map.get((xi, yi), global_avg)


def build_pass_rows(
    events: List[dict],
    frames_by_event: Dict[str, dict],
    match_id: str,
) -> List[dict]:
    shots_by_kp: Dict[str, dict] = {
        e["shot"].get("key_pass_id"): e
        for e in events
        if e.get("type", {}).get("name") == "Shot" and e.get("shot", {}).get("key_pass_id")
    }
    rows = []
    for e in events:
        if e.get("type", {}).get("name") != "Pass":
            continue
        p = e.get("pass", {})
        outcome_name = p.get("outcome", {}).get("name") if p.get("outcome") else None
        if outcome_name in INCOMPLETE_PASS_OUTCOMES:
            continue
        event_id  = e.get("id")
        start_loc = e.get("location", [])
        end_loc   = p.get("end_location", [])
        frame     = frames_by_event.get(event_id, {})
        geom      = _pass_geometry(start_loc, end_loc)
        if geom is None:
            continue
        shot = shots_by_kp.get(event_id)
        row = {
            "match_id":      match_id,
            "event_id":      event_id,
            "team":          e.get("team", {}).get("name"),
            "player":        e.get("player", {}).get("name"),
            "pass_start_x":  start_loc[0] if len(start_loc) > 0 else None,
            "pass_start_y":  start_loc[1] if len(start_loc) > 1 else None,
            "pass_end_x":    end_loc[0]   if len(end_loc)   > 0 else None,
            "pass_end_y":    end_loc[1]   if len(end_loc)   > 1 else None,
            "shot_assist":   bool(p.get("shot_assist")),
            "goal_assist":   bool(p.get("goal_assist")),
            "shot_event_id": shot.get("id") if shot else None,
            "play_pattern":  e.get("play_pattern", {}).get("name"),
            **geom,
            **_pass_meta(p),
            **(
                _pass_360_features(frame, geom["end_x_norm"], geom["end_y_norm"])
                if frame
                else {c: None for c in [
                    "nearest_def_to_receiver", "n_def_2m_receiver", "n_def_3m_receiver",
                    "n_def_5m_receiver", "n_def_between_recv_goal", "n_teammates_5m_receiver",
                ]}
            ),
        }
        rows.append(row)
    return rows


def score_xa_match(
    pshot_model_path: Path,
    xg_model_path: Path,
    zone_xg_path: Path,
    data_root: Path,
    match_id: str,
) -> pd.DataFrame:
    """Score xA for every completed pass in a match.

    Key pass:     xA = P_shot(pass_features) × xG_model(shot_features)
    Non-key pass: xA = P_shot(pass_features) × zone_xg(pass_end_location)
    """
    pshot_model = joblib.load(pshot_model_path)
    xg_model    = joblib.load(xg_model_path)
    zone_data   = joblib.load(zone_xg_path)

    events          = load_events(data_root / "events", match_id)
    frames_by_event = load_three_sixty(data_root / "three-sixty", match_id)

    pass_rows = build_pass_rows(events, frames_by_event, match_id)
    passes    = pd.DataFrame(pass_rows)
    if passes.empty:
        return passes

    for col in PASS_FEATURE_COLUMNS:
        if col not in passes.columns:
            passes[col] = None

    passes["p_shot"]  = pshot_model.predict_proba(passes[PASS_FEATURE_COLUMNS])[:, 1]
    passes["shot_xg"] = None

    # xG from the actual shot for key passes
    key_mask = passes["shot_assist"] & passes["shot_event_id"].notna()
    if key_mask.any():
        shot_feats, valid_idx = [], []
        events_by_id = {e.get("id"): e for e in events}
        for idx, row in passes[key_mask].iterrows():
            shot_event = events_by_id.get(row["shot_event_id"])
            if shot_event is None:
                continue
            loc = shot_event.get("location", [])
            if len(loc) < 2:
                continue
            x, y  = loc[0], loc[1]
            frame = frames_by_event.get(row["shot_event_id"], {})
            feat  = {
                "distance":      distance_to_goal(x, y),
                "angle":         shot_angle(x, y),
                "body_part":     shot_event.get("shot", {}).get("body_part", {}).get("name"),
                "shot_type":     shot_event.get("shot", {}).get("type", {}).get("name"),
                "shot_technique": shot_event.get("shot", {}).get("technique", {}).get("name"),
                "play_pattern":  shot_event.get("play_pattern", {}).get("name"),
                "first_time":    shot_event.get("shot", {}).get("first_time"),
                "one_on_one":    shot_event.get("shot", {}).get("one_on_one"),
                "under_pressure": shot_event.get("under_pressure"),
            }
            feat.update(extract_360_features(frame, x, y) if frame else
                        {c: None for c in REQUIRED_COLUMNS if c not in feat})
            shot_feats.append(feat)
            valid_idx.append(idx)

        if shot_feats:
            shot_df = pd.DataFrame(shot_feats)
            for col in REQUIRED_COLUMNS:
                if col not in shot_df.columns:
                    shot_df[col] = None
            passes.loc[valid_idx, "shot_xg"] = xg_model.predict_proba(shot_df)[:, 1]

    # Zone xG fallback for everything else
    passes["zone_xg"] = passes.apply(
        lambda r: _zone_xg_lookup(
            r.get("pass_end_x") or 60.0,
            r.get("pass_end_y") or 40.0,
            zone_data,
        ),
        axis=1,
    )

    passes["xg_used"] = passes["shot_xg"].fillna(passes["zone_xg"])
    passes["xa"]      = passes["p_shot"] * passes["xg_used"]
    return passes


def score_xa_matches(
    pshot_model_path: Path,
    xg_model_path: Path,
    zone_xg_path: Path,
    data_root: Path,
    match_ids: List[str],
) -> pd.DataFrame:
    frames = [
        score_xa_match(pshot_model_path, xg_model_path, zone_xg_path, data_root, mid)
        for mid in match_ids
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_xa(passes: pd.DataFrame) -> pd.DataFrame:
    return (
        passes.groupby("team", dropna=False)
        .agg(
            passes=("event_id", "count"),
            xa=("xa", "sum"),
            key_passes=("shot_assist", "sum"),
            goal_assists=("goal_assist", "sum"),
        )
        .reset_index()
        .sort_values("xa", ascending=False)
    )


def plot_assist_map(
    passes: pd.DataFrame,
    title: str = "Assist Map",
    save_path: Optional[Path] = None,
) -> "plt.Figure":
    """Pitch plot with one arrow+dot per key pass, sized by xA.

    Arrow: from pass start to end (receiver position).
    Dot size: xA value. Gold: goal assist.
    Top key passes (xA > 0.08) are annotated with player name.
    """
    key_passes = passes[passes["shot_assist"]].copy()
    if key_passes.empty:
        raise ValueError("Nessun key pass da visualizzare.")

    pitch = VerticalPitch(
        pitch_type="statsbomb",
        half=True,
        pitch_color="#1a1a2e",
        line_color="#e0e0e0",
        linewidth=1,
        goal_type="box",
    )
    fig, ax = pitch.draw(figsize=(8, 7))
    fig.patch.set_facecolor("#1a1a2e")

    teams = key_passes["team"].dropna().unique()
    palette = ["#00d4ff", "#ff6b6b"]
    team_colors = {t: palette[i % len(palette)] for i, t in enumerate(sorted(teams))}

    for _, row in key_passes.iterrows():
        x_end   = row.get("pass_end_x")
        y_end   = row.get("pass_end_y")
        x_start = row.get("pass_start_x")
        y_start = row.get("pass_start_y")
        if x_end is None or y_end is None:
            continue

        xa      = float(row.get("xa", 0.01))
        is_goal = bool(row.get("goal_assist", False))
        color   = team_colors.get(row.get("team"), "#ffffff")
        arrow_color = "gold" if is_goal else color

        if x_start is not None and y_start is not None:
            pitch.arrows(
                x_start, y_start, x_end, y_end, ax=ax,
                width=1.5, headwidth=5, headlength=4,
                color=arrow_color,
                alpha=0.65 if is_goal else 0.3,
                zorder=3,
            )

        size = max(40, xa * 1200)
        if is_goal:
            pitch.scatter(x_end, y_end, ax=ax, s=size * 1.6,
                          color="gold", edgecolors="white", linewidth=1.5,
                          zorder=5, alpha=0.95)
        pitch.scatter(x_end, y_end, ax=ax, s=size,
                      color=color, edgecolors="white" if not is_goal else "#1a1a2e",
                      linewidth=0.8, alpha=0.85, zorder=6)

        if xa > 0.08:
            player = row.get("player", "")
            label  = f"{player.split()[-1] if player else ''}\n{xa:.2f}"
            ax.annotate(
                label, xy=(y_end, x_end),
                fontsize=6, color="white", ha="center", va="bottom",
                xytext=(0, 6), textcoords="offset points",
                path_effects=[pe.withStroke(linewidth=2, foreground="#1a1a2e")],
                zorder=7,
            )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=t)
        for t, c in team_colors.items()
    ]
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gold",
               markersize=10, label="Goal assist",
               markeredgecolor="white", markeredgewidth=1)
    )
    ax.legend(handles=legend_elements, loc="lower center",
              ncol=len(legend_elements), frameon=False, fontsize=8,
              labelcolor="white", bbox_to_anchor=(0.5, -0.04))

    summary = (
        key_passes.groupby("team")
        .agg(kp_n=("event_id", "count"), xa_sum=("xa", "sum"), goals=("goal_assist", "sum"))
        .reset_index()
        .sort_values("xa_sum", ascending=False)
    )
    stats_lines = [
        f"{r['team']}: {r['kp_n']} key pass | xA {r['xa_sum']:.2f} | Assist {int(r['goals'])}"
        for _, r in summary.iterrows()
    ]
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold", y=1.01)
    ax.set_title("  —  ".join(stats_lines), color="#aaaaaa", fontsize=7.5, pad=6)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Assist map salvata in: {save_path}")

    return fig


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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Genera e mostra la shot map della partita",
    )
    parser.add_argument(
        "--plot-output",
        help="Salva la shot map in questo path (es: shot_map.png)",
    )
    # ── xA options ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--score-xa",
        action="store_true",
        help="Calcola xA per tutti i passaggi completati (oltre a xG)",
    )
    parser.add_argument(
        "--pshot-model-path",
        default=str(DEFAULT_PSHOT_MODEL_PATH),
        help="Path al modello P_shot (xa_pshot_360.joblib)",
    )
    parser.add_argument(
        "--zone-xg-path",
        default=str(DEFAULT_ZONE_XG_PATH),
        help="Path al zone xG lookup (xa_zone_xg_lookup.joblib)",
    )
    parser.add_argument(
        "--plot-assist-map",
        action="store_true",
        help="Genera e mostra l'assist map della partita",
    )
    parser.add_argument(
        "--assist-map-output",
        help="Salva l'assist map in questo path (es: assist_map.png)",
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

    if args.plot or args.plot_output:
        save_path = Path(args.plot_output) if args.plot_output else None
        fig = plot_shot_map(shots, title=f"Shot Map — match {args.match_id}", save_path=save_path)
        if args.plot:
            plt.show()

    if args.score_xa:
        passes = score_xa_match(
            pshot_model_path=Path(args.pshot_model_path),
            xg_model_path=model_path,
            zone_xg_path=Path(args.zone_xg_path),
            data_root=data_root,
            match_id=args.match_id,
        )
        xa_summary = summarize_xa(passes)
        print(f"\nxA summary — match {args.match_id}:")
        print(xa_summary.to_string(index=False))
        total_xa      = passes["xa"].sum()
        total_assists = passes["goal_assist"].sum()
        print(f"\nTotal xA: {total_xa:.3f}")
        print(f"Total goal assists: {int(total_assists)}")

        if args.plot_assist_map or args.assist_map_output:
            save_path = Path(args.assist_map_output) if args.assist_map_output else None
            fig = plot_assist_map(
                passes, title=f"Assist Map — match {args.match_id}", save_path=save_path
            )
            if args.plot_assist_map:
                plt.show()


if __name__ == "__main__":
    main()
