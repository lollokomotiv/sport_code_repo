"""τ_opp all'inizio e al rilascio di ogni player_possession, su tutte le partite.

Uso (dalla cartella del progetto):
    python scripts/calcola_tau.py            # tutte le partite
    python scripts/calcola_tau.py 1886347    # solo alcune

Scrive reports/tau_opp.csv: una riga per player_possession, portieri esclusi.
Le righe senza tracking al frame richiesto restano, con τ_opp = NaN.

Scelte (fissate prima di guardare i risultati):
- posizione del portatore x_b = posizione di tracking del giocatore in possesso
  al frame_start (t_get) e al frame_end (t_rel);
- velocità con derivata centrale sulle posizioni, senza lisciatura, per giocatore
  e per periodo (lib.pressione.velocita con finestra=None);
- α e V_max del paper (lib.pressione.ALPHA, V_MAX);
- "avversario più vicino" = quello che realizza τ_opp (argmin del tempo di arrivo);
  per confronto si salva anche quello più vicino in distanza.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import data  # noqa: E402
from lib.pressione import tempi_arrivo, velocita  # noqa: E402

OUT = ROOT / "reports" / "tau_opp.csv"
FPS = 10


def carica_tracking(match_id):
    """Array frame × giocatore di x, y, is_detected, e il periodo di ogni frame."""
    frames = list(data.iter_tracking(match_id))
    f0 = frames[0]["frame"]
    n = frames[-1]["frame"] - f0 + 1
    ids = sorted({p["player_id"] for fr in frames for p in fr["player_data"]})
    col = {pid: j for j, pid in enumerate(ids)}
    X = np.full((n, len(ids)), np.nan)
    Y = np.full((n, len(ids)), np.nan)
    D = np.zeros((n, len(ids)), dtype=bool)
    periodo = np.zeros(n, dtype=int)
    for fr in frames:
        i = fr["frame"] - f0
        periodo[i] = fr["period"] or 0
        for p in fr["player_data"]:
            j = col[p["player_id"]]
            X[i, j], Y[i, j], D[i, j] = p["x"], p["y"], p["is_detected"]
    return f0, col, X, Y, D, periodo


def velocita_per_periodo(X, Y, periodo):
    VX = np.full_like(X, np.nan)
    VY = np.full_like(Y, np.nan)
    for per in np.unique(periodo[periodo > 0]):
        righe = np.flatnonzero(periodo == per)
        a, b = righe[0], righe[-1] + 1
        for j in range(X.shape[1]):
            if b - a >= 2:
                VX[a:b, j], VY[a:b, j] = velocita(X[a:b, j], Y[a:b, j], fps=FPS)
    return VX, VY


def pressione_al_frame(frame, portatore, avversari, f0, col, X, Y, VX, VY, D):
    """τ_opp sul portatore a un frame, più chi lo realizza e se è osservato."""
    out = dict(x_b=np.nan, y_b=np.nan, portatore_detected=np.nan, tau_opp=np.nan,
               n_avv_validi=0, avv_tau_id=np.nan, avv_tau_detected=np.nan,
               avv_tau_dist=np.nan, avv_dist_id=np.nan, avv_dist_detected=np.nan)
    i = int(frame) - f0
    if not (0 <= i < X.shape[0]) or portatore not in col:
        return out
    jb = col[portatore]
    xb, yb = X[i, jb], Y[i, jb]
    if not np.isfinite(xb):
        return out
    out.update(x_b=xb, y_b=yb, portatore_detected=bool(D[i, jb]))

    js = [col[p] for p in avversari if p in col and np.isfinite(X[i, col[p]])]
    if not js:
        return out
    js = np.array(js)
    pos = np.column_stack([X[i, js], Y[i, js]])
    vel = np.column_stack([VX[i, js], VY[i, js]])
    tempi = tempi_arrivo((xb, yb), pos, vel)
    dist = np.hypot(pos[:, 0] - xb, pos[:, 1] - yb)
    ids = np.array(list(col))[js]
    out["n_avv_validi"] = int(np.isfinite(tempi).sum())
    kd = int(np.argmin(dist))
    out.update(avv_dist_id=ids[kd], avv_dist_detected=bool(D[i, js[kd]]))
    if np.isfinite(tempi).any():
        k = int(np.nanargmin(tempi))
        out.update(tau_opp=tempi[k], avv_tau_id=ids[k],
                   avv_tau_detected=bool(D[i, js[k]]), avv_tau_dist=dist[k])
    return out


def calcola_partita(match_id):
    meta = data.load_match_meta(match_id)
    squadra = {p["id"]: p["team_id"] for p in meta["players"]}
    L = meta["pitch_length"]

    de = data.load_dynamic_events(match_id)
    pp = de[(de.event_type == "player_possession") & (de.player_position != "GK")]

    f0, col, X, Y, D, periodo = carica_tracking(match_id)
    VX, VY = velocita_per_periodo(X, Y, periodo)

    righe = []
    for ev in pp.itertuples(index=False):
        avversari = [p for p, t in squadra.items() if t != ev.team_id]
        porta_x = L / 2 if ev.attacking_side == "left_to_right" else -L / 2
        riga = dict(
            match_id=match_id, event_id=ev.event_id, player_id=ev.player_id,
            team_id=ev.team_id, player_position=ev.player_position, period=ev.period,
            frame_start=ev.frame_start, frame_end=ev.frame_end,
            direct_play=ev.frame_start == ev.frame_end,
            start_type=ev.start_type, end_type=ev.end_type, pass_outcome=ev.pass_outcome,
            game_interruption_before=ev.game_interruption_before,
            game_interruption_after=ev.game_interruption_after,
            attacking_side=ev.attacking_side, fully_extrapolated=ev.fully_extrapolated,
            time_to_impact_start_id=ev.time_to_impact_start_id,
            time_to_impact_end_id=ev.time_to_impact_end_id,
        )
        for quando, frame in (("start", ev.frame_start), ("end", ev.frame_end)):
            r = pressione_al_frame(frame, ev.player_id, avversari, f0, col, X, Y, VX, VY, D)
            riga.update({f"{k}_{quando}": v for k, v in r.items()})
            riga[f"dist_porta_{quando}"] = np.hypot(r["x_b"] - porta_x, r["y_b"])
        riga["progressione"] = riga["dist_porta_start"] - riga["dist_porta_end"]
        righe.append(riga)
    return pd.DataFrame(righe)


def main(match_ids):
    parti = []
    for m in match_ids:
        df = calcola_partita(m)
        print(f"{m}: {len(df)} possessi, τ_opp start valido {df.tau_opp_start.notna().mean():.1%}",
              flush=True)
        parti.append(df)
    out = pd.concat(parti, ignore_index=True)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"scritto {OUT} ({len(out)} righe)")


if __name__ == "__main__":
    ids = [int(a) for a in sys.argv[1:]] or data.list_matches()
    main(ids)
