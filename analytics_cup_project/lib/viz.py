"""Visualizzazione del tracking SkillCorner.

Regola di codifica visiva, da non violare aggiungendo segnali:

- **riempimento** = squadra
- **contorno bianco** = posizione estrapolata, e nient'altro
- **dimensione + scia + etichetta** = giocatori evidenziati
- la palla la indica il pallino bianco

Il contorno è riservato alla qualità del dato. Se serve un'altra evidenziazione,
usa dimensione o scia — non ricolorare (si perde chi gioca con chi) e non
aggiungere un secondo tipo di bordo (diventa illeggibile).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mplsoccer import Pitch

from . import data

BG = "#252525"
HOME_C, AWAY_C = "#2388c8", "#e08a1e"
BALL_C, SCIA = "#f2f2f2", "#f5d90a"
INK, MUTED = "#e8eaed", "#9aa0a6"

_OFFSET_ETICHETTE = [2.6, -2.6, 4.6, -4.6, 6.6]


def anima_intervallo(
    match_id: int,
    frame_start: int,
    frame_end: int,
    output: str | Path,
    evidenzia: dict[int, str] | None = None,
    padding: int = 20,
    titolo: str = "",
    figsize: tuple[float, float] = (10, 6.6),
) -> Path:
    """Anima un intervallo di frame e salva una GIF.

    `evidenzia` mappa `player_id -> etichetta`: quei giocatori sono disegnati più
    grandi, con la scia del percorso osservato e il nome accanto.

    La scia usa solo le posizioni con `is_detected=True`: un percorso non deve
    mostrare tratti che la telecamera non ha visto.
    """
    meta = data.load_match_meta(match_id)
    evidenzia = evidenzia or {}

    squadra_di = {p["id"]: p["team_id"] for p in meta["players"]}
    home = meta["home_team"]["id"]

    fs, fe = frame_start - padding, frame_end + padding
    frames = [
        f for f in data.iter_tracking(match_id)
        if fs <= f["frame"] <= fe and f["player_data"]
    ]
    if not frames:
        raise ValueError(f"nessun frame popolato fra {fs} e {fe}")

    backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=meta["pitch_length"], pitch_width=meta["pitch_width"],
        pitch_color=BG, line_color=MUTED, linewidth=1.2,
    )
    fig, ax = pitch.draw(figsize=figsize)
    fig.set_facecolor(BG)

    punti = ax.scatter([], [], s=170, zorder=3)
    palla = ax.scatter([], [], s=70, c=BALL_C, edgecolors=BG, linewidths=1.2, zorder=5)
    scie = {pid: ax.plot([], [], lw=1.6, color=SCIA, alpha=0.55, zorder=2)[0]
            for pid in evidenzia}
    storia = {pid: ([], []) for pid in evidenzia}

    testo_titolo = ax.set_title("", color=INK, fontsize=10.5, loc="left", pad=14)
    etichette = [ax.text(0, 0, "", color=INK, fontsize=8.5, ha="center", zorder=6)
                 for _ in evidenzia]

    def disegna(i):
        fr = frames[i]
        xs, ys, colori, bordi, spessori, dim = [], [], [], [], [], []

        for p in fr["player_data"]:
            pid = p["player_id"]
            xs.append(p["x"])
            ys.append(p["y"])
            colori.append(HOME_C if squadra_di.get(pid) == home else AWAY_C)
            if p["is_detected"]:
                bordi.append("none")
                spessori.append(0)
            else:
                bordi.append("#ffffff")
                spessori.append(1.2)
            dim.append(320 if pid in evidenzia else 170)

            if pid in storia and p["is_detected"]:
                storia[pid][0].append(p["x"])
                storia[pid][1].append(p["y"])

        punti.set_offsets(list(zip(xs, ys)))
        punti.set_facecolor(colori)
        punti.set_edgecolor(bordi)
        punti.set_linewidth(spessori)
        punti.set_sizes(dim)

        b = fr.get("ball_data") or {}
        palla.set_offsets([[b["x"], b["y"]]] if b.get("x") is not None else [[-999, -999]])

        for pid, linea in scie.items():
            linea.set_data(*storia[pid])

        # Offset sfalsati: giocatori vicini hanno etichette che collidono.
        for t, (pid, nome), dy in zip(etichette, evidenzia.items(), _OFFSET_ETICHETTE):
            pos = next((p for p in fr["player_data"] if p["player_id"] == pid), None)
            if pos:
                t.set_position((pos["x"], pos["y"] + dy))
                t.set_va("bottom" if dy > 0 else "top")
                t.set_text(nome)

        dentro = frame_start <= fr["frame"] <= frame_end
        det = sum(p["is_detected"] for p in fr["player_data"])
        testo_titolo.set_text(
            f"{titolo}{'   ● in corso' if dentro and titolo else ''}\n"
            f"{fr['timestamp']}   ·   {det}/{len(fr['player_data'])} giocatori osservati"
            "   ·   contorno bianco = posizione estrapolata"
        )
        return [punti, palla, *scie.values(), testo_titolo, *etichette]

    anim = FuncAnimation(fig, disegna, frames=len(frames), interval=100, blit=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output), writer=PillowWriter(fps=10), savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    matplotlib.use(backend)
    return output


def anima_evento(
    match_id: int,
    event_id: str,
    output: str | Path,
    con_simultanei: bool = True,
    padding: int = 20,
) -> Path:
    """Anima un evento di `dynamic_events.csv`, preso per `event_id`.

    Con `con_simultanei`, evidenzia anche gli eventi dello stesso tipo che
    iniziano nello stesso frame — nelle corse senza palla è frequente che siano
    più di una, ed è il contesto che rende leggibile l'azione.
    """
    de = data.load_dynamic_events(match_id)
    ev = de[de.event_id == event_id].iloc[0]

    evidenzia = {int(ev.player_id): ev.player_name}
    if con_simultanei:
        insieme = de[(de.event_type == ev.event_type)
                     & (de.frame_start == ev.frame_start)
                     & (de.event_id != event_id)]
        for _, s in insieme.iterrows():
            evidenzia[int(s.player_id)] = s.player_name

    etichetta = f"{ev.event_type} «{ev.event_subtype}»" if isinstance(ev.event_subtype, str) else ev.event_type
    return anima_intervallo(
        match_id,
        int(ev.frame_start), int(ev.frame_end),
        output=output,
        evidenzia=evidenzia,
        padding=padding,
        titolo=f"{etichetta} — {ev.team_shortname}, {ev.player_name}",
    )


def osservabilita_evento(match_id: int, evento, tracking_buffer: dict | None = None) -> float:
    """Quota di frame dell'evento in cui il protagonista è osservato, non stimato.

    Serve a scartare gli eventi che descrivono posizioni ricostruite:
    fra le corse «behind» della partita 1886347 ce n'è una col corridore
    osservato al 6,8%, che animata mostrerebbe un movimento inventato.

    Nota: `dynamic_events.fully_extrapolated` non copre questo caso — è
    valorizzata solo per `passing_option` e `player_possession`.
    """
    fs, fe = int(evento.frame_start), int(evento.frame_end)
    pid = int(evento.player_id)

    if tracking_buffer is None:
        sorgente = (f for f in data.iter_tracking(match_id) if fs <= f["frame"] <= fe)
    else:
        sorgente = (tracking_buffer[f] for f in range(fs, fe + 1) if f in tracking_buffer)

    popolati = osservati = 0
    for fr in sorgente:
        if not fr["player_data"]:
            continue
        popolati += 1
        osservati += any(p["player_id"] == pid and p["is_detected"] for p in fr["player_data"])

    return osservati / popolati if popolati else 0.0
