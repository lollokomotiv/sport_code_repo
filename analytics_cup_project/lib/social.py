"""Media per i post su X: formato, fonte impressa nell'immagine, limiti della piattaforma.

La codifica visiva è quella di `lib/viz.py` (riempimento = squadra, contorno
bianco = posizione estrapolata). Qui si aggiunge solo ciò che serve a un media
che verrà visto piccolo, su telefono, e ripubblicato da solo:

- formato 16:9, testo più grande;
- la fonte scritta nell'immagine, non solo nel post;
- la legenda dell'estrapolazione sempre presente quando si disegna il campo;
- un controllo dei limiti di X prima di consegnare il file.

Il codice delle figure di un post sta nello script che produce i numeri del post
(`scripts/`), non nella shell e non nel notebook: vedi la skill `post-x`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mplsoccer import Pitch

from . import data
from .viz import AWAY_C, BG, HOME_C, INK, MUTED, anima_intervallo

FONTE = "Data: SkillCorner open data · A-League 2024/25"

# 16:9. Le dimensioni in pixel dipendono dai dpi di ciascun formato.
FIGSIZE = (12, 6.75)
DPI = {"png": 150, "gif": 100, "mp4": 1280 / 12}   # 1800×1012, 1200×675, 1280×720

# Limiti di X verificati a ottobre 2026 su fonti secondarie (le pagine ufficiali
# help.x.com e developer.x.com non sono leggibili senza login). Da ricontrollare
# se un upload fallisce.
LIMITI = {
    "png": {"mb": 5},
    "gif": {"mb": 15, "mb_mobile": 5, "lato_max": (1280, 1080), "frame_max": 350},
    "mp4": {"mb": 512, "secondi_max": 140},
    "alt_text_caratteri": 1000,
    "immagini_per_post": 4,
}


# ---------------------------------------------------------------------------
# Figure statiche
# ---------------------------------------------------------------------------

def figura(titolo: str, sottotitolo: str = "", fonte: str = FONTE, campo_meta: dict | None = None):
    """Figura 16:9 con titolo, sottotitolo e fonte già impaginati.

    Con `campo_meta` (il match.json) disegna il campo e restituisce (fig, ax, pitch);
    altrimenti un asse per un grafico e restituisce (fig, ax, None).
    Il titolo deve dire il risultato, non l'asse.
    """
    if campo_meta is not None:
        # mplsoccer ignora subplots_adjust: il campo va in un riquadro esplicito,
        # sotto l'intestazione e sopra la fonte.
        pitch = Pitch(pitch_type="skillcorner", pitch_length=campo_meta["pitch_length"],
                      pitch_width=campo_meta["pitch_width"], pitch_color=BG,
                      line_color=MUTED, linewidth=1.2)
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_axes([0.02, 0.07, 0.96, 0.74 if sottotitolo else 0.79])
        pitch.draw(ax=ax)
    else:
        pitch = None
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=12)
        for s in ax.spines.values():
            s.set_color(MUTED)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(13)
    fig.set_facecolor(BG)
    fig.text(0.02, 0.965, titolo, color=INK, fontsize=19, fontweight="bold", ha="left", va="top")
    if sottotitolo:
        fig.text(0.02, 0.905, sottotitolo, color=MUTED, fontsize=13, ha="left", va="top")
    fig.text(0.02, 0.02, fonte, color=MUTED, fontsize=10.5, ha="left", va="bottom")
    if pitch is None:
        fig.subplots_adjust(top=0.84 if sottotitolo else 0.89, bottom=0.12, left=0.07, right=0.98)
    return fig, ax, pitch


def legenda_campo(ax, meta: dict, extra: list[tuple[str, dict]] | None = None):
    """Legenda per un campo: le due squadre e il contorno dell'estrapolazione.

    `extra` aggiunge voci come [("time to reach the carrier", dict(color=..., lw=2))].
    L'estrapolazione è sempre in legenda: un frame broadcast senza questa
    informazione presenta posizioni stimate come se fossero osservate.
    """
    voci = [
        Line2D([], [], marker="o", ls="", color=HOME_C, markersize=11, label=meta["home_team"]["short_name"]),
        Line2D([], [], marker="o", ls="", color=AWAY_C, markersize=11, label=meta["away_team"]["short_name"]),
        Line2D([], [], marker="o", ls="", color=BG, markeredgecolor="#ffffff", markeredgewidth=1.6,
               markersize=11, label="position estimated (off camera)"),
    ]
    for etichetta, stile in extra or []:
        voci.append(Line2D([], [], label=etichetta, **stile))
    # In orizzontale nell'intestazione, a destra: sul campo coprirebbe giocatori.
    leg = ax.figure.legend(handles=voci, loc="upper right", bbox_to_anchor=(0.98, 0.975),
                           ncol=len(voci), fontsize=11, frameon=False, labelcolor=INK,
                           handletextpad=0.4, columnspacing=1.2)
    return leg


def salva_png(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI["png"], facecolor=BG)
    plt.close(fig)
    controlla(path)
    return path


# ---------------------------------------------------------------------------
# Animazioni
# ---------------------------------------------------------------------------

def anima(match_id: int, frame_start: int, frame_end: int, output: str | Path,
          evidenzia: dict[int, str] | None = None, titolo: str = "",
          fonte: str = FONTE, padding: int = 20) -> Path:
    """GIF o MP4 di un intervallo di tracking, secondo l'estensione di `output`.

    A 10 fps la GIF regge al massimo 35 s (350 frame): oltre, usare .mp4.
    Il contorno bianco dell'estrapolazione e la fonte sono già nell'immagine.
    """
    output = Path(output)
    formato = output.suffix.lower().lstrip(".")
    n_frame = frame_end - frame_start + 1 + 2 * padding
    if formato == "gif":
        if n_frame > LIMITI["gif"]["frame_max"]:
            raise ValueError(f"{n_frame} frame: oltre i {LIMITI['gif']['frame_max']} di una GIF su X, usa .mp4")
        writer = None
    elif formato == "mp4":
        import imageio_ffmpeg
        from matplotlib import rcParams
        from matplotlib.animation import FFMpegWriter
        rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(fps=10, codec="libx264", bitrate=4000,
                              extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    else:
        raise ValueError(f"formato non gestito: {formato!r} (gif o mp4)")

    # Nell'animazione non c'è spazio per una legenda: le squadre vanno nella riga della fonte.
    meta = data.load_match_meta(match_id)
    fonte = (f"{fonte}   ·   blue: {meta['home_team']['short_name']}"
             f"   ·   orange: {meta['away_team']['short_name']}")
    anima_intervallo(match_id, frame_start, frame_end, output, evidenzia=evidenzia,
                     padding=padding, titolo=titolo, figsize=FIGSIZE,
                     writer=writer, dpi=DPI[formato], fonte=fonte, lingua="en", scala=1.35)
    controlla(output)
    return output


def anima_evento(match_id: int, event_id: str, output: str | Path, **kw) -> Path:
    """Anima un evento di dynamic_events.csv, evidenziandone il protagonista."""
    de = data.load_dynamic_events(match_id)
    ev = de[de.event_id == event_id].iloc[0]
    kw.setdefault("evidenzia", {int(ev.player_id): ev.player_name})
    kw.setdefault("titolo", f"{ev.player_name} ({ev.team_shortname}), {ev.event_type.replace('_', ' ')}")
    return anima(match_id, int(ev.frame_start), int(ev.frame_end), output, **kw)


# ---------------------------------------------------------------------------
# Controlli
# ---------------------------------------------------------------------------

def controlla(path: str | Path) -> dict:
    """Misura il file e lo confronta con i limiti di X. Solleva se li supera.

    Restituisce le misure, così chi chiama può riportarle all'utente; gli
    avvisi (non bloccanti) sono in `avvisi`.
    """
    from PIL import Image

    path = Path(path)
    formato = path.suffix.lower().lstrip(".")
    formato = "png" if formato in ("png", "jpg", "jpeg") else formato
    mb = path.stat().st_size / 1e6
    info = {"file": str(path), "formato": formato, "mb": round(mb, 2), "avvisi": []}
    lim = LIMITI[formato]

    if formato in ("png", "gif"):
        with Image.open(path) as im:
            info["pixel"] = im.size
            info["frame"] = getattr(im, "n_frames", 1)
    if formato == "mp4":
        import imageio_ffmpeg
        n, sec = imageio_ffmpeg.count_frames_and_secs(str(path))
        info["frame"], info["secondi"] = n, round(sec, 1)
        if sec > lim["secondi_max"]:
            raise ValueError(f"{path.name}: {sec:.0f} s, oltre i {lim['secondi_max']} s di X")

    if mb > lim["mb"]:
        raise ValueError(f"{path.name}: {mb:.1f} MB, oltre i {lim['mb']} MB di X per {formato}")
    if formato == "gif":
        w, h = info["pixel"]
        wmax, hmax = lim["lato_max"]
        if w > wmax or h > hmax:
            raise ValueError(f"{path.name}: {w}×{h}, oltre {wmax}×{hmax}")
        if info["frame"] > lim["frame_max"]:
            raise ValueError(f"{path.name}: {info['frame']} frame, oltre {lim['frame_max']}")
        if mb > lim["mb_mobile"]:
            info["avvisi"].append(f"{mb:.1f} MB: oltre i ~{lim['mb_mobile']} MB, l'app mobile può ricomprimerla")
    return info


def fotogrammi(path: str | Path, out_dir: str | Path, n: int = 3) -> list[Path]:
    """Estrae `n` fotogrammi equidistanti da una GIF o da un MP4, come PNG.

    Serve a guardare un'animazione prima di consegnarla: il controllo dei limiti
    non vede etichette sovrapposte o un'azione fuori inquadratura.
    """
    import numpy as np
    from PIL import Image, ImageSequence

    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".gif":
        with Image.open(path) as im:
            frames = [np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(im)]
    else:
        import imageio_ffmpeg
        reader = imageio_ffmpeg.read_frames(str(path))
        w, h = next(reader)["size"]
        frames = [np.frombuffer(f, dtype=np.uint8).reshape(h, w, 3) for f in reader]

    idx = [round(k * (len(frames) - 1) / max(n - 1, 1)) for k in range(n)]
    out = []
    for k, i in enumerate(idx):
        p = out_dir / f"{path.stem}_{k}.png"
        Image.fromarray(frames[i]).save(p)
        out.append(p)
    return out
