"""Body pose SkillCorner: caricamento e orientamento del busto.

Il pose è un file a parte, con regole diverse dal tracking. Le cose che
cambiano e che è facile sbagliare:

- **25 fps**, non 10: `pose_frame = 2.5 * tracking_frame`. Solo un frame di
  pose su 5 cade esattamente su un frame di tracking pari.
- **Solo per giocatori rilevati.** `joints` è `None` quando il pose non è stato
  risolto, anche se `x`/`y` ci sono. Il pose eredita quindi lo stesso limite
  del tracking, amplificato: su partita intera circa un terzo dei player-frame
  ha i giunti.
- **`z` è relativa al centroide del giocatore**, non alle coordinate del campo:
  non è un'altezza dal suolo e può essere negativa. Per direzioni sul piano si
  usano solo `x` e `y`.
- Pose e tracking sono generati separatamente: piccoli disallineamenti fra
  `is_detected` del tracking e presenza dei giunti sono attesi.

Due partite su venti hanno il pose, e i file interi (~600 MB compressi, 3,3 GB
di JSON) stanno su Hugging Face, non nel repo dei dati.

    from lib import pose
    pose.scarica(1925299)                 # una volta, ~600 MB
    for fr in pose.iter_match(1925299):   # streaming, memoria costante
        ...
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .data import OPENDATA

BODYPOSE = OPENDATA / "data" / "bodypose"
MANIFEST = BODYPOSE / "MANIFEST.json"
CAMPIONE = BODYPOSE / "sample_1925299_phase406.jsonl.gz"

#: Dove teniamo gli archivi scaricati da Hugging Face.
ARCHIVI = Path.home() / "Documents/Projects/sport_data/skillcorner-bodypose"

#: Le uniche due partite con pose.
PARTITE = {
    1925299: "Brisbane Roar v Perth Glory (2024-12-21)",
    1996435: "Sydney FC v Adelaide United (2025-02-01)",
}

FPS_POSE, FPS_TRACKING = 25, 10

LEFT_SHOULDER, RIGHT_SHOULDER = "lShoulder", "rShoulder"

#: Larghezza spalle plausibile (metri). Fuori da qui la posa è collassata.
LARGHEZZA_SPALLE_M = (0.15, 0.6)
#: Errore massimo tollerato sul singolo giunto (cm).
ERRORE_MAX_CM = 15.0


# --------------------------------------------------------------------------
# Caricamento
# --------------------------------------------------------------------------

def archivio(match_id: int) -> Path:
    return ARCHIVI / f"{match_id}.jsonl.zip"


def disponibile(match_id: int) -> bool:
    return archivio(match_id).exists()


def scarica(match_id: int, verifica: bool = True) -> Path:
    """Scarica l'archivio da Hugging Face, se non c'è già. ~600 MB."""
    import urllib.request

    dest = archivio(match_id)
    if dest.exists():
        return dest

    manifest = json.loads(MANIFEST.read_text())
    voce = manifest["files"][f"raw/{match_id}.jsonl.zip"]
    url = f"https://huggingface.co/datasets/{manifest['hf_dataset']}/resolve/{manifest['hf_revision']}/raw/{match_id}.jsonl.zip"

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)

    if verifica and not verifica_archivio(match_id):
        raise RuntimeError(f"checksum non corrispondente per {match_id}: cancella e riscarica")
    return dest


def verifica_archivio(match_id: int) -> bool:
    """Confronta lo sha256 dell'archivio con quello dichiarato nel manifest."""
    atteso = json.loads(MANIFEST.read_text())["files"][f"raw/{match_id}.jsonl.zip"]["sha256"]
    h = hashlib.sha256()
    with archivio(match_id).open("rb") as fh:
        for blocco in iter(lambda: fh.read(1 << 20), b""):
            h.update(blocco)
    return h.hexdigest() == atteso


def iter_campione() -> Iterator[dict]:
    """I 309 frame del campione committato: una ripartenza di 12,3 s.

    Serve per sviluppare senza scaricare. **Non** è una partita in miniatura:
    un tempo solo, nessuna sostituzione, nessun tratto senza pose.
    """
    with gzip.open(CAMPIONE, "rt") as fh:
        for riga in fh:
            if riga.strip():
                yield json.loads(riga)


def iter_match(match_id: int, limit: int | None = None) -> Iterator[dict]:
    """Frame di pose di una partita intera, in streaming dallo zip.

    Una partita è ~45 milioni di osservazioni di giunti: va consumata frame per
    frame, mai accumulata in una lista.
    """
    with zipfile.ZipFile(archivio(match_id)) as z:
        nome = next(n for n in z.namelist() if n.endswith(".jsonl"))
        with z.open(nome) as fh:
            for i, riga in enumerate(fh):
                if limit is not None and i >= limit:
                    return
                riga = riga.strip()
                if riga:
                    yield json.loads(riga)


def a_frame_tracking(pose_frame: int) -> int:
    """Riporta un frame di pose (25 fps) sulla griglia del tracking (10 fps).

    Arrotonda per eccesso a metà, come `fold_to_tracking_frame` nell'esempio
    ufficiale: troncare sposterebbe sistematicamente indietro.
    """
    return math.floor(pose_frame * FPS_TRACKING / FPS_POSE + 0.5)


# --------------------------------------------------------------------------
# Orientamento del busto
# --------------------------------------------------------------------------

def orientamento_spalle_deg(lx: float, ly: float, rx: float, ry: float) -> float | None:
    """Direzione sul piano verso cui guardano le spalle, in gradi.

    È l'asse spalla sinistra → destra ruotato di +90°. 0° punta lungo +x,
    antiorario positivo, intervallo (-180, 180].

    Adattato dall'esempio ufficiale `src/features/pose_orientation.py` del repo
    SkillCorner/opendata (MIT).
    """
    # "Avanti" è l'asse spalla sinistra->destra ruotato di +90 gradi.
    # Il segno conta: invertendolo si ottiene un angolo plausibile e sbagliato
    # di 180 gradi, che si scopre solo confrontandolo con la direzione di corsa.
    fx, fy = -(ry - ly), rx - lx
    if math.hypot(fx, fy) < 1e-6:
        return None
    return math.degrees(math.atan2(fy, fx))


def tabella_orientamento(frames, solo_affidabili: bool = True) -> pd.DataFrame:
    """Orientamento delle spalle per ogni giocatore in ogni frame.

    Applica i cinque filtri che l'esempio SkillCorner documenta: posa presente,
    entrambe le spalle, spalle non coincidenti, larghezza anatomicamente
    plausibile, errore del giunto sotto soglia. Ognuno dei cinque, se ignorato,
    produce un numero plausibile e sbagliato.
    """
    righe = []
    for fr in frames:
        for p in fr["player_data"]:
            giunti = p.get("joints")
            if not giunti:
                continue
            ls, rs = giunti.get(LEFT_SHOULDER), giunti.get(RIGHT_SHOULDER)
            if not ls or not rs:
                continue

            lx, ly = ls["xyz"][0], ls["xyz"][1]
            rx, ry = rs["xyz"][0], rs["xyz"][1]
            ang = orientamento_spalle_deg(lx, ly, rx, ry)
            if ang is None:
                continue

            larghezza = math.hypot(rx - lx, ry - ly)
            errore = max(ls["p90_mae_cm"], rs["p90_mae_cm"])
            affidabile = (LARGHEZZA_SPALLE_M[0] <= larghezza <= LARGHEZZA_SPALLE_M[1]
                          and errore <= ERRORE_MAX_CM)
            if solo_affidabili and not affidabile:
                continue

            righe.append({
                "frame": fr["frame"],
                "frame_tracking": a_frame_tracking(fr["frame"]),
                "period": fr["period"],
                "player_id": p["player_id"],
                "x": p.get("x"),
                "y": p.get("y"),
                "orientamento_deg": ang,
                "larghezza_spalle_m": larghezza,
                "errore_cm": errore,
                "affidabile": affidabile,
            })
    return pd.DataFrame(righe)


def copertura(frames) -> pd.DataFrame:
    """Quanti player-frame hanno davvero i giunti, e dove si perdono.

    Una tabella di gate, come quella dell'esempio SkillCorner: senza, è facile
    dichiarare una copertura che non c'è.
    """
    visti = senza_pose = senza_spalle = fuori_larghezza = errore_alto = 0
    for fr in frames:
        for p in fr["player_data"]:
            visti += 1
            giunti = p.get("joints")
            if not giunti:
                senza_pose += 1
                continue
            ls, rs = giunti.get(LEFT_SHOULDER), giunti.get(RIGHT_SHOULDER)
            if not ls or not rs:
                senza_spalle += 1
                continue
            larghezza = math.hypot(rs["xyz"][0] - ls["xyz"][0], rs["xyz"][1] - ls["xyz"][1])
            if not (LARGHEZZA_SPALLE_M[0] <= larghezza <= LARGHEZZA_SPALLE_M[1]):
                fuori_larghezza += 1
                continue
            if max(ls["p90_mae_cm"], rs["p90_mae_cm"]) > ERRORE_MAX_CM:
                errore_alto += 1

    usabili = visti - senza_pose - senza_spalle - fuori_larghezza - errore_alto
    return pd.DataFrame([
        ("player-frame visti", visti, ""),
        ("senza pose (non rilevati)", senza_pose, f"{100*senza_pose/max(visti,1):.1f}%"),
        ("senza una spalla", senza_spalle, f"{100*senza_spalle/max(visti,1):.1f}%"),
        ("larghezza spalle implausibile", fuori_larghezza, f"{100*fuori_larghezza/max(visti,1):.1f}%"),
        (f"errore > {ERRORE_MAX_CM:.0f} cm", errore_alto, f"{100*errore_alto/max(visti,1):.1f}%"),
        ("**usabili**", usabili, f"{100*usabili/max(visti,1):.1f}%"),
    ], columns=["filtro", "n", "quota"])


# --------------------------------------------------------------------------
# Riepilogo di una partita
# --------------------------------------------------------------------------

CACHE = ARCHIVI / "cache"

#: Bande di distanza dalla palla usate per la copertura (metri).
BANDE_M = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 40), (40, 999)]


def _giocatori_in_campo(match_id: int) -> tuple[dict, dict]:
    """Per ogni giocatore: intervallo di frame (scala tracking) e ruolo."""
    from . import data as _data

    meta = _data.load_match_meta(match_id)
    ruoli = {p["id"]: p["player_role"]["acronym"]
             for p in meta["players"] if p.get("player_role")}
    campo = {}
    for p in meta["players"]:
        tot = (p.get("playing_time") or {}).get("total") or {}
        if tot.get("start_frame") is not None and tot.get("minutes_played"):
            campo[p["id"]] = (tot["start_frame"], tot["end_frame"])
    return campo, ruoli


def copertura_partita(match_id: int, forza: bool = False) -> dict:
    """Copertura del pose su una partita intera, aggregata in una passata.

    **Aggrega mentre scorre, senza accumulare righe.** Costruire una tabella da
    3,4 milioni di righe in una lista Python costa più della lettura stessa: la
    prima versione di questa funzione lo faceva e non finiva. Qui si contano e
    basta, e la passata dura circa un minuto.

    Il denominatore sono i **giocatori effettivamente in campo** in quel frame,
    ricavati dai minuti giocati. I 32 record per frame del file includono chi è
    in panchina, e dividere per quelli fa sembrare la copertura peggiore di
    quanto sia (31% invece di 46%).

    Il risultato è piccolo e va in cache come JSON.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{match_id}_copertura.json"
    if not forza and dest.exists():
        return json.loads(dest.read_text())

    campo, ruoli = _giocatori_in_campo(match_id)

    tot = dentro = con_pose = 0
    per_banda = {f"{lo}-{hi}": [0, 0] for lo, hi in BANDE_M}
    per_ruolo: dict[str, list[int]] = {}
    n_frame = 0

    for fr in iter_match(match_id):
        n_frame += 1
        ft = fr["frame"] / 2.5
        b = fr.get("ball_data") or {}
        bx, by = b.get("x"), b.get("y")

        for p in fr["player_data"]:
            tot += 1
            iv = campo.get(p["player_id"])
            if not (iv and iv[0] <= ft <= iv[1]):
                continue
            dentro += 1
            ha = 1 if p.get("joints") else 0
            con_pose += ha

            r = ruoli.get(p["player_id"])
            if r:
                v = per_ruolo.setdefault(r, [0, 0])
                v[0] += ha
                v[1] += 1

            x = p.get("x")
            if bx is not None and x is not None:
                d = math.hypot(x - bx, p["y"] - by)
                for lo, hi in BANDE_M:
                    if lo <= d < hi:
                        v = per_banda[f"{lo}-{hi}"]
                        v[0] += ha
                        v[1] += 1
                        break

    out = {
        "match_id": match_id,
        "frame": n_frame,
        "player_frame_totali": tot,
        "player_frame_in_campo": dentro,
        "con_pose": con_pose,
        "per_banda": per_banda,
        "per_ruolo": per_ruolo,
    }
    dest.write_text(json.dumps(out))
    return out


def quote(conteggi: dict[str, list[int]]) -> pd.Series:
    """Da {chiave: [con_pose, totale]} a percentuali."""
    return pd.Series({k: 100 * a / n for k, (a, n) in conteggi.items() if n})


def errore_per_giunto(match_id: int, limit: int = 20000) -> pd.Series:
    """Errore p90 medio dichiarato, giunto per giunto (cm).

    Su un sottoinsieme di frame: la graduatoria fra giunti è stabilissima e non
    richiede l'intera partita.
    """
    import collections

    acc: dict[str, float] = collections.defaultdict(float)
    n: collections.Counter = collections.Counter()
    for fr in iter_match(match_id, limit=limit):
        for p in fr["player_data"]:
            for nome, v in (p.get("joints") or {}).items():
                acc[nome] += v["p90_mae_cm"]
                n[nome] += 1
    return pd.Series({k: acc[k] / n[k] for k in acc}).sort_values()


def orientamento_vs_moto(match_id: int, limit: int = 60000,
                         v_min: float = 1.0) -> pd.DataFrame:
    """Scarto fra orientamento delle spalle e direzione di corsa.

    La velocità è derivata per differenze finite fra frame consecutivi (25 fps),
    quindi solo le coppie di frame adiacenti contano. Sotto `v_min` m/s la
    direzione di moto è rumore e va scartata.

    Serve come **validazione**: a velocità alta il busto deve allinearsi alla
    corsa. Se non succede, la misura dell'angolo è sbagliata — è così che è
    emerso un segno invertito, che dava una mediana di 160 gradi.
    """
    import numpy as np

    t = tabella_orientamento(iter_match(match_id, limit=limit))
    t = t.sort_values(["player_id", "frame"]).copy()

    g = t.groupby("player_id")
    t["dx"], t["dy"], t["df"] = g.x.diff(), g.y.diff(), g.frame.diff()
    t = t[t.df == 1].copy()

    t["v"] = np.hypot(t.dx, t.dy) * FPS_POSE
    t["dir_moto"] = np.degrees(np.arctan2(t.dy, t.dx))
    t["scarto"] = ((t.orientamento_deg - t.dir_moto + 180) % 360 - 180).abs()
    return t[t.v > v_min]

