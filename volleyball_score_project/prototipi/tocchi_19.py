"""PROTOTIPO usa e getta per la issue #19: tocchi dalla sola traiettoria della palla.

Un tocco è una rottura della traiettoria: la velocità cambia più di quanto la gravità
spieghi. La logica è quella di `player_contacts.py` di fast-volleyball-tracking-inference
(MIT), senza le giocatrici, che il loro modello sull'indoor non trova. In più si cercano
tocchi nei buchi brevi della traccia, perché al tocco la palla spesso sparisce (#13).

Regole fissate prima di guardare i risultati:
- tocchi etichettati: tutti i 151 della 1ª giornata;
- rotture considerate: solo dentro gli scambi etichettati, da 1 s prima della battuta
  alla fine segnata con Z/X; fuori dagli scambi è compito della segmentazione;
- trovato: una rottura entro TOLLERANZA dal tocco, abbinamento uno a uno dal più vicino;
  si riporta anche ±0,5 s e lo scarto mediano, perché il tasto arriva in ritardo;
- inventata: rottura dentro uno scambio non abbinata a nessun tocco;
- lato: punto della palla proiettato sul pavimento con la calibrazione di #19, metà
  campo vicina se sotto i 9 m. La palla in aria finisce proiettata più lontano: il
  lato così è una stima di parte, e si misura quanto.

Uso:
    .venv/bin/python prototipi/tocchi_19.py
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RADICE = Path(__file__).resolve().parents[1]
ETICHETTE = RADICE / "data/labels/2026-10-20_iphone-palestra/pallavolo_1agiornata.csv"
PALLA = RADICE / "output/prova-tracker/pallavolo_1agiornata/ball.csv"
CAMPO = RADICE / "output/prova-campo/pallavolo_1agiornata_court.json"
USCITA = RADICE / "output/prova-tocchi/rotture.csv"
FPS = 29.97
TOLLERANZA = 0.3

# Come in player_contacts.py.
FINESTRA = 5
BUCO_MAX = 4
RESIDUO_MIN = 6.0
RESIDUO_RELATIVO = 0.30
VELOCITA_MIN = 1.0
SOPPRESSIONE_S = 0.30
# Nostro: buchi fino a mezzo secondo possono nascondere un tocco.
BUCO_TOCCO_MAX = 15


def gravita_px(campo):
    # Scala verticale alla rete: rete femminile 2,24 m tra il piede del palo e il bordo alto.
    k = {p["name"]: p for p in campo["keypoints"]}
    px = ((k["5_center_left"]["y"] - k["7_net_left"]["y"]) + (k["6_center_right"]["y"] - k["8_net_right"]["y"])) / 2
    return 981.0 * (px / 224.0) / FPS ** 2


def omografia(campo):
    k = {p["name"]: (p["x"], p["y"]) for p in campo["keypoints"]}
    immagine = np.float32([k["1_back_left"], k["2_back_left"], k["3_back_right"], k["4_back_right"]])
    metri = np.float32([[0, 0], [18, 0], [18, 9], [0, 9]])
    return cv2.getPerspectiveTransform(immagine, metri)


def velocita(f, x, y, i, verso):
    """Velocità media su FINESTRA campioni da i verso avanti o indietro, senza buchi lunghi."""
    sel = [i]
    j = i
    while len(sel) < FINESTRA:
        k = j + verso
        if k < 0 or k >= len(f) or abs(f[k] - f[j]) > BUCO_MAX:
            break
        sel.append(k)
        j = k
    if len(sel) < 3 or abs(f[sel[-1]] - f[sel[0]]) < 1:
        return None
    ff = f[sel]
    return np.polyfit(ff, x[sel], 1)[0], np.polyfit(ff, y[sel], 1)[0], ff.mean()


def rottura(prima, dopo, g):
    (vx0, vy0, c0), (vx1, vy1, c1) = prima, dopo
    v = max(math.hypot(vx0, vy0), math.hypot(vx1, vy1))
    if v < VELOCITA_MIN:
        return None
    residuo = math.hypot(vx1 - vx0, vy1 - vy0 - g * max(c1 - c0, 0))
    soglia = max(RESIDUO_MIN, RESIDUO_RELATIVO * v)
    return residuo if residuo >= soglia else None


def rotture(palla, g):
    f, x, y = palla.Frame.to_numpy(), palla.X.to_numpy(float), palla.Y.to_numpy(float)
    candidati = []
    for i in range(len(f)):
        prima, dopo = velocita(f, x, y, i, -1), velocita(f, x, y, i, +1)
        if prima and dopo and (r := rottura(prima, dopo, g)):
            candidati.append({"frame": f[i], "x": x[i], "y": y[i], "forza": r, "origine": "traccia"})
        # Buco dopo il campione i: si confronta la velocità ai due lati.
        if i + 1 < len(f) and BUCO_MAX < f[i + 1] - f[i] <= BUCO_TOCCO_MAX:
            prima, dopo = velocita(f, x, y, i, -1), velocita(f, x, y, i + 1, +1)
            if prima and dopo and (r := rottura(prima, dopo, g)):
                candidati.append({"frame": (f[i] + f[i + 1]) / 2, "x": (x[i] + x[i + 1]) / 2,
                                  "y": (y[i] + y[i + 1]) / 2, "forza": r, "origine": "buco"})
    finestra = SOPPRESSIONE_S * FPS
    scelti = []
    for c in sorted(candidati, key=lambda c: -c["forza"]):
        if all(abs(c["frame"] - s["frame"]) > finestra for s in scelti):
            scelti.append(c)
    r = pd.DataFrame(sorted(scelti, key=lambda c: c["frame"]))
    r["t"] = r.frame / FPS
    return r


def abbina(veri, trovati, tolleranza):
    coppie = sorted((abs(v.t - t.t), v.Index, t.Index) for v in veri.itertuples() for t in trovati.itertuples()
                    if abs(v.t - t.t) <= tolleranza)
    usati_v, usati_t, abbinati = set(), set(), {}
    for _, iv, it in coppie:
        if iv not in usati_v and it not in usati_t:
            usati_v.add(iv); usati_t.add(it); abbinati[iv] = it
    return abbinati


def main():
    campo = json.loads(CAMPO.read_text())
    g = gravita_px(campo)
    palla = pd.read_csv(PALLA)
    palla = palla[palla.Visibility == 1].reset_index(drop=True)
    r = rotture(palla, g)

    e = pd.read_csv(ETICHETTE)
    scambi = e.groupby(["set", "scambio"]).agg(inizio=("t", "min"), fine=("t_fine", "first"))
    dentro = np.zeros(len(r), bool)
    for s in scambi.itertuples():
        dentro |= (r.t >= s.inizio - 1.0) & (r.t <= s.fine)
    r = r[dentro].reset_index(drop=True)

    H = omografia(campo)
    suolo = cv2.perspectiveTransform(r[["x", "y"]].to_numpy(np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
    r["metri"] = suolo[:, 0]
    r["lato"] = np.where(r.metri < 9, "vicino", "lontano")

    print(f"gravità {g:.2f} px/fotogramma² | campioni di palla {len(palla)} | "
          f"tocchi etichettati {len(e)} | rotture dentro gli scambi {len(r)} "
          f"({(r.origine == 'traccia').sum()} dalla traccia, {(r.origine == 'buco').sum()} dai buchi)\n")

    for tol in (TOLLERANZA, 0.5):
        ab = abbina(e, r, tol)
        print(f"tolleranza ±{tol} s: trovati {len(ab)}/{len(e)} | inventate {len(r) - len(ab)}/{len(r)}")
    ab = abbina(e, r, TOLLERANZA)
    e["trovato"] = e.index.isin(list(ab))
    scarti = pd.Series({iv: r.t[it] - e.t[iv] for iv, it in ab.items()})
    print(f"scarto rottura - etichetta (±{TOLLERANZA} s): mediana {scarti.median():+.2f} s, "
          f"quartili {scarti.quantile(.25):+.2f} / {scarti.quantile(.75):+.2f}\n")

    print("trovati per tipo:")
    print(e.groupby("tipo").trovato.agg(lambda s: f"{s.sum()}/{len(s)}").reindex(list("SREABDF")).dropna().to_string())
    lati = pd.DataFrame([(e.lato[iv], r.lato[it], r.origine[it]) for iv, it in ab.items()],
                        columns=["vero", "stimato", "origine"])
    print(f"\nlato giusto sui trovati: {(lati.vero == lati.stimato).sum()}/{len(lati)}")
    print(pd.crosstab(lati.vero, lati.stimato).to_string())
    per_origine = pd.Series([r.origine[it] for it in ab.values()]).value_counts()
    print(f"\ntrovati per origine: {per_origine.to_dict()} | rotture per origine: {r.origine.value_counts().to_dict()}")

    r["abbinata"] = r.index.isin(list(ab.values()))
    USCITA.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(USCITA, index=False)


if __name__ == "__main__":
    main()
