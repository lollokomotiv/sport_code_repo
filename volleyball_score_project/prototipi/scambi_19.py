"""PROTOTIPO usa e getta per la issue #19: scambi trovati dal tracker contro scambi etichettati.

Regole fissate prima di guardare i risultati:
- scambio etichettato: dalla battuta (primo tocco) alla fine segnata con Z/X;
- scambio trovato: almeno un tratto con is_rally=True si sovrappone; si riporta la
  sovrapposizione migliore (intersezione / unione) e lo scarto su inizio e fine;
- inventato: tratto con is_rally=True che non tocca nessuno scambio etichettato;
- spezzato: scambio coperto da più tratti; fusi: tratto che copre più scambi;
- periodo confrontato: dalla prima battuta etichettata a FINE_CONFRONTO, perché l'ultimo
  scambio, tagliato dalla fine del video, è stato tolto dalle etichette.

Uso:
    .venv/bin/python prototipi/scambi_19.py
"""

import json
from pathlib import Path

import pandas as pd

RADICE = Path(__file__).resolve().parents[1]
ETICHETTE = RADICE / "data/labels/2026-10-20_iphone-palestra/pallavolo_1agiornata.csv"
TRATTI = RADICE / "output/prova-scambi/pallavolo_1agiornata/tracks"
FPS = 29.97
FINE_CONFRONTO = 670.0


def sovrapposizione(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    unione = max(a[1], b[1]) - min(a[0], b[0])
    return inter, inter / unione if unione else 0.0


def main():
    e = pd.read_csv(ETICHETTE)
    veri = (e.groupby(["set", "scambio"]).agg(inizio=("t", "min"), fine=("t_fine", "first"))
            .reset_index().sort_values("inizio"))
    inizio_confronto = veri.inizio.min()

    tratti = []
    for f in sorted(TRATTI.glob("track_*.json")):
        d = json.loads(f.read_text())
        a, b = d["start_frame"] / FPS, d["last_frame"] / FPS
        tratti.append({"id": d["track_id"], "inizio": a, "fine": b,
                       "scambio": d["rally_classification"]["is_rally"]})
    tratti = pd.DataFrame(tratti)
    nel_periodo = tratti[(tratti.fine >= inizio_confronto - 2) & (tratti.inizio <= FINE_CONFRONTO)]
    proposti = nel_periodo[nel_periodo.scambio]
    print(f"scambi etichettati: {len(veri)} | tratti nel periodo: {len(nel_periodo)}, "
          f"di cui marcati come scambio: {len(proposti)}\n")

    righe = []
    for _, v in veri.iterrows():
        sovr = [(t.id, *sovrapposizione((v.inizio, v.fine), (t.inizio, t.fine)), t.inizio, t.fine)
                for t in proposti.itertuples() if sovrapposizione((v.inizio, v.fine), (t.inizio, t.fine))[0] > 0]
        migliore = max(sovr, key=lambda x: x[2]) if sovr else None
        righe.append({"set": v.set, "scambio": v.scambio, "inizio": round(v.inizio, 1),
                      "durata": round(v.fine - v.inizio, 1), "tratti": len(sovr),
                      "iou": round(migliore[2], 2) if migliore else 0.0,
                      "scarto_inizio": round(migliore[3] - v.inizio, 1) if migliore else None,
                      "scarto_fine": round(migliore[4] - v.fine, 1) if migliore else None})
    r = pd.DataFrame(righe)
    print(r.to_string(index=False))

    trovati = (r.tratti > 0).sum()
    inventati = [t for t in proposti.itertuples()
                 if not any(sovrapposizione((v.inizio, v.fine), (t.inizio, t.fine))[0] > 0 for v in veri.itertuples())]
    fusi = [t for t in proposti.itertuples()
            if sum(sovrapposizione((v.inizio, v.fine), (t.inizio, t.fine))[0] > 0 for v in veri.itertuples()) > 1]
    print(f"\ntrovati: {trovati}/{len(veri)} | spezzati (più tratti): {(r.tratti > 1).sum()} | "
          f"inventati: {len(inventati)} | tratti che fondono più scambi: {len(fusi)}")
    ok = r[r.tratti > 0]
    print(f"sovrapposizione mediana: {ok.iou.median():.2f} | scarto inizio mediano: {ok.scarto_inizio.median():+.1f} s | "
          f"scarto fine mediano: {ok.scarto_fine.median():+.1f} s")
    for t in inventati:
        print(f"inventato: tratto {t.id}, {t.inizio:.1f}-{t.fine:.1f} s")


if __name__ == "__main__":
    main()
