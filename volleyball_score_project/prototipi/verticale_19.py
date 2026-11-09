"""PROTOTIPO usa e getta per la issue #19: il tracker regge la diretta verticale?

Taglia 6 minuti dal video verticale (720x1280) in tre versioni, perché il tracker lavora su
immagini 16:9: A così com'è (schiacciata), B ritaglio stretto sul campo, C ritaglio con lo
spazio sopra il campo e bande nere ai lati. MJPEG alta qualità per non comprimere ancora.

Uso:
    .venv/bin/python prototipi/verticale_19.py <video> <secondo_iniziale>
"""

import sys
from pathlib import Path

import cv2
import numpy as np

RADICE = Path(__file__).resolve().parents[1]
USCITA = RADICE / "output" / "prova-verticale" / "input"
USCITA.mkdir(parents=True, exist_ok=True)
video, inizio = sys.argv[1], float(sys.argv[2])
DURATA = 360


def b(f):
    return f[480:885]


def c(f):
    alto = f[300:885]
    banda = np.zeros((alto.shape[0], 160, 3), np.uint8)
    return np.hstack([banda, alto, banda])


versioni = {"vert_a_schiacciato": lambda f: f, "vert_b_stretto": b, "vert_c_alto": c}
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_MSEC, inizio * 1000)
scrittori = {}
for i in range(int(DURATA * fps)):
    ok, f = cap.read()
    if not ok:
        break
    for nome, trasforma in versioni.items():
        g = trasforma(f)
        if nome not in scrittori:
            w = cv2.VideoWriter(str(USCITA / f"{nome}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, (g.shape[1], g.shape[0]))
            w.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)
            scrittori[nome] = w
        scrittori[nome].write(g)
for w in scrittori.values():
    w.release()
print("fatto", {n: (USCITA / f"{n}.avi").stat().st_size // 10**6 for n in versioni})
