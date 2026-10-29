"""PROTOTIPO usa e getta per la issue #19: calibrazione del campo dal colore.

Il campo della palestra è arancione su pavimento blu: si separa l'arancione, si prende il
contorno più grande e lo si riduce a quattro angoli. La linea centrale si ricava con la
prospettiva (campo 18 x 9 m, rete a metà). Scrive il formato di
fast-volleyball-tracking-inference (8 punti) e un fotogramma di controllo.

Uso:
    .venv/bin/python prototipi/campo_19.py <video> <secondo> <uscita.json>
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

video, secondo, uscita = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
cap = cv2.VideoCapture(str(video))
cap.set(cv2.CAP_PROP_POS_MSEC, secondo * 1000)
ok, img = cap.read()
h, w = img.shape[:2]

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Campo: tinta 14-15 e saturazione ~255; la parete color sabbia ha tinta 21 e saturazione ~110.
arancio = cv2.inRange(hsv, (8, 200, 150), (19, 255, 255))
# Le giocatrici bucano il campo: si chiude la maschera prima di cercare il contorno.
arancio = cv2.morphologyEx(arancio, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
contorni, _ = cv2.findContours(arancio, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
campo = max(contorni, key=cv2.contourArea)
angoli = cv2.approxPolyDP(cv2.convexHull(campo), 0.02 * cv2.arcLength(campo, True), True).reshape(-1, 2)
if len(angoli) != 4:
    sys.exit(f"trovati {len(angoli)} angoli invece di 4: serve un altro fotogramma")

# Ordine del formato: vicino-sinistra, lontano-sinistra, lontano-destra, vicino-destra.
per_y = angoli[np.argsort(angoli[:, 1])]
lontani = per_y[:2][np.argsort(per_y[:2, 0])]
vicini = per_y[2:][np.argsort(per_y[2:, 0])]
p1, p2, p3, p4 = vicini[0], lontani[0], lontani[1], vicini[1]

# Prospettiva dal campo (x lungo i 18 m, da vicino a lontano; y lungo i 9 m) all'immagine.
campo_m = np.float32([[0, 0], [18, 0], [18, 9], [0, 9]])
H = cv2.getPerspectiveTransform(campo_m, np.float32([p1, p2, p3, p4]))
centro = cv2.perspectiveTransform(np.float32([[[9, 0]], [[9, 9]]]), H).reshape(-1, 2)

punti = {"1_back_left": p1, "2_back_left": p2, "3_back_right": p3, "4_back_right": p4,
         "5_center_left": centro[0], "6_center_right": centro[1]}
dati = {"video_path": str(video), "frame_index": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
        "frame_width": w, "frame_height": h,
        "keypoints": [{"id": i, "name": n, "x": int(round(p[0])), "y": int(round(p[1])), "visible": True}
                      for i, (n, p) in enumerate(punti.items())]}
uscita.parent.mkdir(parents=True, exist_ok=True)
uscita.write_text(json.dumps(dati, indent=2))

controllo = img.copy()
cv2.polylines(controllo, [np.int32([p1, p2, p3, p4])], True, (0, 255, 0), 3)
cv2.line(controllo, tuple(np.int32(centro[0])), tuple(np.int32(centro[1])), (255, 0, 255), 3)
for n, p in punti.items():
    cv2.circle(controllo, tuple(np.int32(p)), 9, (0, 0, 255), -1)
    cv2.putText(controllo, n[0], tuple(np.int32(p) + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
cv2.imwrite(str(uscita.with_suffix(".jpg")), cv2.resize(controllo, (1280, int(1280 * h / w))))
print(json.dumps({n: [int(p[0]), int(p[1])] for n, p in punti.items()}))
