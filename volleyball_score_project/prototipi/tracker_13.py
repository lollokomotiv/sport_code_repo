"""PROTOTIPO usa e getta per la issue #13: il tracker della palla regge i nostri video?

Non è codice del progetto: niente test, niente riuso. Legge i `ball.csv` prodotti da
fast-volleyball-tracking-inference in `output/prova-tracker/` e per ogni video stampa:

- quota di fotogrammi con la palla trovata;
- tratti di traiettoria continua (fotogrammi consecutivi con la palla) e loro durata;
- con `--campioni N`, N fotogrammi a caso con la palla trovata e N senza, col punto
  stimato disegnato, in `output/prova-tracker/controllo/` (privati, fuori da git), da
  guardare a mano per stimare quante rilevazioni sono davvero sulla palla.

Uso:
    .venv/bin/python prototipi/tracker_13.py --campioni 6
"""

import argparse
import random
from pathlib import Path

import cv2
import pandas as pd

RADICE = Path(__file__).resolve().parents[1]
USCITA = RADICE / "output" / "prova-tracker"
SORGENTI = [USCITA / "input", RADICE / "data" / "raw" / "2026-10-20_iphone-palestra", RADICE / "data" / "raw" / "2026-10-23_iphone-laterale",
            RADICE / "data" / "external" / "fast-volleyball-tracking-inference" / "examples"]


def video_di(nome: str) -> Path | None:
    for cartella in SORGENTI:
        for estensione in (".mp4", ".MOV", ".mov"):
            p = cartella / f"{nome}{estensione}"
            if p.exists():
                return p
    return None


def tratti(visibile: pd.Series) -> list[int]:
    """Lunghezze dei tratti di fotogrammi consecutivi con la palla."""
    lunghezze, corrente = [], 0
    for v in visibile:
        if v:
            corrente += 1
        elif corrente:
            lunghezze.append(corrente)
            corrente = 0
    if corrente:
        lunghezze.append(corrente)
    return lunghezze


def salva_campione(video: Path, riga: pd.Series, destinazione: Path):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(riga.Frame))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return
    if riga.Visibility:
        cv2.circle(img, (int(riga.X), int(riga.Y)), 18, (0, 255, 255), 3)
    scala = 1280 / img.shape[1]
    img = cv2.resize(img, None, fx=scala, fy=scala)
    cv2.imwrite(str(destinazione), img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campioni", type=int, default=0)
    args = parser.parse_args()
    random.seed(13)

    righe = []
    for csv in sorted(USCITA.glob("*/ball.csv")):
        nome = csv.parent.name
        df = pd.read_csv(csv)
        video = video_di(nome)
        fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) if video else float("nan")
        lunghezze = tratti(df.Visibility == 1)
        righe.append({
            "video": nome,
            "fps": round(fps, 1),
            "fotogrammi": len(df),
            "con_palla": int((df.Visibility == 1).sum()),
            "quota": round((df.Visibility == 1).mean(), 3),
            "tratti": len(lunghezze),
            "tratti_oltre_1s": sum(1 for n in lunghezze if n >= fps),
            "tratto_mediano_s": round(pd.Series(lunghezze).median() / fps, 2) if lunghezze else 0,
        })
        if args.campioni and video:
            cartella = USCITA / "controllo" / nome
            cartella.mkdir(parents=True, exist_ok=True)
            con = df[df.Visibility == 1]
            senza = df[df.Visibility == 0]
            for etichetta, gruppo in (("con", con), ("senza", senza)):
                for _, r in gruppo.sample(min(args.campioni, len(gruppo)), random_state=13).iterrows():
                    salva_campione(video, r, cartella / f"{etichetta}_{int(r.Frame):06d}.jpg")

    print(pd.DataFrame(righe).to_string(index=False))


if __name__ == "__main__":
    main()
