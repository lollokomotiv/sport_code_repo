"""MultiSports, parte pallavolo: annotazioni e clip scaricate da Hugging Face.

Il dataset (MCG-NJU/SportsAction, licenza CC BY-NC 4.0) è ad accesso riservato:
serve un token di sola lettura in `~/.cache/huggingface/token` o nella variabile
`HF_TOKEN`, con i termini accettati sulla pagina del dataset.

Si scarica solo ciò che serve. Le annotazioni sono un file da 14 MB. Le clip stanno
in un archivio tar da 6,3 GB con un mp4 per clip: Hugging Face risponde a richieste
per intervalli di byte, quindi si legge l'indice del tar (512 byte per elemento) e
poi si scaricano solo le clip scelte.

Struttura delle annotazioni, dalla scheda del dataset:

- `labels`: nomi delle classi, indice da 0;
- `train_videos`, `test_videos`: `[[clip, ...]]`; `test_videos` è la validation;
- `nframes`, `resolution`: per clip, numero di fotogrammi e `(altezza, larghezza)`;
- `gttubes`: per clip, `{indice classe: [traccia, ...]}`. Una traccia è un'azione
  di una giocatrice: un array con una riga per fotogramma, `fotogramma x1 y1 x2 y2`.
  I fotogrammi partono da 1, nell'ordine in cui OpenCV legge l'mp4.

Il genere dei video non è nel dataset: sta in `docs/multisports-video.csv`, ricavato
dai titoli su YouTube il 2026-10-19.

Uso:
    .venv/bin/python -m lib.multisports --femminili 10
"""

import argparse
import csv
import json
import os
import pickle
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

RADICE = Path(__file__).resolve().parents[1]
CARTELLA = RADICE / "data" / "external" / "multisports"
GENERE_VIDEO = RADICE / "docs" / "multisports-video.csv"

URL_BASE = "https://huggingface.co/datasets/MCG-NJU/SportsAction/resolve/main/data/trainval/"
FILE_ANNOTAZIONI = "multisports_GT.pkl"
FILE_ARCHIVIO = "volleyball.tar"
BLOCCO_TAR = 512
# Le richieste lunghe su Hugging Face cadono a metà: le clip si scaricano a pezzi.
PEZZO = 4 * 1024 * 1024
TENTATIVI = 4


def _token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    return Path("~/.cache/huggingface/token").expanduser().read_text().strip()


def _leggi(url: str, inizio: int | None = None, quanti: int | None = None, autenticato: bool = True):
    """Scarica un file o un intervallo di byte. Restituisce (contenuto, url finale)."""
    intestazioni = {"Authorization": f"Bearer {_token()}"} if autenticato else {}
    if inizio is not None:
        intestazioni["Range"] = f"bytes={inizio}-{inizio + quanti - 1}"
    richiesta = urllib.request.Request(url, headers=intestazioni)
    with urllib.request.urlopen(richiesta, timeout=60) as risposta:
        return risposta.read(), risposta.geturl()


class _SoloNumpy(pickle.Unpickler):
    """Legge il pickle accettando solo gli array di numpy.

    Un pickle può eseguire codice arbitrario quando si carica: qui si rifiuta
    qualunque oggetto che non serva a ricostruire dizionari, liste e array.
    """

    CONSENTITI = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def find_class(self, module, name):
        if (module, name) in self.CONSENTITI:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"oggetto non consentito nel pickle: {module}.{name}")


def scarica_annotazioni() -> Path:
    percorso = CARTELLA / FILE_ANNOTAZIONI
    if not percorso.exists():
        CARTELLA.mkdir(parents=True, exist_ok=True)
        contenuto, _ = _leggi(URL_BASE + FILE_ANNOTAZIONI)
        percorso.write_bytes(contenuto)
    return percorso


def carica_annotazioni(percorso: Path | None = None) -> dict:
    percorso = percorso or scarica_annotazioni()
    with open(percorso, "rb") as f:
        return _SoloNumpy(f, encoding="latin1").load()


def genere_video() -> dict[str, str]:
    with open(GENERE_VIDEO, encoding="utf-8") as f:
        return {r["video"]: r["genere"] for r in csv.DictReader(f)}


def video_della_clip(clip: str) -> str:
    """'volleyball/v_Y0SCvur8ZK0_c011' -> 'Y0SCvur8ZK0' (ID YouTube)."""
    return clip.split("/v_", 1)[1].rsplit("_c", 1)[0]


def clip_pallavolo(gt: dict) -> list[dict]:
    """Una riga per clip di pallavolo, con parte, genere e numero di azioni per classe."""
    genere = genere_video()
    righe = []
    for parte, chiave in (("train", "train_videos"), ("validation", "test_videos")):
        for clip in gt[chiave][0]:
            if not clip.startswith("volleyball/"):
                continue
            tracce = gt["gttubes"].get(clip, {})
            righe.append({
                "clip": clip,
                "video": video_della_clip(clip),
                "parte": parte,
                "genere": genere.get(video_della_clip(clip), "sconosciuto"),
                "fotogrammi": int(gt["nframes"][clip]),
                "azioni": sum(len(t) for t in tracce.values()),
                **{gt["labels"][k].removeprefix("volleyball "): len(t) for k, t in tracce.items()},
            })
    return righe


def scegli_clip(gt: dict, genere: str, quante: int) -> list[str]:
    """Una clip per video, la più ricca di azioni del train, dai primi video per ID.

    Una clip per video invece delle più ricche in assoluto: così le prime clip
    coprono partite, palazzetti e riprese diverse.
    """
    migliore: dict[str, dict] = {}
    for r in clip_pallavolo(gt):
        if r["genere"] != genere or r["parte"] != "train":
            continue
        if r["video"] not in migliore or r["azioni"] > migliore[r["video"]]["azioni"]:
            migliore[r["video"]] = r
    return [migliore[v]["clip"] for v in sorted(migliore)][:quante]


def indice_archivio() -> dict[str, list[int]]:
    """{nome nel tar: [inizio dei dati, dimensione]}, leggendo solo le intestazioni.

    Si salva accanto alle clip: leggerlo costa una richiesta per elemento.
    """
    percorso = CARTELLA / "indice_archivio.json"
    if percorso.exists():
        return json.loads(percorso.read_text())
    # Hugging Face rimanda a un indirizzo firmato: lo si risolve una volta sola.
    _, url = _leggi(URL_BASE + FILE_ARCHIVIO, 0, BLOCCO_TAR)
    indice = {}
    posizione = 0
    while True:
        blocco, _ = _leggi(url, posizione, BLOCCO_TAR, autenticato=False)
        if blocco.strip(b"\0") == b"":
            break
        info = tarfile.TarInfo.frombuf(blocco, "utf-8", "surrogateescape")
        if info.isfile():
            indice[info.name] = [posizione + BLOCCO_TAR, info.size]
        posizione += BLOCCO_TAR + -(-info.size // BLOCCO_TAR) * BLOCCO_TAR
    CARTELLA.mkdir(parents=True, exist_ok=True)
    percorso.write_text(json.dumps(indice))
    return indice


def percorso_clip(clip: str) -> Path:
    return CARTELLA / f"{clip}.mp4"


def _scarica_intervallo(inizio: int, dimensione: int) -> bytes:
    """Un intervallo dell'archivio, a pezzi, riprovando i pezzi che falliscono."""
    _, url = _leggi(URL_BASE + FILE_ARCHIVIO, 0, BLOCCO_TAR)
    parti = []
    letti = 0
    while letti < dimensione:
        quanti = min(PEZZO, dimensione - letti)
        for tentativo in range(1, TENTATIVI + 1):
            try:
                pezzo, _ = _leggi(url, inizio + letti, quanti, autenticato=False)
                if len(pezzo) != quanti:
                    raise OSError(f"ricevuti {len(pezzo)} byte invece di {quanti}")
                break
            except OSError:
                if tentativo == TENTATIVI:
                    raise
                # L'indirizzo firmato può essere scaduto: se ne chiede uno nuovo.
                _, url = _leggi(URL_BASE + FILE_ARCHIVIO, 0, BLOCCO_TAR)
        parti.append(pezzo)
        letti += quanti
    return b"".join(parti)


def scarica_clip(clip_scelte: list[str]) -> list[Path]:
    indice = indice_archivio()
    percorsi = []
    for clip in clip_scelte:
        destinazione = percorso_clip(clip)
        if not destinazione.exists():
            inizio, dimensione = indice[f"{clip}.mp4"]
            contenuto = _scarica_intervallo(inizio, dimensione)
            destinazione.parent.mkdir(parents=True, exist_ok=True)
            # Si scrive accanto e poi si rinomina: una clip a metà non resta come buona.
            parziale = destinazione.with_suffix(".parziale")
            parziale.write_bytes(contenuto)
            parziale.rename(destinazione)
        percorsi.append(destinazione)
    return percorsi


def fotogramma(clip: str, numero: int) -> np.ndarray:
    """Il fotogramma `numero` (da 1, come nelle annotazioni), in RGB."""
    video = cv2.VideoCapture(str(percorso_clip(clip)))
    video.set(cv2.CAP_PROP_POS_FRAMES, numero - 1)
    ok, immagine = video.read()
    video.release()
    if not ok:
        raise ValueError(f"fotogramma {numero} non leggibile in {clip}")
    return cv2.cvtColor(immagine, cv2.COLOR_BGR2RGB)


def azioni_nel_fotogramma(gt: dict, clip: str, numero: int) -> list[tuple[str, np.ndarray]]:
    """Le azioni in corso nel fotogramma: (classe, riquadro x1 y1 x2 y2)."""
    attive = []
    for k, tracce in gt["gttubes"].get(clip, {}).items():
        for traccia in tracce:
            riga = traccia[traccia[:, 0] == numero]
            if len(riga):
                attive.append((gt["labels"][k].removeprefix("volleyball "), riga[0, 1:5]))
    return attive


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--femminili", type=int, default=10, help="quante clip femminili scaricare")
    argomenti = parser.parse_args()

    gt = carica_annotazioni()
    scelte = scegli_clip(gt, "femminile", argomenti.femminili)
    for percorso in scarica_clip(scelte):
        print(f"{percorso.relative_to(RADICE)}  {percorso.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
