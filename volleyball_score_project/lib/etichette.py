"""Lettura del CSV esportato dalla pagina di etichettatura (`strumenti/etichette.html`).

Una riga per tocco:

    partita, set, scambio, tocco, t, lato, tipo, numero, lato_vincitore, t_fine, lato_nostro

I campi dello scambio (`lato_vincitore`, `t_fine`) e il lato della nostra squadra nel
set (`lato_nostro`) sono ripetuti su ogni riga, così il file basta da solo.

Controllo di un file esportato, con le regole di `lib/` (il riferimento):

    .venv/bin/python -m lib.etichette data/labels/<partita>.csv
"""

import csv
import sys
from pathlib import Path

from lib.partita import punteggio
from lib.scambio import Scambio, Tocco, controlla_scambio


def leggi_etichette(percorso: Path) -> tuple[list[Scambio], dict[int, str]]:
    """Gli scambi in ordine, e per ogni set il lato in cui gioca la nostra squadra."""
    with open(percorso, encoding="utf-8", newline="") as f:
        righe = list(csv.DictReader(f))

    per_scambio: dict[tuple[int, int], list[dict]] = {}
    for r in righe:
        per_scambio.setdefault((int(r["set"]), int(r["scambio"])), []).append(r)

    scambi = []
    for (numero_set, indice), gruppo in sorted(per_scambio.items()):
        # Il vincitore è ripetuto su ogni riga: se due righe non concordano il file
        # è rotto, e scegliere una delle due nasconderebbe l'errore.
        vincitori = {r["lato_vincitore"] for r in gruppo}
        if len(vincitori) != 1:
            raise ValueError(f"set {numero_set}, scambio {indice}: vincitori diversi {sorted(vincitori)}")
        tocchi = tuple(
            Tocco(t=float(r["t"]), lato=r["lato"], tipo=r["tipo"],
                  numero=int(r["numero"]) if r["numero"].strip() else None)
            for r in sorted(gruppo, key=lambda r: float(r["t"]))
        )
        scambi.append(Scambio(set=numero_set, indice=indice, tocchi=tocchi,
                              lato_vincitore=gruppo[0]["lato_vincitore"],
                              t_fine=float(gruppo[0]["t_fine"]) if gruppo[0]["t_fine"].strip() else None))
    lato_nostro = {int(r["set"]): r["lato_nostro"] for r in righe}
    return scambi, lato_nostro


def main():
    scambi, lato_nostro = leggi_etichette(Path(sys.argv[1]))
    risultato = punteggio(scambi, lato_nostro)
    problemi = risultato.problemi + [p for s in scambi for p in controlla_scambio(s)]
    print(f"{len(scambi)} scambi, {sum(len(s.tocchi) for s in scambi)} tocchi")
    for numero, (nostri, loro) in enumerate(risultato.set, start=1):
        print(f"set {numero}: noi {nostri} - {loro} loro")
    print(f"set vinti: noi {risultato.partita[0]} - {risultato.partita[1]} loro")
    for p in sorted(problemi, key=lambda p: (p.set, p.scambio, p.tocco or 0)):
        dove = f"set {p.set}, scambio {p.scambio}" + (f", tocco {p.tocco}" if p.tocco else "")
        print(f"PROBLEMA {dove}: {p.motivo}")
    if not problemi:
        print("nessun problema")


if __name__ == "__main__":
    main()
