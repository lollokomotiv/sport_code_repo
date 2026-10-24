"""PROTOTIPO usa e getta per la issue #13: quanta palla vede il tracker sui tocchi etichettati.

Confronta le etichette fatte a mano (data/labels/...) con il ball.csv del tracker sullo
stesso video. Per ogni tocco: il tracker vede la palla in almeno un fotogramma della
finestra attorno all'istante etichettato? È un limite superiore del richiamo, perché una
rilevazione nella finestra non è detto che sia sulla palla (precisione stimata a vista
circa 4 su 5).

Uso:
    .venv/bin/python prototipi/richiamo_13.py
"""

from pathlib import Path

import pandas as pd

RADICE = Path(__file__).resolve().parents[1]
ETICHETTE = RADICE / "data/labels/2026-10-20_iphone-palestra/pallavolo_1agiornata.csv"
TRACKER = RADICE / "output/prova-tracker/pallavolo_1agiornata/ball.csv"
FPS = 29.97

# Il tasto si preme dopo aver visto il tocco: la finestra guarda più indietro che avanti.
FINESTRE = {"da -0,5 a +0,1 s": (-0.5, 0.1), "±0,2 s": (-0.2, 0.2)}


def main():
    etichette = pd.read_csv(ETICHETTE)
    palla = pd.read_csv(TRACKER)
    visibile = palla.set_index("Frame").Visibility.eq(1)
    n_fotogrammi = len(palla)

    def vista_in(t, prima, dopo):
        a = max(0, int((t + prima) * FPS))
        b = min(n_fotogrammi - 1, int((t + dopo) * FPS) + 1)
        return bool(visibile.iloc[a:b + 1].any())

    print(f"{len(etichette)} tocchi etichettati, {etichette.groupby(['set', 'scambio']).ngroups} scambi\n")
    for nome, (prima, dopo) in FINESTRE.items():
        etichette[nome] = [vista_in(t, prima, dopo) for t in etichette.t]
        tot = etichette[nome].sum()
        print(f"finestra {nome}: palla vista su {tot}/{len(etichette)} tocchi ({tot / len(etichette):.0%})")
        for lato, g in etichette.groupby("lato"):
            print(f"   {lato:8s} {g[nome].sum()}/{len(g)}")
        per_tipo = etichette.groupby("tipo")[nome].agg(["sum", "size"])
        print("   per tipo:", ", ".join(f"{t} {r['sum']}/{r['size']}" for t, r in per_tipo.iterrows()))
        print()

    # Fotogrammi con palla dentro gli scambi (dal primo tocco alla fine) e nelle pause.
    in_gioco = pd.Series(False, index=range(n_fotogrammi))
    for _, g in etichette.groupby(["set", "scambio"]):
        a, b = int(g.t.min() * FPS), int(g.t_fine.iloc[0] * FPS)
        in_gioco.iloc[a:b + 1] = True
    inizio, fine = int(etichette.t.min() * FPS), int(etichette.t_fine.max() * FPS)
    dentro = in_gioco.iloc[inizio:fine + 1]
    vis = visibile.reindex(range(n_fotogrammi), fill_value=False).iloc[inizio:fine + 1]
    print(f"fotogrammi con palla durante gli scambi: {vis[dentro].mean():.0%} ({dentro.sum()} fotogrammi)")
    print(f"fotogrammi con palla nelle pause:        {vis[~dentro].mean():.0%} ({(~dentro).sum()} fotogrammi)")


if __name__ == "__main__":
    main()
