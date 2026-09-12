"""Quante smorzate gioca Alcaraz rispetto al resto del circuito? — vedi README.md.

Una misura sola: smorzate ogni 100 colpi (dopo il servizio), di Alcaraz contro
gli altri giocatori annotati nello stesso periodo, grezza e a parità di
superficie. Il calcolo sta in lib/shots.py, condiviso con zverev/smorzate.

    python3 -m lib.download tml --tour atp
    python3 -m lib.download mcp --gender m --points
    python3 analyses/alcaraz/smorzate/run.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lib import loaders, shots  # noqa: E402

SOGGETTO = "Carlos Alcaraz"
# Sotto questa soglia (circa 15 match) il tasso di un giocatore oscilla troppo
# per metterlo in classifica; le soglie di controllo dicono quanto conta.
MIN_COLPI, SOGLIE_CONTROLLO = 3000, (2000, 5000)
N_BOOT, SEME = 10_000, 20260912


def sezione(titolo: str) -> None:
    print(f"\n{'=' * 78}\n{titolo}\n{'=' * 78}")


def tasso(g: pd.DataFrame) -> float:
    """Smorzate ogni 100 colpi; nessun colpo -> NaN, mai 0."""
    colpi = g.colpi.sum()
    return 100 * g.shots.sum() / colpi if colpi else float("nan")


def copertura() -> pd.DataFrame:
    """Quanti match ufficiali esistono e quanti sono annotati, per superficie."""
    c = loaders.player_coverage(SOGGETTO)
    print(f"Match ufficiali (TennisMyLife): {len(c)}, annotati: {int(c.charted.sum())} "
          f"({100 * c.charted.mean():.0f}%)")
    print(f"Match annotati senza corrispettivo ufficiale (Challenger, esibizioni...): "
          f"{c.attrs['annotati_senza_ufficiale']}")
    t = c.groupby("surface").charted.agg(ufficiali="size", annotati="sum")
    t["quota_%"] = (100 * t.annotati / t.ufficiali).round(0)
    print("\nper superficie:")
    print(t.to_string())
    vinte = 100 * c.groupby("charted").won.mean()
    print(f"\n% di match vinti: annotati {vinte.get(True):.1f}, non annotati {vinte.get(False):.1f}")
    return c


def main() -> None:
    with warnings.catch_warnings():
        # Righe malformate dell'indice e duplicati di ShotTypes: già segnalati da
        # lib/ a ogni altro uso, nessuno riguarda il periodo del soggetto.
        warnings.simplefilter("ignore")

        sezione(f"Copertura di {SOGGETTO}")
        cop = copertura()

        d = shots.load_shot_type("Dr")
        a = d[d.player == SOGGETTO]
        inizio = a.date.min()
        periodo = d[d.date >= inizio]
        circuito = periodo[periodo.player != SOGGETTO]

        sezione("Denominatore: Total include i servizi?")
        v = shots.total_vs_decoded(set(a.match_id))

    print(f"{len(v)} giocatori-match del soggetto e dei suoi avversari, anni 2020")
    print(f"  Total / colpi dopo il servizio:           mediana {np.median(v.total / v.senza):.2f}")
    print(f"  Total / colpi dopo il servizio + servizi: mediana {np.median(v.total / v.con):.2f}")
    assert abs(np.median(v.total / v.senza) - 1) < 0.15, "Total non somiglia ai colpi senza servizio"

    print(f"\n{SOGGETTO}: {len(a)} match annotati, {int(a.colpi.sum())} colpi, "
          f"{int(a.shots.sum())} smorzate")
    print(f"Circuito dal {inizio:%d-%m-%Y}: {circuito.match_id.nunique()} match, "
          f"{circuito.player.nunique()} giocatori, {int(circuito.colpi.sum())} colpi")

    g = shots.rates_by_player(periodo)
    sopra = g[g.colpi >= MIN_COLPI].sort_values("tasso", ascending=False)

    sezione("Plausibilità")
    print("circuito per superficie:",
          circuito.groupby("surface").apply(tasso, include_groups=False).round(2).to_dict())
    print("circuito per anno:      ",
          circuito.groupby(circuito.date.dt.year).apply(tasso, include_groups=False)
          .round(2).to_dict())
    cols = ["match", "colpi", "shots", "tasso", "rapporto"]
    print(f"\nI 12 più alti fra i {len(sopra)} con almeno {MIN_COLPI} colpi:")
    print(sopra[cols].head(12).round(2).to_string())
    print("\nI 5 più bassi:")
    print(sopra[cols].tail(5).iloc[::-1].round(2).to_string())
    i = list(sopra.index).index(SOGGETTO)
    print(f"\nAttorno a {SOGGETTO}:")
    print(sopra[cols].iloc[max(0, i - 3):i + 4].round(2).to_string())

    sezione("Tabella")
    per_colpo = circuito.groupby("surface").shots.sum() / circuito.groupby("surface").colpi.sum()
    ic_t, ic_r = shots.bootstrap_player(a, per_colpo, N_BOOT, SEME)
    ta, tc = tasso(a), tasso(circuito)
    pos_t, n, med_t = shots.rank_of(g, SOGGETTO, "tasso", MIN_COLPI)
    pos_r, _, med_r = shots.rank_of(g, SOGGETTO, "rapporto", MIN_COLPI)

    print(f"{SOGGETTO:31s}{ta:5.2f} ogni 100 colpi  [IC 95% {ic_t[0]:.2f} – {ic_t[1]:.2f}]")
    print(f"resto del circuito (aggregato) {tc:5.2f}   -> {ta / tc:.2f} volte")
    print(f"giocatore mediano (n = {n})    {med_t:5.2f}   -> {ta / med_t:.2f} volte")
    print(f"posizione                      {pos_t}ª su {n}")
    print(f"\na parità di superficie: rapporto osservate/attese {g.at[SOGGETTO, 'rapporto']:.2f} "
          f"[IC 95% {ic_r[0]:.2f} – {ic_r[1]:.2f}], {pos_r}ª su {n} (mediana {med_r:.2f})")

    print("\nper superficie:")
    for sup in ("Clay", "Grass", "Hard"):
        sa, sc = a[a.surface == sup], circuito[circuito.surface == sup]
        print(f"  {sup:6s} {SOGGETTO} {tasso(sa):5.2f} ({int(sa.colpi.sum()):6d} colpi)  "
              f"circuito {tasso(sc):5.2f}  -> {tasso(sa) / tasso(sc):.2f} volte")

    # Il tasso del circuito cambia negli anni: il controllo per superficie da
    # solo non basta se il soggetto è annotato più in certe stagioni che in altre.
    celle = periodo.assign(cella=periodo.date.dt.year.astype(str) + "-" + periodo.surface)
    ga = shots.rates_by_player(celle, by="cella")
    pa, na, ma = shots.rank_of(ga, SOGGETTO, "rapporto", MIN_COLPI)
    print(f"\ncontrollo anno × superficie: rapporto {ga.at[SOGGETTO, 'rapporto']:.2f}, "
          f"{pa}ª su {na} (mediana {ma:.2f})")

    # Il campione annotato può pendere verso le vittorie o le sconfitte (vedi la
    # copertura): conta solo se il tasso cambia con l'esito.
    esito = cop[cop.charted].set_index("mcp_match_id").won
    ae = a[a.match_id.isin(esito.index)]
    for vinto, sub in ae.groupby(ae.match_id.map(esito)):
        print(f"controllo esito: match {'vinti' if vinto else 'persi'} {len(sub):3d}, "
              f"{tasso(sub):.2f} ogni 100 colpi")

    print("\ncontrollo della soglia di colpi:")
    for s in SOGLIE_CONTROLLO:
        pt, nn, mt = shots.rank_of(g, SOGGETTO, "tasso", s)
        pr, _, _ = shots.rank_of(g, SOGGETTO, "rapporto", s)
        print(f"  almeno {s} colpi: {nn} giocatori, grezzo {pt}ª (mediana {mt:.2f}), "
              f"a parità di superficie {pr}ª")


if __name__ == "__main__":
    main()
