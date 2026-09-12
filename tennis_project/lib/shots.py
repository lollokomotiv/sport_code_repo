"""Tipi di colpo del Match Charting Project, con il contesto per interpretarli.

Il file `ShotTypes` da solo non basta a dire nulla: sapere che un giocatore ha
tirato 48 smorzate non significa niente senza sapere quanti colpi ha giocato in
totale, su quale superficie, e quanti punti vinceva comunque in quei match.
Queste funzioni mettono insieme le tre cose.

**I codici di `ShotTypes` sono gerarchici, non una partizione**: `Base` (94,6%)
e `Net` (5,4%) dividono il totale, ma `Gs`, `Sl`, `Vo`, `Dr`, `Lo` sono
sottoinsiemi annidati. Sommare righe diverse conta gli stessi colpi più volte.
Qui si estrae **una** riga per volta e la si rapporta sempre a `Total`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import loaders, points
from .paths import RAW_DIR

# Colonne di esito di ShotTypes che hanno senso per un singolo codice di colpo.
_ESITI = ["shots", "winners", "induced_forced", "unforced",
          "shots_in_pts_won", "shots_in_pts_lost"]


def _load_shot_types(gender: str) -> pd.DataFrame:
    """Il file ShotTypes, con una riga sola per match, giocatore e livello.

    13 match (tutti anteriori al 2000) compaiono due volte con conteggi diversi,
    due annotazioni dello stesso incontro: senza scartarne una, ogni join
    moltiplica le righe. Si tiene la prima e lo si dice.
    """
    st = pd.read_csv(RAW_DIR / "mcp" / f"charting-{gender}-stats-ShotTypes.csv",
                     low_memory=False)
    chiave = ["match_id", "player", "row"]
    doppi = st.duplicated(chiave)
    if doppi.any():
        warnings.warn(
            f"ShotTypes ({gender}): scartate {int(doppi.sum())} righe duplicate in "
            f"{st.loc[doppi, 'match_id'].nunique()} match (annotazioni doppie a monte)",
            stacklevel=3,
        )
    return st[~doppi]


def load_shot_type(code: str = "Dr", gender: str = "m") -> pd.DataFrame:
    """Una riga per match e giocatore, per un solo codice di colpo.

    `code` è un livello della colonna `row` di `ShotTypes` (`Dr` smorzate,
    `Vo` volée, `Sl` slice, `Lo` pallonetti, `Net` colpi a rete...). Al conteggio
    del colpo si affiancano:

    - `colpi`: i colpi totali del giocatore in quel match (riga `Total`), che è
      il denominatore giusto per un tasso d'uso;
    - `punti_match` e `punti_vinti` dall'Overview, che servono da riferimento:
      senza di loro la resa del colpo non è interpretabile, perché una
      percentuale alta può voler dire solo che l'avversario era debole;
    - `surface`, `date`, `tournament` e `avversario` dall'indice dei match.

    I match senza superficie nell'indice vengono scartati: sono pochi e senza
    quella colonna non si può controllare la confondente principale.

    **Un giocatore che in un match non ha mai giocato il colpo compare con 0**,
    non sparisce. Il file a monte non scrive le righe a zero (nessuna riga `Dr`
    vale 0), quindi l'assenza della riga *è* lo zero: dal 2021 riguarda il 14%
    dei giocatori-match. Scartarli gonfierebbe il tasso proprio dei giocatori
    che il colpo lo usano poco.
    """
    st = _load_shot_types(gender)

    colpo = st[st.row == code].set_index(["match_id", "player"])[_ESITI]
    totale = (st[st.row == "Total"].set_index(["match_id", "player"])[["shots"]]
              .rename(columns={"shots": "colpi"}))

    ov = loaders.load_mcp_stats("Overview", gender).set_index(["match_id", "player"])
    ov["punti_match"] = ov.serve_pts + ov.return_pts
    ov["punti_vinti"] = ov.first_won + ov.second_won + ov.return_pts_won

    idx = loaders.load_mcp_matches(gender)
    d = totale.join(colpo, how="left")
    d[_ESITI] = d[_ESITI].fillna(0).astype(int)
    d = (d.join(ov[["punti_match", "punti_vinti"]], how="inner")
              .reset_index()
              # due righe per match (una per giocatore) contro una nell'indice:
              # validate= è ciò che ha fatto emergere i match_id duplicati
              .merge(idx[["match_id", "date", "surface", "player_1", "player_2",
                          "tournament", "charted_by"]],
                     on="match_id", how="left", validate="many_to_one"))
    d["avversario"] = d.player_1.where(d.player != d.player_1, d.player_2)
    return d.dropna(subset=["surface"])


def reference_rate(d: pd.DataFrame, player: str, exclude_opponent: str | None = None,
                   by: str = "surface") -> pd.Series:
    """Tasso d'uso del colpo del soggetto, livello per livello della confondente.

    `exclude_opponent` va **sempre** valorizzato con l'avversario che si sta
    giudicando: se il metro di paragone contiene i match sotto esame, il
    confronto è circolare e l'effetto si autoannulla in proporzione a quanto
    quell'avversario pesa nel campione.
    """
    a = d[d.player == player]
    if exclude_opponent is not None:
        a = a[a.avversario != exclude_opponent]
    return a.groupby(by).apply(lambda g: g.shots.sum() / g.colpi.sum(),
                               include_groups=False)


def observed_vs_expected(d: pd.DataFrame, player: str, exclude_opponent: str | None = None,
                         by: str = "surface", min_matches: int = 3,
                         min_shots: int = 500) -> tuple[pd.DataFrame, pd.Series]:
    """Volume e resa del colpo, avversario per avversario, controllando `by`.

    **Volume** — i colpi attesi si ottengono applicando ai colpi giocati su
    ciascun livello della confondente il tasso del soggetto su quel livello, e
    sommando. Il rapporto `osservati / attesi` vale 1 quando l'avversario riceve
    esattamente il trattamento medio; sopra 1 quando ne riceve di più. Un
    confronto grezzo, senza questa correzione, misura per lo più il calendario:
    le smorzate vanno da ~2,5 ogni 100 colpi sul cemento a ~3,7 sulla terra.

    **Resa** — `differenziale` = punti vinti quando gioca quel colpo, meno punti
    vinti complessivamente contro quell'avversario. La sottrazione è il punto:
    senza, si misura quanto il soggetto è più forte dell'avversario, non quanto
    gli rende il colpo.

    Le soglie `min_matches` / `min_shots` escludono gli avversari su cui il
    tasso è troppo instabile per essere confrontato. Ritorna la tabella ordinata
    per rapporto crescente e il tasso di riferimento usato.
    """
    a = d[d.player == player]
    base = reference_rate(d, player, exclude_opponent, by)

    def attesi(sub: pd.DataFrame) -> float:
        # Un livello della confondente assente dal riferimento darebbe un NaN
        # silenzioso: meglio accorgersene qui che a valle, in un rapporto vuoto.
        mancanti = set(sub[by]) - set(base.index)
        if mancanti:
            raise ValueError(f"{by} assente dal riferimento: {sorted(mancanti)}")
        return float(sum(base[liv] * g.colpi.sum() for liv, g in sub.groupby(by)))

    righe = []
    for opp, sub in a.groupby("avversario"):
        if len(sub) < min_matches or sub.colpi.sum() <= min_shots:
            continue
        att = attesi(sub)
        pv, pp = sub.shots_in_pts_won.sum(), sub.shots_in_pts_lost.sum()
        # Denominatore a zero -> NaN, mai 0: un avversario contro cui non ha mai
        # giocato quel colpo non ha una resa pari a zero, ne ha una ignota.
        con_colpo = 100 * pv / (pv + pp) if (pv + pp) else float("nan")
        in_generale = 100 * sub.punti_vinti.sum() / sub.punti_match.sum()
        righe.append({
            "avversario": opp, "match": len(sub), "colpi": int(sub.colpi.sum()),
            "osservate": int(sub.shots.sum()), "attese": round(att, 1),
            "rapporto": sub.shots.sum() / att if att else float("nan"),
            "punti_smorzata": int(pv + pp), "punti_vinti_smorzata": int(pv),
            "con_smorzata": con_colpo, "in_generale": in_generale,
            "differenziale": con_colpo - in_generale,
        })
    return pd.DataFrame(righe).sort_values("rapporto").reset_index(drop=True), base


def shot_mix(d: pd.DataFrame, player: str, match_ids: set[str],
             codes: tuple[str, ...] = ("Base", "Net", "Vo", "Sl", "Dr", "Lo"),
             gender: str = "m") -> pd.DataFrame:
    """Quota dei tipi di colpo del soggetto, dentro e fuori un insieme di match.

    Serve a controllare se un effetto sulle smorzate è parte di un cambiamento
    più ampio del modo di giocare. Ogni codice è rapportato a `Total`, mai
    sommato agli altri: sono categorie annidate.
    """
    st = _load_shot_types(gender)
    st = st[st.player == player]
    piv = st.pivot_table(index="match_id", columns="row", values="shots", aggfunc="sum")

    righe = {}
    for lab, sub in [("dentro", piv[piv.index.isin(match_ids)]),
                     ("fuori", piv[~piv.index.isin(match_ids)])]:
        righe[lab] = {c: 100 * sub[c].sum() / sub["Total"].sum() for c in codes}
    return pd.DataFrame(righe).T


# ------------------------------------------------ un giocatore contro il circuito

def total_vs_decoded(match_ids: set[str], gender: str = "m", era: str = "2020s") -> pd.DataFrame:
    """`Total` di ShotTypes a confronto con i colpi decodificati punto per punto.

    Il nome non dice se `Total` includa i servizi, e cambia un tasso di circa un
    terzo. Per ogni giocatore-match si contano i colpi dopo il servizio nelle
    sequenze (`senza`) e gli stessi più i servizi (`con`): il rapporto con
    `Total` vicino a 1 dice quale dei due conta il file.

    La decodifica conta circa l'8% di colpi in più del file (scarto noto, vedi
    lib/points.py): il confronto regge sull'ordine di grandezza, non al colpo.
    """
    p = loaders.load_mcp_points(gender, era)
    p = p[p.match_id.isin(match_ids)]
    idx = loaders.load_mcp_matches(gender).set_index("match_id")
    total = load_shot_type("Dr", gender).set_index(["match_id", "player"]).colpi.to_dict()

    righe = []
    for mid, g in p.groupby("match_id"):
        colpi, servizi = {1: 0, 2: 0}, {1: 0, 2: 0}
        for svr, seq in zip(g.Svr.astype(int), points.played_sequence(g)):
            servizi[svr] += 1
            for k, _ in enumerate(points.shot_letters(seq)):
                # l'elemento 0 è la risposta, poi i colpi si alternano
                colpi[3 - svr if k % 2 == 0 else svr] += 1
        for lato in (1, 2):
            chiave = (mid, idx.at[mid, f"player_{lato}"])
            if chiave in total:
                righe.append({"match_id": mid, "player": chiave[1], "total": total[chiave],
                              "senza": colpi[lato], "con": colpi[lato] + servizi[lato]})
    return pd.DataFrame(righe)


def rates_by_player(d: pd.DataFrame, by: str = "surface") -> pd.DataFrame:
    """Tasso del colpo per giocatore, grezzo e a parità di `by`. Una riga per giocatore.

    Le smorzate (o il colpo di `d`) attese di un giocatore sono i suoi colpi su
    ciascun livello di `by` per il tasso del circuito su quel livello,
    **calcolato senza di lui**: altrimenti il metro contiene ciò che misura. Per
    quasi tutti è trascurabile; per un giocatore molto annotato (Alcaraz pesa da
    solo il 3% dei colpi del periodo) no.

    `rapporto` vale 1 per un giocatore nella media del circuito a parità di `by`.
    """
    ps = d.groupby(["player", by])[["shots", "colpi"]].sum()
    tot = ps.groupby(by).sum()
    altri = tot.reindex(ps.index.get_level_values(by)).to_numpy() - ps.to_numpy()
    ps["attese"] = ps.colpi * altri[:, 0] / altri[:, 1]

    g = ps.groupby("player")[["shots", "colpi", "attese"]].sum()
    g["match"] = d.groupby("player").match_id.nunique()
    g["tasso"] = 100 * g.shots / g.colpi
    g["rapporto"] = g.shots / g.attese
    return g


def bootstrap_player(a: pd.DataFrame, circuit_rate: pd.Series, n: int = 10_000,
                     seed: int = 0, by: str = "surface") -> tuple:
    """IC 95% di tasso (per 100 colpi) e rapporto, ricampionando i match del soggetto.

    `a` sono le righe del soggetto, `circuit_rate` il tasso del circuito (senza
    di lui) per livello di `by`. Si ricampionano i match interi perché i colpi
    di uno stesso match non sono indipendenti: stesso avversario, superficie,
    piano di gioco.
    """
    rng = np.random.default_rng(seed)
    dr, colpi = a.shots.to_numpy(), a.colpi.to_numpy()
    att = (a.colpi * a[by].map(circuit_rate)).to_numpy()
    i = rng.integers(0, len(a), (n, len(a)))
    tasso = 100 * dr[i].sum(1) / colpi[i].sum(1)
    rapporto = dr[i].sum(1) / att[i].sum(1)
    return np.percentile(tasso, [2.5, 97.5]), np.percentile(rapporto, [2.5, 97.5])


def rank_of(g: pd.DataFrame, player: str, column: str, min_shots: int) -> tuple[int, int, float]:
    """Posizione del giocatore partendo dal più alto, fra quelli con almeno
    `min_shots` colpi: (posizione, giocatori in classifica, mediana della colonna)."""
    s = g[g.colpi >= min_shots].sort_values(column, ascending=False)
    return list(s.index).index(player) + 1, len(s), float(s[column].median())
