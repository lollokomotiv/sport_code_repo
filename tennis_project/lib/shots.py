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

import pandas as pd

from . import loaders
from .paths import RAW_DIR

# Colonne di esito di ShotTypes che hanno senso per un singolo codice di colpo.
_ESITI = ["shots", "winners", "induced_forced", "unforced",
          "shots_in_pts_won", "shots_in_pts_lost"]


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
    """
    st = pd.read_csv(RAW_DIR / "mcp" / f"charting-{gender}-stats-ShotTypes.csv",
                     low_memory=False)

    colpo = st[st.row == code].set_index(["match_id", "player"])[_ESITI]
    totale = (st[st.row == "Total"].set_index(["match_id", "player"])[["shots"]]
              .rename(columns={"shots": "colpi"}))

    ov = loaders.load_mcp_stats("Overview", gender).set_index(["match_id", "player"])
    ov["punti_match"] = ov.serve_pts + ov.return_pts
    ov["punti_vinti"] = ov.first_won + ov.second_won + ov.return_pts_won

    idx = loaders.load_mcp_matches(gender)
    d = (colpo.join(totale, how="inner")
              .join(ov[["punti_match", "punti_vinti"]], how="inner")
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
    st = pd.read_csv(RAW_DIR / "mcp" / f"charting-{gender}-stats-ShotTypes.csv",
                     low_memory=False)
    st = st[st.player == player]
    piv = st.pivot_table(index="match_id", columns="row", values="shots", aggfunc="sum")

    righe = {}
    for lab, sub in [("dentro", piv[piv.index.isin(match_ids)]),
                     ("fuori", piv[~piv.index.isin(match_ids)])]:
        righe[lab] = {c: 100 * sub[c].sum() / sub["Total"].sum() for c in codes}
    return pd.DataFrame(righe).T
