"""Scrive reports/equivalenza_tti.md: time_to_impact può sostituire τ_opp?

Protocollo in plans/01-scegliere-la-domanda.md, «Il test di equivalenza»,
fissato prima di guardare i risultati. Riporta i numeri, non li interpreta.

"Equivalente" = time_to_impact sostituisce τ_opp come pressione sul portatore a
inizio e fine possesso. Definizioni di progressione, zone e perdita di palla
identiche a scripts/report_tau.py. Ogni confronto usa gli stessi possessi: quelli
in cui esistono entrambe le misure.

Segni attesi: τ_opp cresce quando la pressione cala (secondi di margine),
time_to_impact_id cresce quando la pressione sale (1 = very_easy … 5 = very_hard).
Le due misure hanno quindi segni opposti rispetto allo stesso esito.

Parametro tecnico: negli strati del confronto 3 con meno di 30 possessi ρ si
riporta ma non entra nella media pesata.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from report_tau import tabella

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "reports" / "tau_opp.csv"
OUT = ROOT / "reports" / "equivalenza_tti.md"

N_MIN_STRATO = 30
ZONE = ([0, 35, 70, np.inf], ["< 35 m", "35–70 m", "> 70 m"])


def rho(x, y):
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan
    r, p = spearmanr(x, y)
    return r, p


def campione_avanzamento(d):
    return d[~d.direct_play].dropna(
        subset=["tau_opp_start", "time_to_impact_start_id", "progressione"]).copy()


def campione_perdita(d):
    l = d[(d.end_type == "pass") & d.game_interruption_before.isna()
          & d.pass_outcome.isin(["successful", "unsuccessful", "offside"])
          ].dropna(subset=["tau_opp_end", "time_to_impact_end_id"]).copy()
    l["persa"] = (l.pass_outcome != "successful").astype(int)
    l["tipo"] = np.where(l.direct_play, "direct play", "possession play")
    return l


def confronto_diretto(g, tau, tti, esito):
    r_tau, p_tau = rho(g[tau], g[esito])
    r_tti, p_tti = rho(g[tti], g[esito])
    return {"N": len(g), "ρ τ_opp": r_tau, "p τ_opp": p_tau,
            "ρ time_to_impact": r_tti, "p time_to_impact": p_tti}


def stratificato(g, strato, misura, esito):
    """ρ(misura, esito) dentro ogni strato, e media pesata sugli strati con N >= N_MIN_STRATO."""
    righe = []
    for s, h in g.groupby(strato, observed=True):
        r, p = rho(h[misura], h[esito])
        righe.append({"strato": s, "N": len(h), "ρ": r, "p": p})
    t = pd.DataFrame(righe).set_index("strato")
    validi = t[(t.N >= N_MIN_STRATO) & t["ρ"].notna()]
    media = np.average(validi["ρ"], weights=validi.N) if len(validi) else np.nan
    return t, media, int(validi.N.sum())


def main():
    d = pd.read_csv(CSV, low_memory=False)
    righe = []
    w = righe.append

    w("# time_to_impact al posto di τ_opp?\n")
    w("Generato da `scripts/equivalenza_tti.py` su `reports/tau_opp.csv`. Protocollo in "
      "`plans/01-scegliere-la-domanda.md`, «Il test di equivalenza». Solo numeri: "
      "l'interpretazione è da fare.\n")
    w("Segni attesi opposti: τ_opp in secondi (più alto = meno pressione), "
      "`time_to_impact_id` da 1 (very_easy) a 5 (very_hard). Ogni confronto usa gli "
      "stessi possessi, quelli in cui esistono entrambe le misure.\n")

    # 1. avanzamento
    a = campione_avanzamento(d)
    a["zona"] = pd.cut(a.dist_porta_start, ZONE[0], right=False, labels=ZONE[1])
    w("## 1. Pressione all'inizio → avanzamento (paper, Fig. 6)\n")
    w(f"Possessi non direct play, portieri esclusi, con entrambe le misure all'inizio: "
      f"N = {len(a)}.\n")
    t = pd.DataFrame({"tutti": confronto_diretto(a, "tau_opp_start", "time_to_impact_start_id", "progressione")}).T
    for z, g in a.groupby("zona", observed=True):
        t.loc[f"zona {z}"] = confronto_diretto(g, "tau_opp_start", "time_to_impact_start_id", "progressione")
    t.index.name = "campione"
    t["N"] = t["N"].astype(int)
    w(tabella(t) + "\n")
    g = a.groupby("time_to_impact_start_id").progressione.agg(
        N="count", media="mean", **{"P(>0 m)": lambda x: (x > 0).mean(), "P(>6 m)": lambda x: (x > 6).mean()})
    g.index.name = "time_to_impact_start_id"
    w("Progressione per classe di `time_to_impact` all'inizio:\n")
    w(tabella(g) + "\n")

    # 2. perdita
    l = campione_perdita(d)
    w("## 2. Pressione al rilascio → perdita di palla (paper, Fig. 7a)\n")
    w(f"Passaggi in gioco aperto con entrambe le misure al rilascio: N = {len(l)}.\n")
    t = pd.DataFrame({tipo: confronto_diretto(g, "tau_opp_end", "time_to_impact_end_id", "persa")
                      for tipo, g in l.groupby("tipo")}).T
    t.index.name = "tipo"
    t["N"] = t["N"].astype(int)
    w(tabella(t) + "\n")
    g = l.groupby(["tipo", "time_to_impact_end_id"]).persa.agg(N="count", P_perdita="mean").reset_index()
    g.index = g.tipo + " · classe " + g.time_to_impact_end_id.astype(int).astype(str)
    g.index.name = "tipo · classe"
    w("Perdita per classe di `time_to_impact` al rilascio:\n")
    w(tabella(g[["N", "P_perdita"]]) + "\n")

    # 3. informazione aggiuntiva
    w("## 3. Informazione aggiuntiva, nei due sensi\n")
    w(f"Dentro ogni strato di una misura, ρ dell'altra con l'esito. Media pesata per N sugli "
      f"strati con almeno {N_MIN_STRATO} possessi. I quintili di τ_opp sono calcolati sul "
      "campione del confronto.\n")
    casi = [
        ("avanzamento", a, "tau_opp_start", "time_to_impact_start_id", "progressione"),
        ("perdita, possession play", l[l.tipo == "possession play"], "tau_opp_end", "time_to_impact_end_id", "persa"),
        ("perdita, direct play", l[l.tipo == "direct play"], "tau_opp_end", "time_to_impact_end_id", "persa"),
    ]
    sintesi = []
    for nome, g, tau, tti, esito in casi:
        g = g.copy()
        g["quintile_tau"] = pd.qcut(g[tau], 5, labels=[f"Q{i}" for i in range(1, 6)])
        t1, m1, n1 = stratificato(g, tti, tau, esito)
        t2, m2, n2 = stratificato(g, "quintile_tau", tti, esito)
        r_tau, _ = rho(g[tau], g[esito])
        r_tti, _ = rho(g[tti], g[esito])
        sintesi.append({"esito": nome, "N": len(g),
                        "ρ τ_opp (tutti)": r_tau, "ρ τ_opp dentro le classi tti": m1,
                        "ρ tti (tutti)": r_tti, "ρ tti dentro i quintili τ_opp": m2})
        w(f"### {nome}\n")
        t1.index.name = f"classe {tti}"
        w(f"τ_opp dentro le classi di `time_to_impact` (media pesata ρ = {m1:.3f}, su N = {n1}):\n")
        w(tabella(t1.assign(N=t1.N.astype(int))) + "\n")
        t2.index.name = "quintile di τ_opp"
        w(f"`time_to_impact` dentro i quintili di τ_opp (media pesata ρ = {m2:.3f}, su N = {n2}):\n")
        w(tabella(t2.assign(N=t2.N.astype(int))) + "\n")
    s = pd.DataFrame(sintesi).set_index("esito")
    s["N"] = s["N"].astype(int)
    w("### Sintesi\n")
    w(tabella(s) + "\n")

    # 4. copertura
    w("## 4. Copertura\n")
    w("| istante | possessi | entrambe | solo τ_opp | solo time_to_impact | nessuna | solo τ_opp, di cui direct play |")
    w("|---|---|---|---|---|---|---|")
    for q, nome in (("start", "inizio"), ("end", "rilascio")):
        t_ = d[f"tau_opp_{q}"].notna()
        k_ = d[f"time_to_impact_{q}_id"].notna()
        w(f"| {nome} | {len(d)} | {(t_ & k_).sum()} | {(t_ & ~k_).sum()} | {(~t_ & k_).sum()} | "
          f"{(~t_ & ~k_).sum()} | {(t_ & ~k_ & d.direct_play).sum()} |")
    w("")
    dp = d[d.direct_play]
    w(f"Direct play senza `time_to_impact` all'inizio: {dp.time_to_impact_start_id.isna().sum()} su "
      f"{len(dp)} ({dp.time_to_impact_start_id.isna().mean():.1%}).\n")
    l_tutti = d[(d.end_type == "pass") & d.game_interruption_before.isna()
                & d.pass_outcome.isin(["successful", "unsuccessful", "offside"])].dropna(subset=["tau_opp_end"])
    w(f"Campioni rispetto a `reports/tau_opp.md` (solo τ_opp): avanzamento "
      f"{len(a)} su {len(d[~d.direct_play].dropna(subset=['tau_opp_start', 'progressione']))}; "
      f"perdita {len(l)} su {len(l_tutti)}.\n")

    OUT.write_text("\n".join(righe) + "\n")
    print(f"scritto {OUT}")


if __name__ == "__main__":
    main()
