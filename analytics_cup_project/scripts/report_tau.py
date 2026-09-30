"""Scrive reports/tau_opp.md a partire da reports/tau_opp.csv (calcola_tau.py).

Riporta i numeri, non li interpreta: l'interpretazione si fa insieme.

Definizioni (fissate prima di guardare i risultati):
- time_to_impact di SkillCorner: si usa l'id ordinale 1-5
  (very_easy, easy, medium, hard, very_hard);
- progressione = distanza del portatore dalla porta avversaria a inizio possesso
  meno quella a fine possesso (paper, §3.3); esclusi direct play e portieri;
- zone per distanza dalla porta a inizio possesso: < 35 m, 35-70 m, > 70 m (paper);
- perdita di palla: possessi che finiscono con end_type == "pass", in gioco aperto
  (game_interruption_before vuoto), con pass_outcome noto; perdita = esito diverso
  da "successful" (unsuccessful oppure offside). I dati non distinguono passaggi
  ordinari, filtranti e cross: si prendono tutti;
- bin di τ_opp: passo 0,25 s fino a 2,5 s, poi un bin unico oltre.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "reports" / "tau_opp.csv"
OUT = ROOT / "reports" / "tau_opp.md"

BORDI = list(np.arange(0, 2.5001, 0.25)) + [np.inf]


def binna(s):
    return pd.cut(s, BORDI, right=False,
                  labels=[f"{a:.2f}–{b:.2f}" if np.isfinite(b) else f"≥ {a:.2f}"
                          for a, b in zip(BORDI, BORDI[1:])])


def tabella(df):
    """DataFrame -> tabella markdown, senza dipendere da tabulate."""
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    teste = [str(df.index.name or "")] + [str(c) for c in df.columns]
    righe = ["| " + " | ".join(teste) + " |", "|" + "---|" * len(teste)]
    for idx, r in df.iterrows():
        righe.append("| " + " | ".join([str(idx)] + [fmt(v) for v in r]) + " |")
    return "\n".join(righe)


def spearman(d, quando):
    s = d[[f"tau_opp_{quando}", f"time_to_impact_{quando}_id"]].dropna()
    rho, p = spearmanr(s.iloc[:, 0], s.iloc[:, 1])
    return len(s), rho, p


def main():
    d = pd.read_csv(CSV, low_memory=False)
    righe = []
    w = righe.append

    w("# τ_opp sulle 20 partite\n")
    w("Generato da `scripts/report_tau.py` su `reports/tau_opp.csv` "
      "(prodotto da `scripts/calcola_tau.py`). Solo numeri: l'interpretazione è da fare.\n")
    w("Metodo: Narizuka et al., α = 1,0 s⁻¹, V_max = 10,0 m/s; velocità per derivata "
      "centrale sulle posizioni, senza lisciatura; x_b = posizione di tracking del "
      "portatore. Scheda: `notes/letteratura/pressione-tempo-arrivo.md`.\n")

    w("## Copertura\n")
    w(f"- partite: {d.match_id.nunique()}")
    w(f"- player_possession esclusi i portieri: {len(d)} "
      f"(coppie match_id, event_id duplicate: {d.duplicated(['match_id', 'event_id']).sum()})")
    w(f"- τ_opp calcolato all'inizio: {d.tau_opp_start.notna().sum()} "
      f"({d.tau_opp_start.notna().mean():.1%}); al rilascio: {d.tau_opp_end.notna().sum()} "
      f"({d.tau_opp_end.notna().mean():.1%})")
    w(f"- direct play (frame_start == frame_end): {d.direct_play.sum()} ({d.direct_play.mean():.1%})")
    for q in ("start", "end"):
        ok = d[f"tau_opp_{q}"].notna()
        w(f"- avversari con velocità definita, {q}: mediana "
          f"{d.loc[ok, f'n_avv_validi_{q}'].median():.0f}, "
          f"possessi con meno di 11: {(d.loc[ok, f'n_avv_validi_{q}'] < 11).mean():.1%}")
    w("")
    desc = d[["tau_opp_start", "tau_opp_end"]].describe().T
    w(tabella(desc) + "\n")
    np_ = d[~d.direct_play].dropna(subset=["tau_opp_start", "tau_opp_end"])
    diff = np_.tau_opp_start - np_.tau_opp_end
    w(f"τ_opp(t_get) − τ_opp(t_rel), possessi non direct play (N = {len(np_)}): "
      f"media {diff.mean():.3f} s, deviazione standard {diff.std():.3f} s "
      "(paper: 0,276 s e 0,400 s).\n")

    w("## 1. Correlazione con time_to_impact di SkillCorner\n")
    w("Spearman fra τ_opp (s) e `time_to_impact_{start,end}_id` "
      "(1 = very_easy … 5 = very_hard), sugli stessi istanti.\n")
    w("| istante | N | ρ di Spearman | p |")
    w("|---|---|---|---|")
    for q, nome in (("start", "inizio possesso"), ("end", "rilascio")):
        n, rho, p = spearman(d, q)
        w(f"| {nome} | {n} | {rho:.3f} | {p:.2g} |")
    w("")
    for q, nome in (("start", "inizio"), ("end", "rilascio")):
        g = d.groupby(f"time_to_impact_{q}_id")[f"tau_opp_{q}"].agg(["count", "median", "mean"])
        w(f"τ_opp per categoria di time_to_impact, {nome}:\n")
        w(tabella(g) + "\n")

    w("## 2. Quota di possessi con l'avversario più vicino estrapolato\n")
    w("\"Più vicino\" = l'avversario che realizza τ_opp (argmin del tempo di arrivo). "
      "Per confronto anche l'avversario più vicino in distanza, e il portatore stesso.\n")
    w("| istante | N | avversario di τ_opp estrapolato | avversario più vicino in distanza estrapolato | portatore estrapolato | i due avversari non coincidono |")
    w("|---|---|---|---|---|---|")
    for q, nome in (("start", "inizio possesso"), ("end", "rilascio")):
        s = d[d[f"tau_opp_{q}"].notna()]
        w(f"| {nome} | {len(s)} | {(s[f'avv_tau_detected_{q}'] == False).mean():.1%} | "  # noqa: E712
          f"{(s[f'avv_dist_detected_{q}'] == False).mean():.1%} | "  # noqa: E712
          f"{(s[f'portatore_detected_{q}'] == False).mean():.1%} | "  # noqa: E712
          f"{(s[f'avv_tau_id_{q}'] != s[f'avv_dist_id_{q}']).mean():.1%} |")
    s = d.dropna(subset=["tau_opp_start", "tau_opp_end"])
    uno = (s.avv_tau_detected_start == False) | (s.avv_tau_detected_end == False)  # noqa: E712
    w(f"\nPossessi in cui l'avversario di τ_opp è estrapolato in almeno uno dei due istanti: "
      f"{uno.mean():.1%} (N = {len(s)}).\n")
    per_partita = d.groupby("match_id").apply(
        lambda g: (g.avv_tau_detected_start[g.tau_opp_start.notna()] == False).mean(),  # noqa: E712
        include_groups=False)
    w(f"Per partita, all'inizio: min {per_partita.min():.1%}, mediana {per_partita.median():.1%}, "
      f"max {per_partita.max():.1%}.\n")

    w("## 3. Pressione all'inizio → avanzamento della palla (paper, Fig. 6)\n")
    p = d[~d.direct_play].dropna(subset=["tau_opp_start", "progressione"]).copy()
    p["bin"] = binna(p.tau_opp_start)
    w(f"Possessi non direct play, portieri esclusi: N = {len(p)}. "
      f"Progressione in metri (positiva = verso la porta avversaria).\n")
    rho, pv = spearmanr(p.tau_opp_start, p.progressione)
    w(f"Spearman τ_opp(t_get) vs progressione: ρ = {rho:.3f}, p = {pv:.2g}.\n")

    def tab_prog(g):
        return g.groupby("bin", observed=False).progressione.agg(
            N="count", media="mean", sd="std",
            **{"P(>0 m)": lambda x: (x > 0).mean(), "P(>3 m)": lambda x: (x > 3).mean(),
               "P(>6 m)": lambda x: (x > 6).mean()})

    w(tabella(tab_prog(p)) + "\n")
    p["zona"] = pd.cut(p.dist_porta_start, [0, 35, 70, np.inf], right=False,
                       labels=["< 35 m", "35–70 m", "> 70 m"])
    for z, g in p.groupby("zona", observed=True):
        rho, pv = spearmanr(g.tau_opp_start, g.progressione)
        w(f"### Zona {z} dalla porta (N = {len(g)}; Spearman ρ = {rho:.3f}, p = {pv:.2g})\n")
        w(tabella(tab_prog(g)) + "\n")

    w("## 4. Pressione al rilascio → perdita di palla (paper, Fig. 7a)\n")
    l = d[(d.end_type == "pass") & d.game_interruption_before.isna()
          & d.pass_outcome.isin(["successful", "unsuccessful", "offside"])
          ].dropna(subset=["tau_opp_end"]).copy()
    l["persa"] = l.pass_outcome != "successful"
    l["bin"] = binna(l.tau_opp_end)
    l["tipo"] = np.where(l.direct_play, "direct play", "possession play")
    w(f"Passaggi in gioco aperto: N = {len(l)} "
      f"(possession play {(~l.direct_play).sum()}, direct play {l.direct_play.sum()}). "
      f"Esclusi {((d.end_type == 'pass') & d.game_interruption_before.notna()).sum()} "
      "passaggi di possessi iniziati da palla inattiva.\n")
    for t, g in l.groupby("tipo"):
        rho, pv = spearmanr(g.tau_opp_end, g.persa)
        w(f"### {t} (N = {len(g)}; perdita media {g.persa.mean():.1%}; "
          f"Spearman τ_opp(t_rel) vs perdita ρ = {rho:.3f}, p = {pv:.2g})\n")
        w(tabella(g.groupby("bin", observed=False).persa.agg(N="count", P_perdita="mean")) + "\n")

    w("## Differenze dal paper da tenere presenti\n")
    w("- 10 fps invece di 25; nessuna lisciatura aggiuntiva (il paper usa Savitzky–Golay "
      "e spline); traiettorie broadcast con posizioni estrapolate.")
    w("- Intervalli di possesso e esiti presi da `player_possession` di SkillCorner, non "
      "da una sincronizzazione evento-tracking propria.")
    w("- Perdita di palla: nessuna etichetta per passaggio ordinario/filtrante; incluso "
      "ogni `end_type == pass` in gioco aperto. \"Gioco fermo dopo l'azione\" è coperto "
      "solo tramite pass_outcome (unsuccessful/offside).")
    w("- 20 partite contro 306.")

    OUT.write_text("\n".join(righe) + "\n")
    print(f"scritto {OUT}")


if __name__ == "__main__":
    main()
