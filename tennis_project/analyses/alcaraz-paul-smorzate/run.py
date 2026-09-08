"""Alcaraz gioca meno smorzate contro Tommy Paul? — vedi README.md.

Tesi da verificare: contro Paul, Alcaraz userebbe meno la smorzata e si
affiderebbe di più ai colpi da fondo campo.

Questo script contiene **tutti** i calcoli citati nel README, compresi i test di
significatività e i confronti di contorno: nessun numero pubblicato deve venire
da un comando lanciato a mano e poi perso.

    python3 -m lib.download mcp --gender m
    python3 analyses/alcaraz-paul-smorzate/run.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import shots  # noqa: E402
from lib.paths import RAW_DIR  # noqa: E402
from lib.stats import wilson, z_two_proportions  # noqa: E402

A, B = "Carlos Alcaraz", "Tommy Paul"
FIGURES = Path(__file__).parent / "figures"

# Soglie per considerare confrontabile un avversario: pochi match rendono i
# tassi instabili, e con pochi colpi il denominatore non regge.
MIN_MATCH, MIN_COLPI = 3, 500

# Palette: forma "emphasis" — un colore per il soggetto, grigio per il contesto.
BLU, GRIGIO = "#2a78d6", "#898781"
INCHIOSTRO, SECONDARIO, MUTO = "#0b0b0b", "#52514e", "#898781"
GRIGLIA, SUPERFICIE = "#e1e0d9", "#fcfcfb"


# --------------------------------------------------------------------- utilità

def sezione(titolo: str) -> None:
    print(f"\n{'=' * 78}\n{titolo}\n{'=' * 78}")


# -------------------------------------------------------------------- analisi

def esiti_smorzate(a: pd.DataFrame) -> pd.DataFrame:
    """Come finiscono le smorzate, contro Paul e contro tutti gli altri.

    Si guardano più esiti insieme di proposito: contare i soli vincenti puliti
    dà l'effetto più vistoso e più fragile, perché ignora che una smorzata può
    vincere il punto anche forzando l'errore dell'avversario.
    """
    righe = {}
    for lab, sub in [("vs Paul", a[a.avversario == B]), ("vs altri", a[a.avversario != B])]:
        s = sub.shots.sum()
        pv, pp = sub.shots_in_pts_won.sum(), sub.shots_in_pts_lost.sum()
        righe[lab] = {
            "smorzate": int(s),
            "vincenti_%": round(100 * sub.winners.sum() / s, 1),
            "forzano_%": round(100 * sub.induced_forced.sum() / s, 1),
            "gratuiti_%": round(100 * sub.unforced.sum() / s, 1),
            "decisive_%": round(100 * (sub.winners.sum() + sub.induced_forced.sum()) / s, 1),
            "punto_vinto_%": round(100 * pv / (pv + pp), 1),
            "punti_in_generale_%": round(100 * sub.punti_vinti.sum() / sub.punti_match.sum(), 1),
            # sottratto sui valori pieni: arrotondare prima sposta il risultato
            # di un decimo e lo fa divergere dal grafico
            "differenziale": round(100 * pv / (pv + pp)
                                   - 100 * sub.punti_vinti.sum() / sub.punti_match.sum(), 1),
        }
    return pd.DataFrame(righe).T


def composizione_colpi(ids_paul: set[str]) -> pd.DataFrame:
    """Quota dei tipi di colpo di Alcaraz, contro Paul e contro gli altri.

    Verifica la seconda metà della tesi ("si affida di più al fondo campo").
    I codici di `ShotTypes` sono gerarchici: `Base` e `Net` dividono il totale,
    `Vo` e `Sl` sono sottoinsiemi. Si confrontano quote sul totale, mai somme.
    """
    out = shots.shot_mix(None, A, ids_paul).round(2)
    out.index = ["vs Paul", "vs altri"]
    return out.rename(columns={"Base": "fondo_campo", "Net": "a_rete", "Vo": "volee",
                               "Sl": "slice", "Dr": "smorzate", "Lo": "pallonetti"})


def test(a: pd.DataFrame, ids_paul: set[str]) -> None:
    """Tutti i test di significatività citati nel README."""
    vs, altri = a[a.avversario == B], a[a.avversario != B]

    prove = [
        ("smorzate: solo vincenti", vs.winners.sum(), vs.shots.sum(),
         altri.winners.sum(), altri.shots.sum()),
        ("smorzate: decisive (vinc+forz)",
         vs.winners.sum() + vs.induced_forced.sum(), vs.shots.sum(),
         altri.winners.sum() + altri.induced_forced.sum(), altri.shots.sum()),
        ("smorzate: punto poi vinto",
         vs.shots_in_pts_won.sum(), vs.shots_in_pts_won.sum() + vs.shots_in_pts_lost.sum(),
         altri.shots_in_pts_won.sum(), altri.shots_in_pts_won.sum() + altri.shots_in_pts_lost.sum()),
    ]

    # Colpi a rete: si prende dal file dei tipi di colpo, non dalle smorzate.
    st = pd.read_csv(RAW_DIR / "mcp" / "charting-m-stats-ShotTypes.csv", low_memory=False)
    st = st[st.player == A]
    piv = st.pivot_table(index="match_id", columns="row", values="shots", aggfunc="sum")
    p_vs, p_al = piv[piv.index.isin(ids_paul)], piv[~piv.index.isin(ids_paul)]
    prove.append(("colpi a rete sul totale", p_vs["Net"].sum(), p_vs["Total"].sum(),
                  p_al["Net"].sum(), p_al["Total"].sum()))

    print(f"{'confronto':34}{'vs Paul':>12}{'vs altri':>12}{'z':>8}{'p':>9}")
    print("-" * 78)
    for lab, k1, n1, k2, n2 in prove:
        z, p = z_two_proportions(k1, n1, k2, n2)
        print(f"{lab:34}{100*k1/n1:>11.1f}%{100*k2/n2:>11.1f}%{z:>8.2f}{p:>9.4f}")
    print("\nIC 95% sui punti vinti con la smorzata:")
    for lab, sub in [("vs Paul", vs), ("vs altri", altri)]:
        pv = sub.shots_in_pts_won.sum()
        n = pv + sub.shots_in_pts_lost.sum()
        lo, hi = wilson(pv, n)
        print(f"  {lab:9} {100*pv/n:5.1f}%  [{lo:.1f} – {hi:.1f}]  n={int(n)}")


# --------------------------------------------------------------------- grafici

def grafico_volume(r: pd.DataFrame, dest: Path) -> None:
    """Dot plot del solo volume: un punto per avversario, Paul evidenziato."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q1, mediana, q3 = r.rapporto.quantile([0.25, 0.5, 0.75])
    y = list(range(len(r)))
    evid = (r.avversario == B).to_numpy()

    fig, ax = plt.subplots(figsize=(9, 8.5), facecolor=SUPERFICIE)
    ax.set_facecolor(SUPERFICIE)
    ax.axvspan(q1, q3, color=GRIGLIA, alpha=0.55, lw=0, zorder=0)
    ax.axvline(mediana, color=MUTO, lw=1, ls="--", zorder=1)
    ax.hlines(y, 0, r.rapporto, color=GRIGLIA, lw=1, zorder=2)
    ax.scatter(r.rapporto[~evid], [i for i, e in zip(y, evid) if not e], s=70,
               color=GRIGIO, zorder=3, edgecolors=SUPERFICIE, linewidths=1.5)
    ax.scatter(r.rapporto[evid], [i for i, e in zip(y, evid) if e], s=150,
               color=BLU, zorder=4, edgecolors=SUPERFICIE, linewidths=2)

    # Etichette solo dove servono: il soggetto e i due estremi.
    for i, row in r.iterrows():
        if row.avversario in {B, r.avversario.iloc[0], r.avversario.iloc[-1]}:
            colore = BLU if row.avversario == B else SECONDARIO
            ax.annotate(f"{row.rapporto:.2f}", (row.rapporto, i), xytext=(10, 0),
                        textcoords="offset points", va="center", fontsize=10, color=colore,
                        fontweight="bold" if row.avversario == B else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(r.avversario, fontsize=9.5)
    for tick, e in zip(ax.get_yticklabels(), evid):
        tick.set_color(BLU if e else SECONDARIO)
        if e:
            tick.set_fontweight("bold")
    ax.set_xlim(0.55, 1.52)
    ax.set_xlabel("smorzate giocate ÷ smorzate attese (a parità di superficie)",
                  fontsize=10, color=SECONDARIO, labelpad=10)
    ax.tick_params(axis="x", colors=MUTO, labelsize=9)
    ax.grid(axis="x", color=GRIGLIA, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for lato in ("top", "right", "left", "bottom"):
        ax.spines[lato].set_visible(False)
    ax.annotate("mediana", (mediana, len(r) - 0.35), xytext=(5, 0),
                textcoords="offset points", fontsize=9, color=MUTO, va="center")

    fig.suptitle("Contro Paul, Alcaraz gioca il suo numero normale di smorzate",
                 x=0.02, y=0.98, ha="left", fontsize=15, color=INCHIOSTRO, fontweight="bold")
    fig.text(0.02, 0.94, "Rapporto fra smorzate giocate e attese, avversario per avversario.\n"
             "Le attese tengono conto della superficie; la banda grigia contiene il 50% centrale\n"
             f"degli avversari. Solo chi Alcaraz ha affrontato almeno {MIN_MATCH} volte nei match annotati.",
             fontsize=10, color=SECONDARIO, va="top")
    fig.text(0.02, 0.015, "Fonte: Match Charting Project — match annotati a mano, "
             "non un campione casuale del circuito.", fontsize=8.5, color=MUTO)
    fig.tight_layout(rect=(0, 0.035, 1, 0.885))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=200, facecolor=SUPERFICIE)
    print(f"  scritto {dest.name}")


def grafico_doppio(r: pd.DataFrame, diff_altri: float, dest: Path) -> None:
    """Due pannelli affiancati, stesse righe: quante smorzate e quanto rendono.

    A destra il **differenziale**: punti vinti quando gioca una smorzata meno
    punti vinti in generale contro quell'avversario. Zero significa che la
    smorzata non aggiunge nulla al suo rendimento normale; è la lettura che
    rende confrontabili avversari di forza molto diversa.

    Le barre d'errore restano indispensabili: con 18-191 punti per avversario
    gran parte delle differenze non è distinguibile.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q1, mediana, q3 = r.rapporto.quantile([0.25, 0.5, 0.75])
    y = list(range(len(r)))
    evid = (r.avversario == B).to_numpy()

    # L'incertezza sta sul tasso con smorzata; la percentuale generale poggia su
    # migliaia di punti, quindi si tratta come nota.
    ic = [wilson(k, n) for k, n in zip(r.punti_vinti_smorzata, r.punti_smorzata)]
    lo = r.differenziale - (pd.Series([a for a, _ in ic], index=r.index) - r.in_generale)
    hi = (pd.Series([b for _, b in ic], index=r.index) - r.in_generale) - r.differenziale

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 8.6), facecolor=SUPERFICIE,
                                   sharey=True, gridspec_kw={"width_ratios": [1, 1.15]})

    ax1.axvspan(q1, q3, color=GRIGLIA, alpha=0.55, lw=0, zorder=0)
    ax1.axvline(mediana, color=MUTO, lw=1, ls="--", zorder=1)
    ax1.hlines(y, 0, r.rapporto, color=GRIGLIA, lw=1, zorder=2)
    ax1.scatter(r.rapporto[~evid], [i for i, e in zip(y, evid) if not e], s=65,
                color=GRIGIO, zorder=3, edgecolors=SUPERFICIE, linewidths=1.5)
    ax1.scatter(r.rapporto[evid], [i for i, e in zip(y, evid) if e], s=140,
                color=BLU, zorder=4, edgecolors=SUPERFICIE, linewidths=2)
    ax1.set_xlim(0.55, 1.52)
    ax1.set_title("Quante ne gioca", loc="left", fontsize=12, color=INCHIOSTRO,
                  fontweight="bold", pad=10)
    ax1.set_xlabel("giocate ÷ attese (a parità di superficie)", fontsize=9.5, color=SECONDARIO)

    ax2.axvline(0, color=MUTO, lw=1.2, zorder=1)
    ax2.errorbar(r.differenziale, y, xerr=[lo, hi], fmt="none", ecolor=GRIGLIA,
                 elinewidth=2.5, capsize=0, zorder=2)
    ax2.scatter(r.differenziale[~evid], [i for i, e in zip(y, evid) if not e], s=65,
                color=GRIGIO, zorder=3, edgecolors=SUPERFICIE, linewidths=1.5)
    ax2.scatter(r.differenziale[evid], [i for i, e in zip(y, evid) if e], s=140,
                color=BLU, zorder=4, edgecolors=SUPERFICIE, linewidths=2)
    ax2.set_title("Quanto gli rendono", loc="left", fontsize=12, color=INCHIOSTRO,
                  fontweight="bold", pad=10)
    ax2.set_xlabel("punti vinti quando gioca una smorzata, meno punti vinti in generale\n"
                   "(punti percentuali) — barre: intervallo al 95%",
                   fontsize=9.5, color=SECONDARIO)

    i = int(r.index[evid][0])
    for ax, valore, testo in ((ax1, r.rapporto.iloc[i], f"{r.rapporto.iloc[i]:.2f}"),
                              (ax2, r.differenziale.iloc[i], f"{r.differenziale.iloc[i]:+.1f}")):
        ax.annotate(testo, (valore, i), xytext=(12, 0), textcoords="offset points",
                    va="center", fontsize=10, color=BLU, fontweight="bold")
        ax.grid(axis="x", color=GRIGLIA, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=MUTO, labelsize=9)
        for lato in ("top", "right", "left", "bottom"):
            ax.spines[lato].set_visible(False)
    ax2.tick_params(axis="y", length=0)

    ax1.set_yticks(y)
    ax1.set_yticklabels(r.avversario, fontsize=9.5)
    for tick, e in zip(ax1.get_yticklabels(), evid):
        tick.set_color(BLU if e else SECONDARIO)
        if e:
            tick.set_fontweight("bold")
    ax1.annotate("mediana", (mediana, len(r) - 0.35), xytext=(5, 0),
                 textcoords="offset points", fontsize=9, color=MUTO, va="center")
    ax2.annotate("nessun vantaggio", (0, len(r) - 0.35), xytext=(6, 0),
                 textcoords="offset points", fontsize=9, color=MUTO, va="center")

    fig.suptitle("Contro Paul la smorzata di Alcaraz non gli fa vincere più punti del solito",
                 x=0.015, y=0.98, ha="left", fontsize=15.5, color=INCHIOSTRO, fontweight="bold")
    fig.text(0.015, 0.935,
             "Smorzate di Alcaraz per avversario, nei match annotati. A sinistra il volume, corretto "
             "per la superficie.\nA destra quanto la smorzata migliora il suo rendimento: contro tutti "
             f"gli altri vale in media {diff_altri:+.1f} punti percentuali, contro Paul "
             f"{r.differenziale[r.avversario == B].iloc[0]:+.1f}.",
             fontsize=10, color=SECONDARIO, va="top")
    fig.text(0.015, 0.015, "Fonte: Match Charting Project — match annotati a mano, "
             "non un campione casuale del circuito.", fontsize=8.5, color=MUTO)
    fig.tight_layout(rect=(0, 0.035, 1, 0.9))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=200, facecolor=SUPERFICIE)
    print(f"  scritto {dest.name}")


# ------------------------------------------------------------------------ main

def main() -> None:
    d = shots.load_shot_type("Dr", "m")
    a = d[d.player == A]
    vs = a[a.avversario == B]
    ids_paul = set(vs.match_id)

    sezione(f"ALCARAZ: {len(a)} MATCH ANNOTATI, {len(vs)} CONTRO {B.upper()}")
    print(f"tasso complessivo: {100 * a.shots.sum() / a.colpi.sum():.2f} smorzate ogni 100 colpi")

    r, base = shots.observed_vs_expected(d, A, exclude_opponent=B)
    print("\ntasso di riferimento per superficie (esclusi i match con Paul):")
    print((100 * base).round(2).to_string())

    riga = r[r.avversario == B].iloc[0]
    pos_vol = int(r.index[r.avversario == B][0]) + 1
    per_diff = r.sort_values("differenziale").reset_index(drop=True)
    pos_diff = int(per_diff.index[per_diff.avversario == B][0]) + 1

    sezione(f"CONTRO {B.upper()}")
    print(f"volume:        osservate {riga.osservate} | attese {riga.attese} | "
          f"rapporto {riga.rapporto:.2f} | posizione {pos_vol}/{len(r)} "
          f"(mediana {r.rapporto.median():.2f})")
    print(f"resa:          {riga.con_smorzata:.1f}% dei punti con smorzata contro "
          f"{riga.in_generale:.1f}% in generale | differenziale {riga.differenziale:+.1f} | "
          f"posizione {pos_diff}/{len(r)}")

    sezione("TUTTI GLI AVVERSARI, PER DIFFERENZIALE")
    print(per_diff[["avversario", "match", "osservate", "rapporto", "punti_smorzata",
                    "con_smorzata", "in_generale", "differenziale"]]
          .round({"rapporto": 2, "con_smorzata": 1, "in_generale": 1, "differenziale": 1})
          .to_string(index=False))

    sezione("COME FINISCONO LE SMORZATE")
    esiti = esiti_smorzate(a)
    print(esiti.to_string())
    print("\nContro Paul meno vincenti puliti ma più errori forzati: contare solo i")
    print("vincenti darebbe l'effetto più vistoso e meno solido (vedi i test).")

    sezione("COMPOSIZIONE DEI COLPI DI ALCARAZ (% sul totale)")
    print(composizione_colpi(ids_paul).to_string())

    sezione("TEST DI SIGNIFICATIVITÀ")
    test(a, ids_paul)

    sezione("GRAFICI")
    diff_altri = float(esiti.loc["vs altri", "differenziale"])
    grafico_volume(r, FIGURES / "smorzate-per-avversario.png")
    grafico_doppio(r, diff_altri, FIGURES / "smorzate-volume-e-resa.png")

    print(f"\nCampione: {len(vs)} match contro Paul, {int(vs.shots.sum())} smorzate. "
          "Le differenze per singolo match sono rumore.")


if __name__ == "__main__":
    main()
