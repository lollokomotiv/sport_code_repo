"""Alcaraz vs Paul — il confronto diretto visto dai dati.

Due livelli, due fonti:

  - **tutti** gli incontri, con le statistiche ufficiali di servizio e risposta,
    da TennisMyLife;
  - il **dettaglio colpo per colpo** (scambi, dritto, rete, punti chiave) solo
    sui match annotati dal Match Charting Project.

Nessun dato di mercato: interessa come si sono giocati i punti.

Il secondo livello copre meno match del primo, e non a caso: lo script misura
lo scarto e lo stampa prima di qualunque media.

    python3 -m lib.download tml --tour atp
    python3 -m lib.download mcp --gender m
    python3 analyses/alcaraz-paul-h2h/run.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib import loaders  # noqa: E402
from lib.paths import RAW_DIR  # noqa: E402

A, B = "Carlos Alcaraz", "Tommy Paul"
MCP_DIR = RAW_DIR / "mcp"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def pct(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.where(den > 0)).round(3)


def h2h_official() -> pd.DataFrame:
    """Tutti gli incontri disputati, dalle statistiche ufficiali."""
    t = loaders.load_tml("atp")
    n = loaders.normalize_name
    a, b = n(A), n(B)
    w, l = n(t["winner_name"]), n(t["loser_name"])
    hit = t[((w == a) & (l == b)) | ((w == b) & (l == a))]
    return hit.sort_values("tourney_date").reset_index(drop=True)


def h2h_charted(gender: str = "m") -> pd.DataFrame:
    """Gli incontri annotati colpo per colpo."""
    m = loaders.load_mcp_matches(gender)
    pair = {A, B}
    hit = m[m.apply(lambda r: {r["player_1"], r["player_2"]} == pair, axis=1)]
    return hit.sort_values("date").reset_index(drop=True)


def load_stat(name: str, ids: list[str], rows: list[str] | None = None) -> pd.DataFrame:
    """Un file di statistiche MCP, ristretto ai match e alle righe che servono.

    `row` è la dimensione che cambia da file a file: per Rally è la lunghezza
    dello scambio, per KeyPointsServe il tipo di punto, per ShotTypes il tipo di
    colpo. Filtrarla è indispensabile: sommare righe di livelli diversi conta
    gli stessi punti più volte.
    """
    df = pd.read_csv(MCP_DIR / f"charting-m-stats-{name}.csv", low_memory=False)
    df = df[df["match_id"].isin(ids)]
    return df[df["row"].isin(rows)] if rows else df


def main() -> None:
    official = h2h_official()
    charted = h2h_charted()
    if official.empty:
        print("Nessun incontro trovato. Scarica i dati: python3 -m lib.download tml --tour atp")
        return

    # ---------------------------------------------------------------- copertura
    section(f"{len(official)} INCONTRI DISPUTATI, {len(charted)} ANNOTATI COLPO PER COLPO")

    link = loaders.link_mcp_to_tml(charted, official) if not charted.empty else pd.DataFrame()
    charted_tml = set(link["tml_match_id"]) if len(link) else set()
    official = official.assign(annotato=official["tml_match_id"].isin(charted_tml))

    view = official[["tourney_date", "tourney_name", "surface", "round", "best_of",
                     "winner_name", "score", "annotato"]].copy()
    view["tourney_date"] = view["tourney_date"].dt.date
    view["annotato"] = view["annotato"].map({True: "si", False: "NO"})
    print(view.to_string(index=False))

    wins = official["winner_name"].value_counts()
    wins_ch = official[official.annotato]["winner_name"].value_counts()
    print(f"\nH2H completo:  {wins.get(A, 0)}-{wins.get(B, 0)} Alcaraz"
          f"   ({B.split()[-1]} vince il {100 * wins.get(B, 0) / len(official):.0f}%)")
    if len(charted):
        print(f"H2H annotato:  {wins_ch.get(A, 0)}-{wins_ch.get(B, 0)} Alcaraz"
              f"   ({B.split()[-1]} vince il {100 * wins_ch.get(B, 0) / int(official.annotato.sum()):.0f}%)")
        print("\nIl sottoinsieme annotato pende dalla parte di Paul: le sezioni sul\n"
              "dettaglio di gioco vanno lette sapendolo.")

    # ------------------------------------------- livello 1: tutti gli incontri
    section("SERVIZIO E RISPOSTA — TUTTI GLI INCONTRI (statistiche ufficiali)")
    long = loaders.add_serve_metrics(loaders.tml_to_long(official))
    long["match"] = (long["tourney_date"].dt.strftime("%Y-%m-%d") + "  "
                     + long["tourney_name"] + " (" + long["surface"] + ")")
    cols = ["match", "player", "serve_pts", "first_in_pct", "first_won_pct",
            "second_won_pct", "serve_pts_won_pct", "return_pts_won_pct", "dominance_ratio"]
    print(long[cols].round(3).sort_values(["match", "player"]).to_string(index=False))

    section("AGGREGATO UFFICIALE vs AGGREGATO SUI SOLI MATCH ANNOTATI")

    def summarise(df: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "match": df["serve_pts"].count() // 1,
            "serve_pts": df["serve_pts"].sum(),
            "serve_won_pct": round((df.first_won.sum() + df.second_won.sum()) / df.serve_pts.sum(), 3),
            "first_won_pct": round(df.first_won.sum() / df.first_in.sum(), 3),
            "return_won_pct": round(df.return_pts_won.sum() / df.return_pts.sum(), 3),
            "ace_pct": round(df.aces.sum() / df.serve_pts.sum(), 3),
            "bp_salvate_pct": round(df.bp_saved.sum() / df.bk_pts.sum(), 3),
        })

    rows = {}
    for player in (A, B):
        allm = long[long.player == player]
        sub = long[(long.player == player) & long.tml_match_id.isin(charted_tml)]
        rows[(player, "tutti")] = summarise(allm)
        if len(sub):
            rows[(player, "annotati")] = summarise(sub)
    print(pd.DataFrame(rows).T.to_string())
    print("\nLa differenza tra le due righe di ogni giocatore è la distorsione del\n"
          "campione annotato: non è enorme, ma esiste e va nella stessa direzione.")

    if charted.empty:
        return

    # --------------------------------- livello 2: solo i match annotati (MCP)
    ids = charted["match_id"].tolist()

    section("PUNTI VINTI PER LUNGHEZZA DELLO SCAMBIO — solo match annotati")
    # Il file Rally usa pl1/pl2, riferiti a player_1 e player_2 dell'indice:
    # vanno ricondotti ai nomi, altrimenti si sommano giocatori diversi.
    idx = charted.set_index("match_id")
    recs = []
    for _, x in load_stat("Rally", ids, ["1-3", "4-6", "7-9", "10"]).iterrows():
        a_is_p1 = idx.loc[x.match_id, "player_1"] == A
        recs.append({"len": x.row, "pts": x.pts, A: x.pl1_won if a_is_p1 else x.pl2_won})
    d = pd.DataFrame(recs).groupby("len")[["pts", A]].sum().reindex(["1-3", "4-6", "7-9", "10"])
    d[f"{A} %"] = pct(d[A], d["pts"])
    print(d.to_string())

    section("DRITTO E ROVESCIO A RIMBALZO — solo match annotati")
    st = load_stat("ShotTypes", ids, ["F", "B"]).groupby(["player", "row"])[
        ["shots", "winners", "unforced"]].sum()
    st["winner_per_100"] = (100 * st.winners / st.shots).round(2)
    st["unforced_per_100"] = (100 * st.unforced / st.shots).round(2)
    st.index = st.index.set_levels(["rovescio", "dritto"], level=1)
    print(st.to_string())

    section("PALLE BREAK E GIOCO A RETE — solo match annotati")
    ks = load_stat("KeyPointsServe", ids, ["BP"]).groupby("player")[["pts", "pts_won"]].sum()
    ks["salvate_pct"] = pct(ks.pts_won, ks.pts)
    ks.columns = ["bp_affrontate", "bp_salvate", "salvate_pct"]
    kr = load_stat("KeyPointsReturn", ids, ["BPO"]).groupby("player")[["pts", "pts_won"]].sum()
    kr["convertite_pct"] = pct(kr.pts_won, kr.pts)
    kr.columns = ["bp_avute", "bp_convertite", "convertite_pct"]
    print(ks.join(kr).to_string())
    net = load_stat("NetPoints", ids, ["NetPoints"]).groupby("player")[
        ["net_pts", "pts_won", "net_winner", "passed_at_net"]].sum()
    net["vinti_pct"] = pct(net.pts_won, net.net_pts)
    print("\n" + net.to_string())

    print(f"\n{'-' * 78}")
    print(f"Livello 1: {len(official)} match, {int(long.serve_pts.sum())} punti (ufficiale).")
    print(f"Livello 2: {len(charted)} match annotati. Numeri piccoli: le differenze "
          "per singolo match sono rumore.")


if __name__ == "__main__":
    main()
