"""Caricamento e normalizzazione dei dati grezzi.

Copre le due fonti scaricate da `lib.download`:

  - **Match Charting Project (MCP)**: elenco match, statistiche aggregate per
    match/giocatore/set, e sequenze punto per punto con la notazione dei colpi.
  - **tennis-data.co.uk**: un match per riga, con punteggio per set, ranking e
    quote dei bookmaker.

Limiti da dichiarare in qualunque risultato costruito su questi dati:
  - il MCP è **annotato a mano da volontari**: non è un campione casuale dei
    match giocati, è sbilanciato verso i big match e i giocatori popolari.
    Va bene per studiare *come* si gioca un punto, non per stimare frequenze
    sul circuito;
  - tennis-data.co.uk copre invece tutti i match, ma **senza statistiche di
    gioco**: solo punteggio, ranking e quote;
  - nessuna delle due fonti ha le statistiche di servizio complete su tutto il
    circuito. Il dataset che le aveva (`JeffSackmann/tennis_atp`) non è più
    pubblico — vedi docs/fonti-dati.md.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from .paths import RAW_DIR

MCP_DIR = RAW_DIR / "mcp"
TD_DIR = RAW_DIR / "tennis-data"


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} assente. Scaricalo con:\n    {hint}")
    return path


# --------------------------------------------------------------------------- MCP

# Superfici valide: serve anche a intercettare le righe con colonne disallineate.
MCP_SURFACES = {"Hard", "Clay", "Grass", "Carpet"}


def load_mcp_matches(gender: str = "m", clean: bool = True) -> pd.DataFrame:
    """Elenco dei match annotati: giocatori, torneo, superficie, data.

    Le colonne originali hanno spazi e maiuscole ("Player 1", "Best of"):
    qui diventano snake_case, perché ogni analisi altrimenti le rinomina da sé.

    `clean=True` scarta le righe malformate del file a monte — poche unità su
    migliaia, con le colonne disallineate (un nome di arbitro finito in
    `surface`, la data mancante). Sono innocue da guardare e velenose da usare:
    una duplica un `match_id` e fa duplicare le righe a ogni merge. Il numero di
    righe scartate viene segnalato, non nascosto.
    """
    path = _require(
        MCP_DIR / f"charting-{gender}-matches.csv",
        f"python3 -m lib.download mcp --gender {gender}",
    )
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns={
        "Player 1": "player_1", "Player 2": "player_2",
        "Pl 1 hand": "player_1_hand", "Pl 2 hand": "player_2_hand",
        "Date": "date", "Tournament": "tournament", "Round": "round",
        "Time": "time", "Court": "court", "Surface": "surface",
        "Umpire": "umpire", "Best of": "best_of", "Final TB?": "final_tb",
        "Charted by": "charted_by",
    })
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["gender"] = gender

    if clean:
        n = len(df)
        df = df[df["surface"].isin(MCP_SURFACES) & df["date"].notna()]
        df = df.drop_duplicates(subset="match_id", keep="first")
        if len(df) < n:
            warnings.warn(
                f"load_mcp_matches({gender!r}): scartate {n - len(df)} righe malformate "
                f"o duplicate su {n} (colonne disallineate nel CSV a monte)",
                stacklevel=2,
            )
        df = df.reset_index(drop=True)

    return df


def load_mcp_stats(name: str = "Overview", gender: str = "m", totals_only: bool = True) -> pd.DataFrame:
    """Un file di statistiche aggregate del MCP (elenco in lib.download.MCP_STATS).

    Ogni file ha una riga per match, giocatore e set, con `set == "Total"` per il
    match intero: `totals_only=True` (default) tiene solo quelle, che è quasi
    sempre ciò che serve — sommare i set produrrebbe doppi conteggi.
    """
    path = _require(
        MCP_DIR / f"charting-{gender}-stats-{name}.csv",
        f"python3 -m lib.download mcp --gender {gender}",
    )
    df = pd.read_csv(path, low_memory=False)
    if totals_only and "set" in df.columns:
        df = df[df["set"].astype(str) == "Total"].reset_index(drop=True)
    return df


def load_mcp_points(gender: str = "m", eras: str | list[str] = "2020s") -> pd.DataFrame:
    """Sequenze punto per punto. Pesanti: ~56 MB per il solo file maschile 2020s.

    Le colonne `1st` e `2nd` contengono la notazione MatchChart dello scambio
    (un carattere per colpo): non è testo libero, va decodificata con la legenda
    citata in docs/fonti-dati.md prima di essere usata come feature.
    """
    if isinstance(eras, str):
        eras = [eras]
    frames = []
    for era in eras:
        path = _require(
            MCP_DIR / f"charting-{gender}-points-{era}.csv",
            f"python3 -m lib.download mcp --gender {gender} --points",
        )
        df = pd.read_csv(path, low_memory=False)
        df["era"] = era
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_serve_metrics(overview: pd.DataFrame) -> pd.DataFrame:
    """Percentuali di servizio e risposta sulle statistiche Overview del MCP.

    Denominatori espliciti, perché è lì che si sbaglia:
      - `first_in_pct`   = first_in / serve_pts
      - `first_won_pct`  = first_won / first_in      (sulle prime **in campo**)
      - `second_won_pct` = second_won / second_in    dove `second_in` è
        `serve_pts - first_in`: sono i punti giocati con la seconda, **doppi
        falli inclusi** (verificato: l'identità vale sul 100% delle righe Total)
      - `dominance_ratio` = punti vinti in risposta / punti persi al servizio;
        > 1 significa essere più efficaci in risposta di quanto lo sia
        l'avversario. È la metrica di riferimento di Sackmann.

    I denominatori a zero diventano NaN, mai 0: una media su zeri finti è il
    modo più veloce per pubblicare un numero sbagliato.
    """
    df = overview.copy()

    def ratio(num: pd.Series, den: pd.Series) -> pd.Series:
        return num / den.where(den > 0)

    second_in = df["serve_pts"] - df["first_in"]
    df["second_in"] = df.get("second_in", second_in)

    df["first_in_pct"] = ratio(df["first_in"], df["serve_pts"])
    df["first_won_pct"] = ratio(df["first_won"], df["first_in"])
    df["second_won_pct"] = ratio(df["second_won"], second_in)
    df["serve_pts_won"] = df["first_won"] + df["second_won"]
    df["serve_pts_won_pct"] = ratio(df["serve_pts_won"], df["serve_pts"])
    df["ace_pct"] = ratio(df["aces"], df["serve_pts"])
    df["df_pct"] = ratio(df["dfs"], df["serve_pts"])
    df["bp_saved_pct"] = ratio(df["bp_saved"], df["bk_pts"])
    df["return_pts_won_pct"] = ratio(df["return_pts_won"], df["return_pts"])
    df["dominance_ratio"] = ratio(df["return_pts_won_pct"], 1 - df["serve_pts_won_pct"])

    if "winners" in df.columns and "unforced" in df.columns:
        df["winner_ue_ratio"] = ratio(df["winners"], df["unforced"])

    return df


def add_match_context(stats: pd.DataFrame, gender: str = "m") -> pd.DataFrame:
    """Aggiunge data, torneo, superficie e avversario a un file di statistiche MCP.

    I file di statistiche hanno solo `match_id` e `player`: senza questo join
    non si può filtrare per superficie o periodo, cioè la prima cosa che serve.
    """
    matches = load_mcp_matches(gender)
    cols = ["match_id", "date", "tournament", "round", "surface", "court", "best_of",
            "player_1", "player_2"]
    # Il merge deve conservare il numero di righe di `stats`: match_id è unico
    # in `matches` solo dopo la pulizia, e vale la pena verificarlo.
    n_before = len(stats)
    out = stats.merge(matches[cols], on="match_id", how="left", validate="many_to_one")
    assert len(out) == n_before, "il join con l'elenco match ha duplicato delle righe"
    # L'avversario è l'altro dei due nomi in tabella.
    out["opponent"] = out["player_1"].where(out["player"] != out["player_1"], out["player_2"])
    return out


# ------------------------------------------------------------------ tennis-data

# Colonne quote: bookmaker singoli (B365, Pinnacle) e aggregati di mercato (Max, Avg).
TD_ODDS_COLS = ["B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL"]


def load_odds_matches(tour: str = "atp", years: int | range | list[int] = range(2015, 2026)) -> pd.DataFrame:
    """Match del circuito con punteggio, ranking e quote, da tennis-data.co.uk.

    Richiede `openpyxl` (i file sono .xlsx). `Comment` distingue i match conclusi
    dai ritiri/walkover: filtrarlo è quasi sempre necessario.
    """
    if isinstance(years, int):
        years = [years]
    years = list(years)

    frames, missing = [], []
    for year in years:
        path = TD_DIR / tour / f"{year}.xlsx"
        if not path.exists():
            missing.append(year)
            continue
        df = pd.read_excel(path)
        df["season"] = year
        frames.append(df)

    if missing:
        raise FileNotFoundError(
            f"Mancano le stagioni {missing} in {TD_DIR / tour}. Scaricale con:\n"
            f"    python3 -m lib.download td --tour {tour} --from {min(missing)} --to {max(missing)}"
        )

    out = pd.concat(frames, ignore_index=True)
    out.columns = [str(c).strip() for c in out.columns]
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    return out


def implied_probabilities(odds: pd.DataFrame, winner_col: str = "AvgW",
                          loser_col: str = "AvgL") -> pd.DataFrame:
    """Probabilità implicite dalle quote, **normalizzate** per togliere il margine.

    1/quota non è una probabilità: la somma dei due inversi supera 1 (overround,
    il margine del bookmaker). Qui si divide per quella somma, che è la
    correzione più semplice — non l'unica né la più accurata, ma esplicita.
    """
    df = odds.copy()
    inv_w = 1 / df[winner_col]
    inv_l = 1 / df[loser_col]
    overround = inv_w + inv_l
    df["overround"] = overround
    df["p_winner"] = inv_w / overround
    df["p_loser"] = inv_l / overround
    return df
