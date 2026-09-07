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

import re
import unicodedata
import warnings
from pathlib import Path

import pandas as pd

from .paths import PROCESSED_DIR, RAW_DIR

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
    # `best_of` arriva come stringa dal CSV (le righe malformate lo rendono
    # testuale); TennisMyLife lo ha intero. Se restano tipi diversi, ogni
    # concatenazione tra le due fonti produce una colonna mista che il parquet
    # rifiuta — e che il CSV accetterebbe in silenzio.
    if "best_of" in df.columns:
        df["best_of"] = pd.to_numeric(df["best_of"], errors="coerce").astype("Int64")

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


# ----------------------------------------------------------------- TennisMyLife

TML_DIR = RAW_DIR / "tml"

# Corrispondenza tra le colonne TennisMyLife (schema Sackmann) e quelle del MCP.
# Tenere gli stessi nomi permette di passare entrambe le fonti alle stesse
# funzioni di metrica, ed è ciò che rende possibile il fallback.
_TML_TO_MCP = {
    "svpt": "serve_pts", "ace": "aces", "df": "dfs", "1stIn": "first_in",
    "1stWon": "first_won", "2ndWon": "second_won", "bpSaved": "bp_saved",
    "bpFaced": "bk_pts", "SvGms": "serve_games",
}


def load_tml(tour: str = "atp", years: int | range | list[int] | None = None) -> pd.DataFrame:
    """Match ufficiali da TennisMyLife: una riga per match, schema Sackmann.

    `tourney_date` è la data di **inizio del torneo**, non del singolo match:
    è la differenza che rende impossibile un join per data esatta con il MCP.
    """
    d = TML_DIR / tour
    if years is None:
        paths = sorted(d.glob("*.csv"))
        if not paths:
            raise FileNotFoundError(
                f"{d} vuota. Scarica con: python3 -m lib.download tml --tour {tour}")
    else:
        if isinstance(years, int):
            years = [years]
        paths, missing = [], []
        for y in years:
            p = d / f"{y}.csv"
            (paths if p.exists() else missing).append(p if p.exists() else y)
        if missing:
            raise FileNotFoundError(
                f"Mancano le stagioni {missing} in {d}. Scarica con:\n"
                f"    python3 -m lib.download tml --tour {tour} "
                f"--from {min(missing)} --to {max(missing)}")

    out = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    out["tourney_date"] = pd.to_datetime(out["tourney_date"], format="%Y%m%d", errors="coerce")
    out["tour"] = tour
    # Chiave di match: TennisMyLife non ne ha una, si compone come in Sackmann.
    # Due insidie, entrambe piccole ma capaci di rompere ogni join a valle:
    # `match_num` è float (l'id verrebbe "…-55.0") ed è vuoto su 489 righe su
    # 200mila; in più un paio di tornei ripetono la coppia tourney_id+match_num.
    # Le righe problematiche prendono un id posizionale, così la chiave resta
    # unica per costruzione invece di esserlo per fortuna.
    num = pd.to_numeric(out["match_num"], errors="coerce").astype("Int64")
    key = out["tourney_id"].astype(str) + "-" + num.astype(str)
    broken = num.isna() | key.duplicated(keep=False)
    key = key.mask(broken, out["tourney_id"].astype(str) + "-r"
                   + pd.Series(range(len(out)), index=out.index).astype(str))
    out["tml_match_id"] = key
    assert out["tml_match_id"].is_unique, "tml_match_id non è unico"
    if "best_of" in out.columns:
        out["best_of"] = pd.to_numeric(out["best_of"], errors="coerce").astype("Int64")
    return out


def tml_to_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Da una riga per match a due righe (una per giocatore), con i nomi MCP.

    In uscita le colonne sono quelle di `charting-*-stats-Overview.csv`, così
    `add_serve_metrics()` funziona identica sulle due fonti. `winners` e
    `unforced` restano assenti: TennisMyLife non li ha, e sono esattamente il
    tipo di dato per cui serve l'annotazione manuale del MCP.
    """
    m = matches.copy()
    if "tml_match_id" not in m.columns:
        m["tml_match_id"] = m["tourney_id"].astype(str) + "-" + m["match_num"].astype(str)

    context = [c for c in ["tml_match_id", "tourney_id", "tourney_name", "tourney_date",
                           "surface", "indoor", "tourney_level", "round", "best_of",
                           "score", "minutes", "draw_size", "tour"] if c in m.columns]

    def side(win: bool) -> pd.DataFrame:
        me, opp = ("w", "l") if win else ("l", "w")
        me_n, opp_n = ("winner", "loser") if win else ("loser", "winner")

        out = m[context].copy()
        out["player"] = m[f"{me_n}_name"]
        out["opponent"] = m[f"{opp_n}_name"]
        for extra in ["hand", "ht", "ioc", "age", "rank", "rank_points"]:
            if f"{me_n}_{extra}" in m.columns:
                out[f"player_{extra}"] = m[f"{me_n}_{extra}"]
        for src, dst in _TML_TO_MCP.items():
            if f"{me}_{src}" in m.columns:
                out[dst] = m[f"{me}_{src}"]
        # Risposta: si ricava dal servizio dell'avversario, non esiste come colonna.
        opp_won = m[f"{opp}_1stWon"] + m[f"{opp}_2ndWon"]
        out["return_pts"] = m[f"{opp}_svpt"]
        out["return_pts_won"] = m[f"{opp}_svpt"] - opp_won
        out["won"] = int(win)
        return out

    long = pd.concat([side(True), side(False)], ignore_index=True)
    long["second_in"] = long["serve_pts"] - long["first_in"]
    long["set"] = "Total"  # stesso livello delle righe Total del MCP
    return long.sort_values(["tourney_date", "tml_match_id", "won"],
                            ascending=[True, True, False]).reset_index(drop=True)


def normalize_name(name: pd.Series | str) -> pd.Series | str:
    """Normalizza un nome di giocatore per confrontarlo tra fonti diverse.

    Il MCP scrive "Felix Auger Aliassime" e "Christopher Oconnell", dove
    TennisMyLife ha "Felix Auger-Aliassime" e "Christopher O'Connell": trattini,
    apostrofi, accenti e maiuscole differiscono sistematicamente. Senza questo
    passaggio un join sui nomi perde centinaia di match — e li perde in
    silenzio, che è il modo peggiore.
    """
    def one(x: str) -> str:
        if not isinstance(x, str):
            return ""
        # separa gli accenti dalle lettere e li scarta (é -> e)
        x = unicodedata.normalize("NFKD", x)
        x = "".join(c for c in x if not unicodedata.combining(c))
        x = x.lower()
        # Gli apostrofi si tolgono, non si sostituiscono con uno spazio:
        # il MCP scrive "Oconnell", TML "O'Connell" — devono coincidere.
        x = re.sub(r"['’]", "", x)
        x = re.sub(r"[^a-z0-9]+", " ", x)
        return x.strip()

    return name.map(one) if isinstance(name, pd.Series) else one(name)


def link_mcp_to_tml(mcp_matches: pd.DataFrame, tml: pd.DataFrame,
                    days_before: int = 3, days_after: int = 21) -> pd.DataFrame:
    """Collega i match del MCP a quelli di TennisMyLife.

    Le due fonti non condividono una chiave: si incrociano per **coppia di
    giocatori** e prossimità di data. La finestra è asimmetrica perché
    `tourney_date` è l'inizio del torneo, quindi la data del match nel MCP cade
    quasi sempre *dopo*; qualche giorno prima si concede per le edizioni in cui
    il tabellone parte in anticipo.

    I nomi passano da `normalize_name()`, perché le due fonti li scrivono in
    modo diverso (trattini, apostrofi, accenti). Chi resta comunque fuori non
    viene collegato: meglio un buco dichiarato di un accoppiamento sbagliato.
    """
    left = mcp_matches[["match_id", "date", "player_1", "player_2", "round"]].copy()
    left["pair"] = [frozenset(x) for x in zip(normalize_name(left.player_1),
                                              normalize_name(left.player_2))]

    right = tml[["tml_match_id", "tourney_date", "tourney_name", "round",
                 "winner_name", "loser_name"]].copy()
    right["pair"] = [frozenset(x) for x in zip(normalize_name(right.winner_name),
                                               normalize_name(right.loser_name))]

    merged = left.merge(right, on="pair", how="inner", suffixes=("_mcp", "_tml"))
    delta = (merged["date"] - merged["tourney_date"]).dt.days
    merged = merged[(delta >= -days_before) & (delta <= days_after)].copy()
    merged["delta_giorni"] = delta[merged.index]

    # Due giocatori possono affrontarsi due volte in poche settimane: nell'agosto
    # 2023 Alcaraz e Paul si sono incontrati in Canada (torneo dal 7) e a
    # Cincinnati (dal 14), e le finestre si sovrappongono. La data da sola sceglie
    # male — il quarto di Canada del 12 agosto è più "vicino" a Cincinnati.
    # Il turno disambigua: si preferisce sempre il candidato con lo stesso round,
    # e solo a parità si guarda la distanza in giorni.
    merged["round_diverso"] = (merged["round_mcp"].astype(str)
                               != merged["round_tml"].astype(str)).astype(int)
    merged["dist"] = merged["delta_giorni"].abs()
    merged = merged.sort_values(["round_diverso", "dist"])

    # Assegnazione uno-a-uno: scartare i doppioni solo dal lato MCP non basta,
    # perché due match annotati diversi possono rivendicare lo stesso match
    # ufficiale (stessa coppia, finestre sovrapposte). Si scorrono i candidati
    # dal migliore al peggiore e si tiene una coppia solo se entrambi i lati
    # sono ancora liberi: un abbinamento incerto in meno è preferibile a un
    # abbinamento sbagliato in più.
    used_mcp: set = set()
    used_tml: set = set()
    keep = []
    for i, mcp_id, tml_id in zip(merged.index, merged["match_id"], merged["tml_match_id"]):
        if mcp_id in used_mcp or tml_id in used_tml:
            continue
        used_mcp.add(mcp_id)
        used_tml.add(tml_id)
        keep.append(i)

    out = merged.loc[keep, ["match_id", "tml_match_id", "date", "tourney_date",
                            "tourney_name", "round_mcp", "round_tml", "delta_giorni"]]
    assert out["match_id"].is_unique and out["tml_match_id"].is_unique
    return out.reset_index(drop=True)


def combined_serve_stats(gender: str = "m", tour: str = "atp",
                         years: int | range | list[int] | None = None) -> pd.DataFrame:
    """Statistiche per giocatore-match, TennisMyLife con fallback sul MCP.

    Logica: TennisMyLife è la fonte primaria perché copre **tutti** i match ed è
    ufficiale. Dove le sue statistiche mancano — circa il 5% dei match, più
    tutto ciò che non è circuito maggiore — si ricade sulle righe `Total` del
    Match Charting Project, che sono annotate a mano ma valide.

    La colonna `source` dice sempre da dove viene ogni riga: senza quella, un
    dataset misto è impossibile da giudicare.
    """
    tml = load_tml(tour, years)          # caricata una volta sola: sono 200k match
    tml_long = tml_to_long(tml)
    complete = tml_long["serve_pts"].notna() & (tml_long["serve_pts"] > 0)
    primary = tml_long[complete].copy()
    primary["source"] = "tml"

    mcp = load_mcp_stats("Overview", gender)
    mcp_matches = load_mcp_matches(gender)
    link = link_mcp_to_tml(mcp_matches, tml)

    # Il fallback vale solo nella finestra coperta da TML: fuori da lì non c'è
    # nulla da integrare, e includerlo gonfierebbe il risultato con match di
    # tutt'altra epoca (l'errore è facile e silenzioso).
    lo, hi = tml["tourney_date"].min(), tml["tourney_date"].max()
    in_window = mcp_matches[mcp_matches["date"].between(lo, hi)]["match_id"]

    # Un match del MCP è coperto solo se il gemello TML ha davvero le statistiche.
    covered = set(link[link.tml_match_id.isin(primary.tml_match_id)]["match_id"])
    fallback = mcp[mcp["match_id"].isin(in_window) & ~mcp["match_id"].isin(covered)].copy()
    fallback = fallback.merge(
        mcp_matches[["match_id", "date", "tournament", "round", "surface", "best_of"]],
        on="match_id", how="left")
    fallback = fallback.rename(columns={"date": "tourney_date", "tournament": "tourney_name"})
    fallback["source"] = "mcp"

    out = pd.concat([primary, fallback], ignore_index=True)
    return add_serve_metrics(out)


# ------------------------------------------------------------------- processed

def load_processed(name: str, strict: bool = False) -> pd.DataFrame:
    """Legge un dataset da data/processed/, verificando che non sia stantio.

    Se un file grezzo è cambiato dopo la costruzione, il dataset non riflette
    più i dati: qui si emette un warning (o si solleva un errore con
    `strict=True`). Un dataset derivato silenziosamente vecchio è peggio di uno
    assente, perché continua a produrre numeri dall'aria plausibile.

    Costruire o ricostruire: `python3 -m lib.build --all`
    """
    from .build import is_stale  # import locale: build importa loaders

    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} non esiste. Costruiscilo con:  python3 -m lib.build {name}")

    stale, why = is_stale(name)
    if stale:
        msg = (f"il dataset '{name}' è più vecchio delle sue fonti ({why}). "
               f"Ricostruiscilo con: python3 -m lib.build {name} --force")
        if strict:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)

    return pd.read_parquet(path)


def processed_info(name: str) -> dict:
    """Il manifesto di un dataset: fonti, data di costruzione, commit."""
    mf = PROCESSED_DIR / f"{name}.json"
    if not mf.exists():
        raise FileNotFoundError(f"{mf} non esiste.")
    import json
    return json.loads(mf.read_text())


def h2h(player_a: str, player_b: str, gender: str = "m", tour: str = "atp") -> pd.DataFrame:
    """Gli scontri diretti fra due giocatori, con la copertura dell'annotazione.

    Ritorna i match **ufficiali** (TennisMyLife) con una colonna `charted` che
    dice se il Match Charting Project li ha annotati colpo per colpo, e
    `mcp_match_id` per raggiungere il dettaglio.

    È il primo passo di qualunque analisi su un confronto: senza sapere quanti
    match mancano all'annotazione, e chi li ha vinti, ogni media calcolata sul
    sottoinsieme annotato può essere distorta senza che si veda.

    I nomi passano da `normalize_name()`, quindi "Auger-Aliassime" e
    "Auger Aliassime" trovano lo stesso giocatore.
    """
    tml = load_tml(tour)
    a, b = normalize_name(player_a), normalize_name(player_b)
    w, l = normalize_name(tml["winner_name"]), normalize_name(tml["loser_name"])
    uff = tml[((w == a) & (l == b)) | ((w == b) & (l == a))].sort_values("tourney_date")

    mcp = load_mcp_matches(gender)
    coppia = {normalize_name(x) for x in (player_a, player_b)}
    ann = mcp[mcp.apply(
        lambda r: {normalize_name(r["player_1"]), normalize_name(r["player_2"])} == coppia,
        axis=1)]

    link = link_mcp_to_tml(ann, uff) if len(ann) and len(uff) else pd.DataFrame(
        columns=["match_id", "tml_match_id"])
    out = uff.merge(link[["tml_match_id", "match_id"]], on="tml_match_id",
                    how="left", validate="one_to_one")
    out = out.rename(columns={"match_id": "mcp_match_id"})
    out["charted"] = out["mcp_match_id"].notna()

    # Match annotati che non hanno un corrispettivo ufficiale: esibizioni,
    # Challenger, qualificazioni. Vanno segnalati, non nascosti.
    out.attrs["annotati_senza_ufficiale"] = int(len(ann) - out["charted"].sum())
    return out.reset_index(drop=True)
