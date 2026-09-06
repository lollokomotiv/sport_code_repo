"""Ispezione dei file grezzi: cosa c'è dentro, prima di scriverci sopra codice.

`lib.catalog` risponde a "quali dati ho"; questo risponde a "com'è fatto questo
file". Serve soprattutto per il MCP, dove ogni file di statistiche ha una
colonna `row` con un significato diverso — la lunghezza dello scambio, il tipo
di punto, il tipo di colpo — e senza guardarla si aggrega alla cieca.

CLI:
    python3 -m lib.explore                              # tutti i file grezzi
    python3 -m lib.explore --file mcp/charting-m-stats-Rally.csv
    python3 -m lib.explore --file tml/atp/2025.csv --rows 5
    python3 -m lib.explore --grep Rally                 # file il cui nome contiene...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .paths import RAW_DIR

# Quante righe leggere per stimare lo schema senza caricare file da 56 MB.
_PEEK = 5000


def _files(pattern: str | None = None) -> list[Path]:
    out = [f for f in sorted(RAW_DIR.rglob("*.csv")) if f.is_file()]
    out += [f for f in sorted(RAW_DIR.rglob("*.xlsx")) if f.is_file()]
    if pattern:
        out = [f for f in out if pattern.lower() in str(f.relative_to(RAW_DIR)).lower()]
    return out


def _count_rows(path: Path) -> int | None:
    """Conta le righe senza caricare il file in memoria (solo CSV)."""
    if path.suffix != ".csv":
        return None
    with path.open("rb") as fh:
        return sum(1 for _ in fh) - 1


def listing(pattern: str | None = None) -> None:
    files = _files(pattern)
    if not files:
        print("Nessun file. Scarica con: python3 -m lib.download --help")
        return
    print(f"{'file':<46}{'righe':>10}{'colonne':>9}{'MB':>8}")
    print("-" * 73)
    for f in files:
        rel = str(f.relative_to(RAW_DIR))
        n = _count_rows(f)
        try:
            ncol = len(pd.read_csv(f, nrows=1).columns) if f.suffix == ".csv" else None
        except Exception:
            ncol = None
        print(f"{rel:<46}{(n if n is not None else '-'):>10}"
              f"{(ncol if ncol else '-'):>9}{f.stat().st_size / 1e6:>8.1f}")


def resolve(rel_path: str) -> Path:
    """Trova un file grezzo dal percorso relativo o da un pezzo del nome."""
    path = RAW_DIR / rel_path
    if path.exists():
        return path
    cands = _files(rel_path)
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise FileNotFoundError(f"{rel_path} non trovato sotto {RAW_DIR}")
    raise ValueError(f"{rel_path} è ambiguo: {[str(c.relative_to(RAW_DIR)) for c in cands[:8]]}")


def read(rel_path: str, **kwargs) -> pd.DataFrame:
    """Apre un file grezzo per nome, senza costruire il percorso a mano.

    Pensata per i notebook: `explore.read("Rally")` invece di ricordarsi
    `data/raw/mcp/charting-m-stats-Rally.csv`.
    """
    path = resolve(rel_path)
    if path.suffix == ".xlsx":
        return pd.read_excel(path, **kwargs)
    return pd.read_csv(path, low_memory=False, **kwargs)


def schema(rel_path_or_df: str | pd.DataFrame, max_values: int = 25) -> pd.DataFrame:
    """Lo schema di un file (o di un DataFrame) come tabella.

    Stessa informazione di `describe()`, ma restituita invece che stampata: nei
    notebook una tabella si ordina e si filtra, del testo no.
    """
    df = (read(rel_path_or_df, nrows=_PEEK) if isinstance(rel_path_or_df, str)
          else rel_path_or_df)
    rows = []
    for c in df.columns:
        s = df[c]
        nun = s.nunique(dropna=True)
        if not pd.api.types.is_numeric_dtype(s) and nun <= max_values:
            esempio = ", ".join(sorted(str(v) for v in s.dropna().unique())[:max_values])
        elif pd.api.types.is_numeric_dtype(s) and nun > 1:
            esempio = f"min {s.min():g} / mediana {s.median():g} / max {s.max():g}"
        else:
            esempio = ", ".join(str(v) for v in s.dropna().unique()[:3])
        rows.append({"colonna": c, "tipo": str(s.dtype), "distinti": nun,
                     "vuoti_%": round(100 * s.isna().mean(), 1), "valori": esempio})
    return pd.DataFrame(rows)


def describe(rel_path: str, rows: int = 3) -> None:
    """Schema, tipi, colonne dimensionali e righe di esempio di un file."""
    path = RAW_DIR / rel_path
    if not path.exists():
        cands = _files(rel_path)
        if len(cands) == 1:
            path = cands[0]
        else:
            print(f"{path} non trovato." + (f" Forse: {[str(c.relative_to(RAW_DIR)) for c in cands[:5]]}"
                                            if cands else ""))
            return

    print(f"\n{path.relative_to(RAW_DIR)}  —  {path.stat().st_size / 1e6:.1f} MB")
    df = (pd.read_csv(path, nrows=_PEEK, low_memory=False) if path.suffix == ".csv"
          else pd.read_excel(path, nrows=_PEEK))
    total = _count_rows(path)
    print(f"righe: {total if total is not None else '?'} | colonne: {len(df.columns)}"
          + (f"  (schema stimato sulle prime {_PEEK})" if total and total > _PEEK else ""))

    print("\nCOLONNE")
    for c in df.columns:
        s = df[c]
        nun = s.nunique(dropna=True)
        nulls = f"{100 * s.isna().mean():.0f}% vuoti" if s.isna().any() else ""
        # Per le colonne categoriche i valori sono più informativi del tipo:
        # è il caso di `row` nel MCP, la dimensione su cui si filtra.
        sample = ""
        # Il test è "non numerico" e non "dtype == object": da pandas 3 le
        # colonne testuali hanno dtype `str`, e il vecchio controllo le mancava
        # tutte — proprio quelle che servono, come `row`.
        if not pd.api.types.is_numeric_dtype(s) and nun <= 25:
            sample = "  " + str(sorted(str(v) for v in s.dropna().unique())[:25])
        elif pd.api.types.is_numeric_dtype(s) and nun > 1:
            sample = f"  min {s.min():g}  mediana {s.median():g}  max {s.max():g}"
        print(f"  {c:<24} {str(s.dtype):<9} {nun:>6} valori  {nulls:<12}{sample}")

    if "row" in df.columns:
        print("\nATTENZIONE: questo file ha una colonna `row`. È la dimensione su cui\n"
              "filtrare prima di aggregare — sommare righe di livelli diversi conta\n"
              "gli stessi punti più volte.")

    print(f"\nPRIME {rows} RIGHE")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(df.head(rows).to_string())


def main() -> None:
    p = argparse.ArgumentParser(description="Ispeziona i file grezzi in data/raw/")
    p.add_argument("--file", help="percorso relativo a data/raw/, o parte del nome")
    p.add_argument("--grep", help="filtra l'elenco per parte del nome")
    p.add_argument("--rows", type=int, default=3, help="righe di esempio da mostrare")
    args = p.parse_args()

    if args.file:
        describe(args.file, args.rows)
    else:
        listing(args.grep)


if __name__ == "__main__":
    main()
