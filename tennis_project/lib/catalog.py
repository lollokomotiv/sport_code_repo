"""Inventario di ciò che c'è in data/ e ricerca rapida dentro i dati.

Serve a rispondere alle due domande che ci si fa prima di iniziare un'analisi:
*quali dati ho già scaricato* e *quanto materiale c'è su questo giocatore /
torneo / superficie*, senza aprire un notebook.

CLI:
    python3 -m lib.catalog                      # cosa c'è in data/raw
    python3 -m lib.catalog --player Sinner      # match annotati di un giocatore
    python3 -m lib.catalog --tournament Wimbledon --since 2020
"""

from __future__ import annotations

import argparse

import pandas as pd

from .paths import DATA_DIR, PROCESSED_DIR, RAW_DIR


def _human(n: int) -> str:
    return f"{n / 1_000_000:.1f} MB" if n >= 1_000_000 else f"{n / 1_000:.0f} KB"


def inventory() -> None:
    """Elenca i file presenti in data/, raggruppati per fonte."""
    if not DATA_DIR.exists():
        print("data/ non esiste ancora.")
        return

    total = 0
    for source in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        files = sorted(f for f in source.rglob("*") if f.is_file() and f.name != ".gitkeep")
        if not files:
            continue
        size = sum(f.stat().st_size for f in files)
        total += size
        print(f"\ndata/raw/{source.name}/  —  {len(files)} file, {_human(size)}")
        for f in files:
            rel = f.relative_to(source)
            print(f"    {str(rel):<44} {_human(f.stat().st_size):>9}")

    processed = [f for f in PROCESSED_DIR.rglob("*") if f.is_file() and f.name != ".gitkeep"]
    if processed:
        size = sum(f.stat().st_size for f in processed)
        total += size
        print(f"\ndata/processed/  —  {len(processed)} file, {_human(size)}")
        for f in sorted(processed):
            print(f"    {f.name:<44} {_human(f.stat().st_size):>9}")

    if total == 0:
        print("Nessun dato scaricato. Parti da:  python3 -m lib.download mcp --gender m")
    else:
        print(f"\nTotale: {_human(total)} — nulla di tutto questo finisce su git (vedi .gitignore).")


def search(player: str | None = None, tournament: str | None = None,
           surface: str | None = None, since: int | None = None,
           gender: str = "m", limit: int = 15) -> pd.DataFrame:
    """Cerca nei match annotati del Match Charting Project.

    Il confronto sui nomi è per sottostringa e case-insensitive: i nomi nel MCP
    sono scritti per esteso ("Jannik Sinner") ma non sono normalizzati, quindi
    cercare un cognome è più affidabile che cercare il nome completo.
    """
    from .loaders import load_mcp_matches  # import locale: evita un ciclo

    df = load_mcp_matches(gender)

    if player:
        mask = (df["player_1"].str.contains(player, case=False, na=False)
                | df["player_2"].str.contains(player, case=False, na=False))
        df = df[mask]
    if tournament:
        df = df[df["tournament"].str.contains(tournament, case=False, na=False)]
    if surface:
        df = df[df["surface"].str.lower() == surface.lower()]
    if since:
        df = df[df["date"].dt.year >= since]

    print(f"{len(df)} match annotati corrispondono al filtro.")
    if df.empty:
        return df

    print(f"Periodo: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Superfici: {df['surface'].value_counts().to_dict()}")
    if not player:
        top = pd.concat([df["player_1"], df["player_2"]]).value_counts().head(8)
        print(f"Giocatori più presenti: {top.to_dict()}")

    print(f"\nUltimi {min(limit, len(df))} match:")
    cols = ["date", "tournament", "round", "surface", "player_1", "player_2"]
    print(df.sort_values("date", ascending=False)[cols].head(limit).to_string(index=False))
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Inventario e ricerca nei dati tennis")
    p.add_argument("--player", help="filtra i match annotati per giocatore (sottostringa)")
    p.add_argument("--tournament", help="filtra per torneo (sottostringa)")
    p.add_argument("--surface", help="Hard, Clay, Grass, Carpet")
    p.add_argument("--since", type=int, help="solo dall'anno indicato in poi")
    p.add_argument("--gender", default="m", choices=["m", "w"])
    p.add_argument("--limit", type=int, default=15)
    args = p.parse_args()

    if any([args.player, args.tournament, args.surface, args.since]):
        search(args.player, args.tournament, args.surface, args.since, args.gender, args.limit)
    else:
        inventory()


if __name__ == "__main__":
    main()
