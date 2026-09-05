"""Download dei dati grezzi in data/raw/. Solo stdlib, nessuna dipendenza.

Due fonti (dettagli e limiti in docs/fonti-dati.md):

  mcp  — Match Charting Project (github.com/JeffSackmann/tennis_MatchChartingProject):
         punto per punto e colpo per colpo, ~7.500 match maschili e ~3.000 femminili.
  td   — tennis-data.co.uk: risultati di tutti i match del circuito dal 2000 con
         quote dei bookmaker, un file .xlsx per stagione e per tour.

CLI:
    python3 -m lib.download mcp --gender m                  # matches + stats
    python3 -m lib.download mcp --gender m --points         # + punto per punto (pesante)
    python3 -m lib.download td  --tour atp --from 2015 --to 2025
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path

from .paths import RAW_DIR

MCP_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master"
TD_BASE = "http://www.tennis-data.co.uk"

# I file di statistiche aggregate del MCP, per genere ("m" o "w").
MCP_STATS = [
    "Overview", "ServeBasics", "ServeDirection", "ServeInfluence", "ReturnOutcomes",
    "ReturnDepth", "KeyPointsServe", "KeyPointsReturn", "Rally", "NetPoints", "SnV",
    "ShotTypes", "ShotDirection", "ShotDirOutcomes", "SvBreakSplit", "SvBreakTotal",
]
MCP_POINT_ERAS = ["to-2009", "2010s", "2020s"]

# tennis-data.co.uk rifiuta le richieste senza user agent e a volte risponde con una
# pagina HTML di cortesia al posto del file: entrambe le cose vanno gestite qui.
_UA = "Mozilla/5.0 (compatible; sport_code_repo/tennis_project)"


def _get(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _save(url: str, dest: Path, overwrite: bool = False, expect: bytes | None = None) -> Path | None:
    """Scarica url in dest. `expect` è la firma attesa dei primi byte (es. b"PK")."""
    if dest.exists() and not overwrite:
        print(f"  = {dest.name} (già presente)")
        return dest
    try:
        content = _get(url)
    except urllib.error.HTTPError as exc:
        print(f"  ! {dest.name}: HTTP {exc.code}")
        return None
    except urllib.error.URLError as exc:
        print(f"  ! {dest.name}: {exc.reason}")
        return None

    if expect and not content.startswith(expect):
        # Tipico di tennis-data.co.uk: "The page is temporarily unavailable".
        print(f"  ! {dest.name}: la risposta non è il file atteso ({len(content)} byte), riprova più tardi")
        return None

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    print(f"  + {dest.name} ({len(content) / 1_000_000:.1f} MB)")
    return dest


# --------------------------------------------------------------------------- MCP

def download_mcp(gender: str = "m", points: bool = False, stats: bool = True, overwrite: bool = False) -> list[Path]:
    """Scarica il Match Charting Project. `points=True` aggiunge ~130 MB per genere."""
    if gender not in ("m", "w"):
        raise ValueError("gender deve essere 'm' o 'w'")
    out = RAW_DIR / "mcp"
    print(f"Match Charting Project ({gender}) → {out}")

    written = []
    names = [f"charting-{gender}-matches.csv"]
    if stats:
        names += [f"charting-{gender}-stats-{s}.csv" for s in MCP_STATS]
    if points:
        names += [f"charting-{gender}-points-{era}.csv" for era in MCP_POINT_ERAS]

    for name in names:
        path = _save(f"{MCP_BASE}/{name}", out / name, overwrite)
        if path:
            written.append(path)
    return written


# ------------------------------------------------------------------ tennis-data

def download_tennis_data(tour: str = "atp", year_from: int = 2015, year_to: int = 2025,
                         overwrite: bool = False) -> list[Path]:
    """Scarica un .xlsx per stagione da tennis-data.co.uk.

    L'anno indica la stagione: il file 2024 sta in /2024/2024.xlsx.
    Dati disponibili dal 2000 (ATP) e dal 2007 (WTA).
    """
    if tour not in ("atp", "wta"):
        raise ValueError("tour deve essere 'atp' o 'wta'")
    out = RAW_DIR / "tennis-data" / tour
    print(f"tennis-data.co.uk {tour.upper()} {year_from}-{year_to} → {out}")

    written = []
    for year in range(year_from, year_to + 1):
        # I file WTA hanno il prefisso 'w' nel nome; quelli ATP no.
        name = f"{year}.xlsx" if tour == "atp" else f"{year}w.xlsx"
        path = _save(f"{TD_BASE}/{year}/{name}", out / f"{year}.xlsx", overwrite, expect=b"PK")
        if path:
            written.append(path)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Scarica dati tennis in data/raw/")
    sub = p.add_subparsers(dest="source", required=True)

    mcp = sub.add_parser("mcp", help="Match Charting Project (punto per punto)")
    mcp.add_argument("--gender", default="m", choices=["m", "w"])
    mcp.add_argument("--points", action="store_true", help="scarica anche i file punto per punto (pesanti)")
    mcp.add_argument("--no-stats", action="store_true", help="solo l'elenco match")
    mcp.add_argument("--overwrite", action="store_true")

    td = sub.add_parser("td", help="tennis-data.co.uk (risultati + quote)")
    td.add_argument("--tour", default="atp", choices=["atp", "wta"])
    td.add_argument("--from", dest="year_from", type=int, default=2015)
    td.add_argument("--to", dest="year_to", type=int, default=2025)
    td.add_argument("--overwrite", action="store_true")

    args = p.parse_args()
    if args.source == "mcp":
        download_mcp(args.gender, points=args.points, stats=not args.no_stats, overwrite=args.overwrite)
    else:
        if args.year_to < args.year_from:
            raise SystemExit("--to deve essere >= --from")
        download_tennis_data(args.tour, args.year_from, args.year_to, args.overwrite)


if __name__ == "__main__":
    main()
