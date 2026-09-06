"""Costruzione dei dataset derivati in data/processed/.

**Perché esiste.** Non per velocità: i dati grezzi si caricano in meno di due
secondi, quindi una cache non servirebbe a niente. `data/processed/` serve a
due cose diverse:

1. **Congelare una decisione discutibile.** Collegare un match del MCP al
   corrispondente ufficiale richiede giudizio — normalizzazione dei nomi,
   finestra di date, turno come discriminante. Materializzare il risultato
   significa avere *un* collegamento rivisto una volta, invece di ogni analisi
   che se lo ricalcola con parametri leggermente diversi.

2. **Rendere ripetibile un risultato pubblicato.** TennisMyLife si aggiorna ogni
   giorno e il MCP cresce: la stessa analisi rifatta fra un mese dà numeri
   diversi. Un dataset costruito e datato è ciò che permette di ricontrollare i
   numeri di un articolo mesi dopo.

**Regole.** I file qui dentro non si modificano mai a mano e non si versionano:
si rigenerano da questo script. Ogni dataset ha accanto un manifesto `.json` con
le fonti usate, la data di costruzione e il commit del codice; se un file grezzo
cambia dopo la costruzione, il caricamento lo segnala.

CLI:
    python3 -m lib.build --list
    python3 -m lib.build --all
    python3 -m lib.build matches --force
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import loaders
from .paths import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR


def _git_commit() -> str | None:
    """Commit corrente, per sapere con che versione del codice è stato costruito."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _fingerprint(paths: list[Path]) -> list[dict]:
    """Dimensione e data di modifica delle fonti: serve a rilevare i dati stantii."""
    out = []
    for p in sorted(paths):
        if p.exists():
            st = p.stat()
            out.append({"path": str(p.relative_to(PROJECT_ROOT)),
                        "size": st.st_size, "mtime": int(st.st_mtime)})
    return out


def write_dataset(name: str, df: pd.DataFrame, sources: list[Path], note: str = "") -> Path:
    """Scrive un dataset e il suo manifesto, in modo atomico.

    La scrittura passa da un file temporaneo e poi rinomina: se il processo
    muore a metà non resta un parquet troncato che sembra valido.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dest = PROCESSED_DIR / f"{name}.parquet"
    tmp = dest.with_suffix(".parquet.tmp")

    # Il parquet rifiuta le colonne di tipo misto: è una tutela, non un intralcio
    # (nascono unendo fonti diverse, e in CSV passerebbero inosservate). Qui si
    # traduce l'errore di pyarrow in un messaggio che dice quale colonna guardare.
    try:
        df.to_parquet(tmp, index=False)
    except Exception as exc:
        mixed = [c for c in df.columns
                 if df[c].dtype == object
                 and df[c].dropna().map(type).nunique() > 1]
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"scrittura di {name} fallita: {exc}\n"
            + (f"colonne con tipi misti: {mixed}" if mixed else "nessuna colonna mista rilevata")
        ) from exc
    os.replace(tmp, dest)

    manifest = {
        "name": name,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": len(df),
        "columns": list(df.columns),
        "git_commit": _git_commit(),
        "note": note,
        "sources": _fingerprint(sources),
    }
    (PROCESSED_DIR / f"{name}.json").write_text(json.dumps(manifest, indent=2))
    print(f"  scritto {dest.name}  ({len(df):,} righe, {dest.stat().st_size / 1e6:.1f} MB)")
    return dest


# --------------------------------------------------------------------- dataset

def build_matches(gender: str = "m", tour: str = "atp") -> tuple[pd.DataFrame, list[Path], str]:
    """Indice unificato dei match: ogni incontro ufficiale, con il suo gemello MCP.

    È il dataset che congela la decisione di collegamento. `charted` dice se
    esiste l'annotazione colpo per colpo, `mcp_match_id` come raggiungerla.
    """
    tml = loaders.load_tml(tour)
    mcp = loaders.load_mcp_matches(gender)
    link = loaders.link_mcp_to_tml(mcp, tml)

    out = tml.merge(link[["tml_match_id", "match_id", "delta_giorni"]],
                    on="tml_match_id", how="left", validate="one_to_one")
    out = out.rename(columns={"match_id": "mcp_match_id"})
    out["charted"] = out["mcp_match_id"].notna()

    keep = ["tml_match_id", "tourney_id", "tourney_date", "tourney_name", "tourney_level",
            "surface", "indoor", "draw_size", "round", "best_of", "score", "minutes",
            "winner_name", "loser_name", "winner_rank", "loser_rank",
            "mcp_match_id", "charted", "delta_giorni"]
    out = out[[c for c in keep if c in out.columns]]

    sources = sorted((RAW_DIR / "tml" / tour).glob("*.csv")) + [RAW_DIR / "mcp" / f"charting-{gender}-matches.csv"]
    note = (f"{len(out):,} match ufficiali {tour.upper()}, "
            f"{int(out.charted.sum()):,} con annotazione MCP "
            f"({100 * out.charted.mean():.1f}%)")
    return out, sources, note


def build_player_matches(gender: str = "m", tour: str = "atp") -> tuple[pd.DataFrame, list[Path], str]:
    """Una riga per giocatore-match, con le metriche di servizio e risposta.

    Statistiche ufficiali dove ci sono, annotazione MCP dove mancano: la colonna
    `source` dice sempre quale delle due.
    """
    out = loaders.combined_serve_stats(gender, tour)
    sources = (sorted((RAW_DIR / "tml" / tour).glob("*.csv"))
               + [RAW_DIR / "mcp" / f"charting-{gender}-matches.csv",
                  RAW_DIR / "mcp" / f"charting-{gender}-stats-Overview.csv"])
    counts = out["source"].value_counts().to_dict()
    note = f"{len(out):,} righe giocatore-match; provenienza {counts}"
    return out, sources, note


BUILDS = {
    "matches": build_matches,
    "player_matches": build_player_matches,
}


def is_stale(name: str) -> tuple[bool, str]:
    """Il dataset è più vecchio delle sue fonti? Ritorna (stantio, motivo)."""
    mf = PROCESSED_DIR / f"{name}.json"
    if not (PROCESSED_DIR / f"{name}.parquet").exists() or not mf.exists():
        return True, "non ancora costruito"
    manifest = json.loads(mf.read_text())
    for src in manifest.get("sources", []):
        p = PROJECT_ROOT / src["path"]
        if not p.exists():
            return True, f"fonte sparita: {src['path']}"
        st = p.stat()
        if st.st_size != src["size"] or int(st.st_mtime) != src["mtime"]:
            return True, f"fonte cambiata: {src['path']}"
    return False, f"aggiornato ({manifest['built_at']}, {manifest['rows']:,} righe)"


def main() -> None:
    p = argparse.ArgumentParser(description="Costruisce i dataset in data/processed/")
    p.add_argument("steps", nargs="*", choices=list(BUILDS) + [], help="quali costruire")
    p.add_argument("--all", action="store_true", help="costruisci tutto")
    p.add_argument("--list", action="store_true", help="mostra stato e freschezza")
    p.add_argument("--force", action="store_true", help="ricostruisci anche se aggiornato")
    p.add_argument("--gender", default="m", choices=["m", "w"])
    p.add_argument("--tour", default="atp")
    args = p.parse_args()

    if args.list or (not args.steps and not args.all):
        print(f"{'dataset':<18}{'stato'}")
        print("-" * 70)
        for name in BUILDS:
            stale, why = is_stale(name)
            print(f"{name:<18}{'DA RICOSTRUIRE — ' if stale else ''}{why}")
        if not args.list:
            print("\nCostruisci con:  python3 -m lib.build --all")
        return

    steps = list(BUILDS) if args.all else args.steps
    for name in steps:
        stale, why = is_stale(name)
        if not stale and not args.force:
            print(f"{name}: {why} — salto (--force per rifarlo)")
            continue
        print(f"{name}: costruzione ({why})")
        df, sources, note = BUILDS[name](args.gender, args.tour)
        print(f"  {note}")
        write_dataset(name, df, sources, note)


if __name__ == "__main__":
    main()
