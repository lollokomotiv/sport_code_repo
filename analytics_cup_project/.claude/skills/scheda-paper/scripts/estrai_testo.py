#!/usr/bin/env python3
"""Estrae testo dai documenti in docs/: PDF e Apple Pages.

    python3 estrai_testo.py docs/Better_Prevent_than_Tackle.pdf
    python3 estrai_testo.py docs/Research_Directions.pages
    python3 estrai_testo.py docs/paper.pdf --pagine 1-4
    python3 estrai_testo.py docs/ --elenco

I .pages non sono leggibili direttamente: il contenuto sta in `Index/*.iwa`,
protobuf compresso con Snappy raw dentro un framing proprietario (1 byte di
flag + 3 byte di lunghezza per blocco). Qui viene decompresso e ne vengono
estratte le stringhe leggibili.

L'estrazione dai .pages è **approssimativa**: gli accenti possono spezzarsi e
l'ordine dei frammenti non è garantito. Se un documento .pages conta davvero,
chiedi all'utente di esportarlo in PDF o Markdown invece di fidarti di questo.

Dipendenze: pypdf, cramjam.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path


def testo_pdf(percorso: Path, pagine: tuple[int, int] | None = None) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(percorso))
    n = len(reader.pages)
    inizio, fine = (1, n) if pagine is None else pagine
    fine = min(fine, n)

    out = [f"# {percorso.name}  ({n} pagine)"]
    for i in range(inizio - 1, fine):
        testo = (reader.pages[i].extract_text() or "").strip()
        out.append(f"\n--- pagina {i + 1} ---\n{testo}")
    return "\n".join(out)


def testo_pages(percorso: Path) -> str:
    import cramjam

    frammenti: list[str] = []
    with zipfile.ZipFile(percorso) as z:
        nomi = [n for n in z.namelist() if n.endswith(".iwa") and "Document" in n]
        for nome in nomi or [n for n in z.namelist() if n.endswith(".iwa")]:
            grezzo = z.read(nome)
            blocchi = bytearray()
            i = 0
            while i < len(grezzo) - 4:
                if grezzo[i] != 0:
                    break
                lunghezza = int.from_bytes(grezzo[i + 1:i + 4], "little")
                if lunghezza == 0 or i + 4 + lunghezza > len(grezzo):
                    break
                try:
                    blocchi += bytes(cramjam.snappy.decompress_raw(grezzo[i + 4:i + 4 + lunghezza]))
                except Exception:
                    pass
                i += 4 + lunghezza

            for f in re.findall(rb"[\x20-\x7e\xc2-\xc3][\x20-\x7e\x80-\xbf]{20,}", bytes(blocchi)):
                t = f.decode("utf-8", "replace").strip()
                if len(t) > 30 and not t.startswith(("Helvetica", "Times", "Application")):
                    frammenti.append(t)

    visti, unici = set(), []
    for t in frammenti:
        if t not in visti:
            visti.add(t)
            unici.append(t)

    return (f"# {percorso.name}  (.pages — estrazione approssimativa)\n\n"
            + "\n\n".join(unici))


def elenco(cartella: Path) -> str:
    from pypdf import PdfReader

    righe = ["# Documenti in " + str(cartella), ""]
    totale = 0
    for p in sorted(cartella.iterdir()):
        if p.suffix.lower() == ".pdf":
            try:
                n = len(PdfReader(str(p)).pages)
                totale += n
                righe.append(f"{n:4d} pp   {p.name}")
            except Exception as e:
                righe.append(f"   ?     {p.name}   ERRORE: {e}")
        elif p.suffix.lower() == ".pages":
            righe.append(f"  --      {p.name}   (Apple Pages)")
    righe.append(f"\ntotale PDF: {totale} pagine")
    return "\n".join(righe)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("percorso", type=Path)
    ap.add_argument("--pagine", help="intervallo, es. 1-4 (solo PDF)")
    ap.add_argument("--elenco", action="store_true", help="elenca i documenti di una cartella")
    args = ap.parse_args()

    if not args.percorso.exists():
        print(f"non esiste: {args.percorso}", file=sys.stderr)
        return 1

    if args.elenco or args.percorso.is_dir():
        print(elenco(args.percorso))
        return 0

    pagine = None
    if args.pagine:
        a, _, b = args.pagine.partition("-")
        pagine = (int(a), int(b or a))

    suffisso = args.percorso.suffix.lower()
    if suffisso == ".pdf":
        print(testo_pdf(args.percorso, pagine))
    elif suffisso == ".pages":
        print(testo_pages(args.percorso))
    else:
        print(f"formato non gestito: {suffisso}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
