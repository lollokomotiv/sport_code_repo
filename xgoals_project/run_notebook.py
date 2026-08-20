#!/usr/bin/env python3
"""Esegue un notebook headless, cella per cella, in un solo processo.

Nessuna dipendenza extra (niente nbconvert/papermill): legge il .ipynb,
estrae le celle di codice e le esegue in ordine in un namespace condiviso.

    # smoke test veloce (8 match, modelli in una dir scratch)
    python run_notebook.py xa_notebook_statsbomb.ipynb --limit-matches 8 \
        --models-dir /tmp/xa_smoke --figures-dir /tmp/xa_smoke/figs

    # run completo
    python run_notebook.py xa_notebook_statsbomb.ipynb --figures-dir figures/xa

    # solo un intervallo di celle (dopo aver caricato uno stato con --dump-state)
    python run_notebook.py xa_notebook_statsbomb.ipynb --from 24 --to 33
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("notebook", help="Path del .ipynb da eseguire")
    p.add_argument("--from", dest="cell_from", type=int, default=0, help="Indice prima cella (incluso)")
    p.add_argument("--to", dest="cell_to", type=int, default=None, help="Indice ultima cella (incluso)")
    p.add_argument("--limit-matches", type=int, help="Limita i match di training (smoke test)")
    p.add_argument("--models-dir", help="Override della MODELS_DIR del notebook")
    p.add_argument("--figures-dir", help="Salva ogni figura matplotlib qui invece di plt.show()")
    p.add_argument("--skip-plots", action="store_true", help="Salta le celle che contengono solo plot")
    return p.parse_args()


def load_code_cells(nb_path: Path) -> list[tuple[int, str]]:
    """Restituisce [(indice_nel_notebook, sorgente)] per le sole celle di codice."""
    nb = json.loads(nb_path.read_text())
    cells = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]).strip()
        if src:
            cells.append((i, src))
    return cells


def make_bootstrap(figures_dir: Path | None) -> str:
    """Codice eseguito prima della prima cella: backend headless + cattura figure."""
    if figures_dir is None:
        return "import matplotlib; matplotlib.use('Agg')\n"
    return f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as _plt
from pathlib import Path as _Path

_FIG_DIR = _Path({str(figures_dir)!r})
_FIG_DIR.mkdir(parents=True, exist_ok=True)
_fig_counter = {{'n': 0}}

def _show(*args, **kwargs):
    for num in _plt.get_fignums():
        _fig_counter['n'] += 1
        out = _FIG_DIR / f"fig_{{_fig_counter['n']:02d}}.png"
        _plt.figure(num).savefig(out, dpi=130, bbox_inches='tight')
        print(f"  [figura salvata] {{out}}")
    _plt.close('all')

_plt.show = _show
"""


def main() -> int:
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    nb_path = Path(args.notebook).resolve()
    if not nb_path.exists():
        print(f"Notebook non trovato: {nb_path}", file=sys.stderr)
        return 2

    # I knob sono letti dal notebook via os.environ, così le celle restano
    # identiche quando le esegui a mano in VS Code.
    if args.limit_matches:
        os.environ["XA_MAX_MATCHES"] = str(args.limit_matches)
    if args.models_dir:
        models_dir = Path(args.models_dir).resolve()
        models_dir.mkdir(parents=True, exist_ok=True)
        os.environ["XA_MODELS_DIR"] = str(models_dir)

    figures_dir = Path(args.figures_dir).resolve() if args.figures_dir else None
    cells = load_code_cells(nb_path)
    cells = [(i, s) for i, s in cells if i >= args.cell_from and (args.cell_to is None or i <= args.cell_to)]

    ns: dict = {"__name__": "__main__", "__file__": str(nb_path)}
    os.chdir(nb_path.parent)
    exec(compile(make_bootstrap(figures_dir), "<bootstrap>", "exec"), ns)

    print(f"▶ {nb_path.name} — {len(cells)} celle di codice\n")
    t_run = time.perf_counter()

    for n, (idx, src) in enumerate(cells, 1):
        if args.skip_plots and src.lstrip().startswith(("import matplotlib", "fig,")) and "plt.show()" in src:
            print(f"[{n}/{len(cells)}] cella {idx} — saltata (--skip-plots)")
            continue

        first_line = src.split("\n", 1)[0][:70]
        print(f"[{n}/{len(cells)}] cella {idx} — {first_line}")
        t0 = time.perf_counter()
        try:
            exec(compile(src, f"<cella {idx}>", "exec"), ns)
        except Exception:
            dt = time.perf_counter() - t0
            print(f"\n{'='*72}\n✗ ERRORE nella cella {idx} dopo {dt:.1f}s\n{'='*72}")
            print("--- sorgente ---")
            for ln, line in enumerate(src.split("\n"), 1):
                print(f"{ln:3d} | {line}")
            print("--- traceback ---")
            print(traceback.format_exc())
            print(f"\nVariabili definite finora: {len([k for k in ns if not k.startswith('_')])}")
            return 1
        print(f"      ✓ {time.perf_counter() - t0:.1f}s\n")

    print(f"{'='*72}\n✓ Completato in {time.perf_counter() - t_run:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
