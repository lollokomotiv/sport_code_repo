"""<Titolo dell'analisi> — vedi README.md.

Da lanciare dalla radice di tennis_project:
    python3 analyses/<matchup>/<analisi>/run.py
"""

import sys
from pathlib import Path

# parents[3] è la radice di tennis_project una volta copiato il template in
# analyses/<matchup>/<analisi>/: qui, dentro _template/, non punta dove serve.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lib import loaders  # noqa: E402


def main() -> None:
    overview = loaders.load_mcp_stats("Overview", gender="m")
    stats = loaders.add_match_context(loaders.add_serve_metrics(overview), gender="m")

    # Controllo di plausibilità prima di qualunque conclusione: se questi numeri
    # non somigliano al tennis (≈62% prime in campo, ≈64% punti vinti al
    # servizio), è inutile guardare il resto.
    print(stats[["first_in_pct", "serve_pts_won_pct", "ace_pct"]].mean().round(3))

    # ... l'analisi vera va qui.


if __name__ == "__main__":
    main()
