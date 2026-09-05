# Tennis — Analysis Workspace

**Not a single project: a workspace for several independent tennis analyses.**

Each study under [`analyses/`](analyses/) asks one question, answers it with the
data that can actually answer it, and states what it does not prove. They share
the download and loading layer in [`lib/`](lib/) and the data in `data/` —
nothing else. Two analyses may use different sources and reach unrelated
conclusions; that is the point.

---

## Data sources

The de-facto standard tennis dataset — Jeff Sackmann's `tennis_atp` /
`tennis_wta` repositories — **is no longer public** (checked 5 Sep 2026: HTTP
404; only the Match Charting Project remains on that account). Most tutorials
and code found online still point at it and no longer work.

What this workspace uses instead, both verified working:

| Source | What it gives | Main limitation |
|---|---|---|
| **[Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject)** | 7.5k men's + 3k women's matches annotated **shot by shot**: serve direction, rally length, winners/errors, point-by-point sequences | Hand-annotated by volunteers — **not a random sample**, skewed towards big matches |
| **[tennis-data.co.uk](http://www.tennis-data.co.uk/)** | Every ATP match since 2000 (WTA since 2007): score, rankings, **bookmaker odds** | No play statistics at all — result, score and market only |

Full detail, schemas and caveats: [`docs/fonti-dati.md`](docs/fonti-dati.md).

---

## Setup

```bash
pip install -r requirements.txt

# Match Charting Project — match list + 16 aggregated stat files (~200 MB)
python3 -m lib.download mcp --gender m

# Optional: point-by-point sequences (~130 MB more)
python3 -m lib.download mcp --gender m --points

# Match results + betting odds, one .xlsx per season
python3 -m lib.download td --tour atp --from 2015 --to 2025
```

Raw data lands in `data/raw/` and is not versioned.

## What's in there

```bash
python3 -m lib.catalog                              # inventory of data/, with sizes
python3 -m lib.catalog --player Sinner              # charted matches for a player
python3 -m lib.catalog --tournament Wimbledon --since 2020
```

## Usage

```python
from lib import loaders

overview = loaders.load_mcp_stats("Overview", gender="m")   # one row per match/player
stats = loaders.add_serve_metrics(overview)                 # serve/return percentages
stats = loaders.add_match_context(stats, gender="m")        # + date, tournament, surface

stats.groupby("surface")["ace_pct"].mean()
# Clay 0.053 | Hard 0.092 | Grass 0.102
```

That last line is also the smoke test: if the surface ordering of ace rate does
not come out clay < hard < grass, something is wrong upstream of any analysis.

## Layout

```
tennis_project/
├── lib/                 # shared: paths, download, loading, normalisation
│   ├── paths.py
│   ├── download.py      # CLI: python3 -m lib.download {mcp,td} ...
│   ├── loaders.py
│   └── catalog.py       # CLI: what is downloaded, and search inside it
├── data/
│   ├── raw/             # as downloaded (not versioned)
│   └── processed/       # reusable derived datasets (not versioned)
├── analyses/            # one folder per analysis — see analyses/README.md
│   └── _template/       # copy this to start one
├── notebooks/           # scratch exploration, not deliverables
└── docs/fonti-dati.md   # sources, schemas, limitations
```

## Status

Workspace ready and validated on real data; **no analysis published yet**. The
index in [`analyses/README.md`](analyses/README.md) lists them as they arrive.
