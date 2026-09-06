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
404). Most tutorials and code found online still point at it and no longer work.

What this workspace uses instead, all three verified working:

| Source | What it gives | Main limitation |
|---|---|---|
| **[TennisMyLife](https://stats.tennismylife.org/tennis-match-database)** | Every tour match since 1968 (200k ATP) with **official serve statistics** — the same schema as the defunct `tennis_atp`, updated daily | Match-level totals only; nothing inside the point |
| **[Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject)** | 7.5k men's + 3k women's matches annotated **shot by shot**: serve direction, rally length, winners/errors, point-by-point sequences | Hand-annotated by volunteers — **not a random sample**, skewed towards big matches |
| **[tennis-data.co.uk](http://www.tennis-data.co.uk/)** | Every ATP match since 2000 (WTA since 2007): score, rankings, **bookmaker odds** | No play statistics at all |

The two first sources are combined by `loaders.combined_serve_stats()`: official
statistics as the primary, hand-charted rows as the fallback where the official
ones are missing (~2% of matches — Olympics, Davis Cup, juniors, Challengers).
Every row carries a `source` column saying where it came from.

Full detail, schemas and caveats: [`docs/fonti-dati.md`](docs/fonti-dati.md).

---

## Setup

```bash
pip install -r requirements.txt

# TennisMyLife — every tour match since 1968 with official serve stats (~35 MB)
python3 -m lib.download tml --tour atp

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

python3 -m lib.explore                              # every raw file: rows, columns, size
python3 -m lib.explore --file charting-m-stats-Rally.csv   # schema, dimensions, sample
python3 -m lib.explore --file tml/atp/2025.csv --rows 5
```

## Derived datasets

```bash
python3 -m lib.build --list     # what exists and whether it is stale
python3 -m lib.build --all      # (re)build into data/processed/
```

`data/processed/` is **not a cache** — the raw data loads in under two seconds.
It exists to freeze two things: the MCP↔official match linking (a judgement call
that should be reviewed once, not re-derived per analysis), and a dated snapshot
so a published result stays checkable after the sources move on.

Each dataset is a Parquet file with a `.json` manifest recording its sources,
build time and git commit. `loaders.load_processed()` warns if a source file has
changed since the build. Never edit these by hand — rebuild them.

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
│   ├── catalog.py       # CLI: what is downloaded, and search inside it
│   ├── explore.py       # CLI: schema and contents of any raw file
│   └── build.py         # CLI: build data/processed/ with provenance
├── data/
│   ├── raw/             # as downloaded (not versioned)
│   └── processed/       # reusable derived datasets (not versioned)
├── analyses/            # one folder per analysis — see analyses/README.md
│   └── _template/       # copy this to start one
├── notebooks/           # scratch exploration — start here to look at the data
│   ├── 01-esplorare-i-file.ipynb
│   └── 02-query-di-esempio.ipynb
└── docs/fonti-dati.md   # sources, schemas, limitations
```

## Status

Workspace ready and validated on real data; **no analysis published yet**. The
index in [`analyses/README.md`](analyses/README.md) lists them as they arrive.
