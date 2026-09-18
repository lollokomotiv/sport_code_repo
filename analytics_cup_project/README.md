# Analytics Cup — workbench

Spazio di preparazione alla **SkillCorner X PySport Analytics Cup**, su dati di
tracking broadcast open source.

Questa cartella non è la submission: è il workbench, cioè esplorazioni, piani e
ipotesi scartate. La submission è un fork separato del template PySport, sta in
`submission/` ed è esclusa da questo repo. Vedi [CLAUDE.md](CLAUDE.md).

## Stato

| | |
|---|---|
| Edizione target | **2027** (la 2026 si è chiusa col finale di Parigi, 5 febbraio 2026) |
| Track | *da decidere* — Research o Analyst |
| Fork submission | *non ancora creato* — il template 2027 non è ancora pubblicato |
| Dati | clonati in `sport_data/skillcorner-opendata/` |

## Struttura

```
plans/          filoni di lavoro, numerati per priorità
explorations/   notebook esplorativi, una domanda ciascuno
notes/          risorse, glossari, letteratura
submission/     fork del template PySport (gitignored, non ancora creato)
```

## Setup

```bash
python3 -m venv ~/Documents/Projects/sport_venvs/analytics_cup_project
source ~/Documents/Projects/sport_venvs/analytics_cup_project/bin/activate
pip install -r requirements.txt
```

## I dati

20 partite della **A-League australiana 2024/25**, tracking broadcast a 10 fps,
più eventi dinamici, fasi di gioco e aggregati stagionali.

```bash
git clone https://github.com/SkillCorner/opendata.git \
  ~/Documents/Projects/sport_data/skillcorner-opendata
```

I file di tracking sono su **Git LFS** (~90 MB ciascuno): servono `git lfs install`
e `git lfs pull` per scaricarli davvero. Senza, sul disco trovi solo dei puntatori
da 133 byte.

In alternativa, e obbligatoriamente nella submission, `kloppy` scarica una partita
alla volta direttamente da GitHub:

```python
from kloppy import skillcorner
dataset = skillcorner.load_open_data(match_id=1886347, coordinates="skillcorner")
```

## Riferimenti

- [Analytics Cup](https://pysport.org/analytics-cup) · [regolamento](https://pysport.org/analytics-cup/rules)
- [SkillCorner/opendata](https://github.com/SkillCorner/opendata) — dati, 17 notebook tutorial, viz tools
- Template submission: [Research](https://github.com/PySport/analytics_cup_research) · [Analyst](https://github.com/PySport/analytics_cup_analyst)
- Submission dell'edizione 2026: [presentazioni](https://skillcorner.com/event-2026/analytics-cup-2026-presentations)
- Risorse e librerie: [notes/risorse.md](notes/risorse.md)
