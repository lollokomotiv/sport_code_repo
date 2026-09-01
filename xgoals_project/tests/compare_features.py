"""Test di regressione: le implementazioni della feature engineering devono coincidere.

La stessa feature engineering esiste in tre copie — notebook xG, notebook xA e
inference/score_match.py — e sono gia' divergute tre volte (visible_area, dtype
booleani, formula dell'angolo). Questo script le confronta sugli stessi tiri.

Riferimento: score_match.py, identico alla funzione usata per addestrare il modello xG.

    python3 tests/compare_features.py

Una colonna con "% righe diverse" > 0 e' un disallineamento da spiegare.
"""
import json, sys
from pathlib import Path
import pandas as pd

PROJ = Path("/Users/lorenzoguercio/Documents/Projects/sport_code_repo/xgoals_project")
sys.path.insert(0, str(PROJ / "inference"))
D = Path("/Users/lorenzoguercio/Documents/Projects/sport_data/open-data/data")

import score_match as SM

# --- carica le funzioni del notebook xA (celle 10 e 22) in un namespace isolato
nb = json.loads((PROJ / "xa_notebook_statsbomb.ipynb").read_text())
xa_ns = {}
for i in (10, 22):
    exec(compile(''.join(nb['cells'][i]['source']), f"<xa_cell_{i}>", "exec"), xa_ns)
xa_extract = xa_ns['extract_shot_features']

FEATS = ["distance","angle","nearest_defender_dist","keeper_dist_to_shot",
         "keeper_dist_to_goal","keeper_present","n_defenders_within_1m",
         "n_defenders_within_2m","n_defenders_within_3m",
         "n_players_in_cone_to_goal","visible_area_size",
         "first_time","one_on_one","under_pressure",
         "body_part","shot_type","shot_technique","play_pattern"]

ev_ids = {p.stem for p in (D/"events").glob("*.json")}
t360   = {p.stem for p in (D/"three-sixty").glob("*.json")}
matches = sorted(ev_ids & t360)[:25]

rows_ref, rows_xa = [], []
for mid in matches:
    events = json.loads((D/"events"/f"{mid}.json").read_text())
    th = json.loads((D/"three-sixty"/f"{mid}.json").read_text())
    frames = {f['event_uuid']: f for f in th}

    # riferimento: la stessa strada del training xG e dell'inferenza
    for r in SM.build_shot_rows(events, frames, mid, require_360=True, shot_scope="all_non_penalty"):
        rows_ref.append(r)

    # notebook xA
    for e in events:
        if e.get('type',{}).get('name') != 'Shot': continue
        if e.get('shot',{}).get('type',{}).get('name') == 'Penalty': continue
        eid = e.get('id')
        if eid not in frames: continue
        f = xa_extract(e, frames[eid])
        if f is None: continue
        rows_xa.append({'event_id': eid, **f})

ref = pd.DataFrame(rows_ref).set_index('event_id')
xa  = pd.DataFrame(rows_xa).set_index('event_id')
common = ref.index.intersection(xa.index)
ref, xa = ref.loc[common], xa.loc[common]
print(f"Tiri confrontati: {len(common):,} su {len(matches)} match\n")

print(f"{'feature':30} {'% righe diverse':>16} {'media ref':>12} {'media xA':>12}")
print("-"*74)
for c in FEATS:
    if c not in ref.columns or c not in xa.columns: continue
    a, b = ref[c], xa[c]
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        diff = ~( (a.fillna(-999) - b.fillna(-999)).abs() < 1e-6 )
        ma, mb = f"{a.mean():.3f}", f"{b.mean():.3f}"
    else:
        diff = a.astype(str) != b.astype(str)
        ma = mb = "-"
    pct = diff.mean()*100
    flag = "  <<<" if pct > 0.5 else ""
    print(f"{c:30} {pct:15.1f}% {ma:>12} {mb:>12}{flag}")

# ── Effetto end-to-end sulle predizioni del modello xG ────────────────────────
import joblib, numpy as np
XG_COLS = ["distance","angle","nearest_defender_dist","keeper_dist_to_shot",
           "keeper_dist_to_goal","keeper_present","n_defenders_within_1m",
           "n_defenders_within_2m","n_defenders_within_3m",
           "n_players_in_cone_to_goal","visible_area_size",
           "first_time","one_on_one","under_pressure",
           "body_part","shot_type","shot_technique","play_pattern"]

model = joblib.load(PROJ/"models"/"xg_model_360_no_penalty.joblib")
p_ref = model.predict_proba(ref[XG_COLS])[:,1]
p_xa  = model.predict_proba(xa.reindex(columns=XG_COLS))[:,1]

sb, goals = [], []
for mid in matches:
    for e in json.loads((D/"events"/f"{mid}.json").read_text()):
        if e.get('type',{}).get('name')=='Shot' and e.get('id') in set(common):
            sb.append((e['id'], e.get('shot',{}).get('statsbomb_xg'),
                       int(e.get('shot',{}).get('outcome',{}).get('name')=='Goal')))
sbdf = pd.DataFrame(sb, columns=['event_id','statsbomb_xg','goal']).set_index('event_id').loc[common]

print("\n" + "="*74)
print(f"{'':34}{'media xG':>12}{'corr. Pearson':>16}")
print("-"*74)
print(f"{'feature score_match (= training)':34}{p_ref.mean():12.4f}{np.corrcoef(p_ref, sbdf.statsbomb_xg)[0,1]:16.4f}")
print(f"{'feature notebook xA':34}{p_xa.mean():12.4f}{np.corrcoef(p_xa, sbdf.statsbomb_xg)[0,1]:16.4f}")
print(f"{'statsbomb_xg (benchmark)':34}{sbdf.statsbomb_xg.mean():12.4f}{1.0:16.4f}")
print(f"{'tasso gol reale':34}{sbdf.goal.mean():12.4f}")
