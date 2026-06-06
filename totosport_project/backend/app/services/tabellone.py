"""
Logica di business del Tabellone (REGOLAMENTO §2 e §7).

Funzioni pure rispetto a FastAPI: ricevono modelli/sessione e sollevano
eccezioni di DOMINIO (non HTTPException). I router le traducono in risposte HTTP.

Lo scoring (`score_tabellone`) verrà aggiunto in un secondo momento, dopo aver
chiarito alcune regole ambigue del REGOLAMENTO.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.season import Season, SeasonStatus
from app.models.season_outcome import SeasonOutcome
from app.models.table_modification import TablePredictionModification
from app.models.table_prediction import TablePrediction

PENALTY_PER_MODIFICATION = -5

# ─── Costanti punti (REGOLAMENTO §2) ──────────────────────────────────────────
PTS_SCUDETTO = 25
PTS_SCUDETTO_BONUS = 8
PTS_RELEGATED_A = 25
PTS_TOP_SCORER = 15
PTS_TOP_SCORER_BONUS = 8
PTS_PROMOTED_DIRECT = 30
PTS_PROMOTED_DIRECT_WRONG_METHOD = 15  # squadra giusta, ma promossa via playoff
PTS_PROMOTED_FIRST_BONUS = 8
PTS_PLAYOFF_TEAM = 20
PTS_PROMOTED_PLAYOFF = 30
PTS_PROMOTED_PLAYOFF_WRONG_METHOD = 15  # squadra giusta, ma promossa direttamente
PTS_NO_PLAYOFFS = 40
PTS_RELEGATED_BC = 30
PTS_PLAYOUT_TEAM = 20
PTS_RELEGATED_BC_PLAYOUT = 30
PTS_NO_PLAYOUTS = 35
PTS_COPPA = 25

# Raggruppamento dei campi in "voci" (REGOLAMENTO §7.3): la previsione principale
# e la sua domanda bonus formano un'unica voce → una sola penalità di -5.
VOICE_FIELDS: dict[str, list[str]] = {
    "scudetto": ["scudetto_team", "scudetto_points_guess"],
    "relegated_a_1": ["relegated_a_1"],
    "relegated_a_2": ["relegated_a_2"],
    "relegated_a_3": ["relegated_a_3"],
    "top_scorer_a": ["top_scorer_a", "top_scorer_a_goals"],
    "promoted_b_direct_1": ["promoted_b_direct_1", "promoted_b_first_points"],
    "promoted_b_direct_2": ["promoted_b_direct_2"],
    "playoff_b_1": ["playoff_b_1"],
    "playoff_b_2": ["playoff_b_2"],
    "playoff_b_3": ["playoff_b_3"],
    "playoff_b_4": ["playoff_b_4"],
    "playoff_b_5": ["playoff_b_5"],
    "playoff_b_6": ["playoff_b_6"],
    "promoted_b_playoff": ["promoted_b_playoff"],
    "relegated_b_c_direct_1": ["relegated_b_c_direct_1"],
    "relegated_b_c_direct_2": ["relegated_b_c_direct_2"],
    "relegated_b_c_direct_3": ["relegated_b_c_direct_3"],
    "playout_b_1": ["playout_b_1"],
    "playout_b_2": ["playout_b_2"],
    "relegated_b_c_playout": ["relegated_b_c_playout"],
    "top_scorer_b": ["top_scorer_b", "top_scorer_b_goals"],
    "coppa_italia_winner": ["coppa_italia_winner"],
    "champions_winner": ["champions_winner"],
    "europa_winner": ["europa_winner"],
    "conference_winner": ["conference_winner"],
}

# Mappa inversa campo → voce
FIELD_TO_VOICE: dict[str, str] = {
    field: voice for voice, fields in VOICE_FIELDS.items() for field in fields
}


# ─── Eccezioni di dominio ────────────────────────────────────────────────────


class TabelloneError(Exception):
    """Base per gli errori di dominio del tabellone."""


class TabelloneWindowClosed(TabelloneError):
    """La finestra di compilazione/modifica non è aperta o la deadline è passata."""


class TabelloneNotInMercato(TabelloneError):
    """La stagione non è in stato 'mercato': nessuna modifica ammessa."""


class TabelloneNotFound(TabelloneError):
    """Il giocatore non ha ancora un tabellone per questa stagione."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─── Query ───────────────────────────────────────────────────────────────────


async def get_tabellone(
    player_id: uuid.UUID, season_id: uuid.UUID, db: AsyncSession
) -> TablePrediction | None:
    result = await db.execute(
        select(TablePrediction).where(
            TablePrediction.player_id == player_id,
            TablePrediction.season_id == season_id,
        )
    )
    return result.scalar_one_or_none()


async def list_tabelloni(season_id: uuid.UUID, db: AsyncSession) -> list[TablePrediction]:
    result = await db.execute(
        select(TablePrediction).where(TablePrediction.season_id == season_id)
    )
    return list(result.scalars().all())


# ─── Season Outcome (risultati reali, lato admin) ─────────────────────────────


async def get_outcome(season_id: uuid.UUID, db: AsyncSession) -> SeasonOutcome | None:
    result = await db.execute(
        select(SeasonOutcome).where(SeasonOutcome.season_id == season_id)
    )
    return result.scalar_one_or_none()


async def upsert_season_outcome(
    season_id: uuid.UUID, data: dict, db: AsyncSession
) -> SeasonOutcome:
    """Crea o aggiorna i risultati reali della stagione (una sola riga per stagione)."""
    outcome = await get_outcome(season_id, db)
    if outcome is None:
        outcome = SeasonOutcome(season_id=season_id)
        db.add(outcome)
    for field, value in data.items():
        setattr(outcome, field, value)
    await db.flush()
    return outcome


# ─── Compilazione / aggiornamento (stato setup|active, pre-deadline) ──────────


async def submit_tabellone(
    player_id: uuid.UUID, season: Season, data: dict, db: AsyncSession
) -> TablePrediction:
    """
    Crea o sovrascrive il tabellone del giocatore. Ammesso solo se la stagione è
    in stato 'setup' o 'active' e la deadline di compilazione non è passata.
    `data` contiene solo i campi forniti (model_dump(exclude_unset=True)).
    """
    if season.status not in (SeasonStatus.setup, SeasonStatus.active):
        raise TabelloneWindowClosed("La stagione non accetta compilazioni del tabellone")
    if season.tabellone_deadline is not None and _now() > season.tabellone_deadline:
        raise TabelloneWindowClosed("Deadline di compilazione del tabellone superata")

    pred = await get_tabellone(player_id, season.id, db)
    if pred is None:
        pred = TablePrediction(player_id=player_id, season_id=season.id)
        db.add(pred)

    for field, value in data.items():
        setattr(pred, field, value)
    pred.submitted_at = _now()

    await db.flush()
    return pred


# ─── Modifica post-mercato (stato mercato, con penalità) ──────────────────────


async def modify_tabellone(
    player_id: uuid.UUID, season: Season, changes: dict, db: AsyncSession
) -> tuple[TablePrediction, list[TablePredictionModification]]:
    """
    Applica le modifiche post-mercato (REGOLAMENTO §7). Per ogni VOCE realmente
    cambiata: registra una riga in TablePredictionModification e applica -5pt.
    I campi invariati non vengono addebitati. Ritorna (tabellone, modifiche fatte).
    """
    if season.status != SeasonStatus.mercato:
        raise TabelloneNotInMercato("Le modifiche sono ammesse solo a mercato aperto")
    if season.modification_deadline is not None and _now() > season.modification_deadline:
        raise TabelloneWindowClosed("Deadline di modifica del tabellone superata")

    pred = await get_tabellone(player_id, season.id, db)
    if pred is None:
        raise TabelloneNotFound("Nessun tabellone da modificare per questa stagione")

    # Raggruppa i campi effettivamente cambiati per voce
    changed_by_voice: dict[str, list[tuple[str, object, object]]] = {}
    for field, new_value in changes.items():
        if field not in FIELD_TO_VOICE:
            continue  # ignora campi non modificabili / sconosciuti
        old_value = getattr(pred, field)
        if old_value == new_value:
            continue  # invariato → nessun addebito (REGOLAMENTO §7, punto 1)
        voice = FIELD_TO_VOICE[field]
        changed_by_voice.setdefault(voice, []).append((field, old_value, new_value))
        setattr(pred, field, new_value)

    modifications: list[TablePredictionModification] = []
    for voice, field_changes in changed_by_voice.items():
        primary_field, old_value, new_value = field_changes[0]
        mod = TablePredictionModification(
            prediction_id=pred.id,
            field_name=voice,
            old_value=str(old_value) if old_value is not None else None,
            new_value=str(new_value) if new_value is not None else None,
            penalty_points=PENALTY_PER_MODIFICATION,
        )
        db.add(mod)
        modifications.append(mod)
        pred.mercato_penalty += PENALTY_PER_MODIFICATION

    await db.flush()
    return pred, modifications


# ─── Scoring (REGOLAMENTO §2, §7) ─────────────────────────────────────────────


def _score_top_scorer(
    pick: str | None, goals_guess: int | None, actual: str | None, actual_goals: int | None
) -> int:
    """
    Capocannoniere: 15pt (divisi per il numero di co-vincitori in caso di parità,
    arrotondati per difetto) + 8pt bonus se anche i gol coincidono.
    L'admin inserisce eventuali co-vincitori separati da virgola.
    """
    if not pick or not actual:
        return 0
    winners = [n.strip() for n in actual.split(",") if n.strip()]
    if pick not in winners:
        return 0
    pts = PTS_TOP_SCORER // len(winners)
    if goals_guess is not None and goals_guess == actual_goals:
        pts += PTS_TOP_SCORER_BONUS  # bonus non diviso in caso di parità
    return pts


def _score_set_group(
    pred_values: list[str | None], actual_set: set[str], points: int, credited: set[str]
) -> list[int]:
    """Scoring 'a insiemi': ogni squadra prevista presente nell'insieme reale vale
    `points`, senza doppioni (una squadra credita un solo slot)."""
    out = []
    for v in pred_values:
        if v and v in actual_set and v not in credited:
            credited.add(v)
            out.append(points)
        else:
            out.append(0)
    return out


def score_prediction(pred: TablePrediction, outcome: SeasonOutcome) -> dict[str, int]:
    """
    Punti PIENI per voce (senza cap modifiche né penalità). Chiave = voce
    (stessa nomenclatura di VOICE_FIELDS), così i cap si applicano per voce.
    """
    b: dict[str, int] = {}

    # ── Scudetto (+ bonus punti) ──────────────────────────────────────────────
    s = 0
    if pred.scudetto_team and pred.scudetto_team == outcome.scudetto_team:
        s += PTS_SCUDETTO
        if pred.scudetto_points_guess is not None and pred.scudetto_points_guess == outcome.scudetto_points:
            s += PTS_SCUDETTO_BONUS
    b["scudetto"] = s

    # ── Retrocesse Serie A (a insiemi) ────────────────────────────────────────
    rel_a = {x for x in (outcome.relegated_a_1, outcome.relegated_a_2, outcome.relegated_a_3) if x}
    credited: set[str] = set()
    for field, pts in zip(
        ("relegated_a_1", "relegated_a_2", "relegated_a_3"),
        _score_set_group([pred.relegated_a_1, pred.relegated_a_2, pred.relegated_a_3], rel_a, PTS_RELEGATED_A, credited),
    ):
        b[field] = pts

    # ── Capocannoniere A ──────────────────────────────────────────────────────
    b["top_scorer_a"] = _score_top_scorer(
        pred.top_scorer_a, pred.top_scorer_a_goals, outcome.top_scorer_a, outcome.top_scorer_a_goals
    )

    # ── Promosse dirette B (cross-method) + bonus 1ª classificata ─────────────
    directly = {x for x in (outcome.promoted_b_direct_1, outcome.promoted_b_direct_2) if x}
    playoff_promoted = outcome.promoted_b_playoff
    credited_promo: set[str] = set()

    def _score_direct(team: str | None) -> int:
        if not team or team in credited_promo:
            return 0
        if team in directly:
            credited_promo.add(team)
            return PTS_PROMOTED_DIRECT
        if playoff_promoted and team == playoff_promoted:
            credited_promo.add(team)
            return PTS_PROMOTED_DIRECT_WRONG_METHOD
        return 0

    p1 = _score_direct(pred.promoted_b_direct_1)
    # Bonus punti 1ª classificata: legato al campo promoted_b_direct_1 = campione B
    if pred.promoted_b_direct_1 and pred.promoted_b_direct_1 == outcome.promoted_b_direct_1:
        if pred.promoted_b_first_points is not None and pred.promoted_b_first_points == outcome.promoted_b_first_points:
            p1 += PTS_PROMOTED_FIRST_BONUS
    b["promoted_b_direct_1"] = p1
    b["promoted_b_direct_2"] = _score_direct(pred.promoted_b_direct_2)

    # ── Playoff Serie B (o alternativa "no playoff") ──────────────────────────
    playoff_fields = ("playoff_b_1", "playoff_b_2", "playoff_b_3", "playoff_b_4", "playoff_b_5", "playoff_b_6")
    if outcome.playoffs_held is False:
        for f in playoff_fields:
            b[f] = 0
        b["promoted_b_playoff"] = (
            PTS_NO_PLAYOFFS
            if pred.promoted_b_playoff and pred.promoted_b_playoff == outcome.promoted_b_playoff
            else 0
        )
    else:
        playoff_set = {
            x for x in (outcome.playoff_b_1, outcome.playoff_b_2, outcome.playoff_b_3,
                        outcome.playoff_b_4, outcome.playoff_b_5, outcome.playoff_b_6) if x
        }
        credited_po: set[str] = set()
        for field, pts in zip(
            playoff_fields,
            _score_set_group(
                [getattr(pred, f) for f in playoff_fields], playoff_set, PTS_PLAYOFF_TEAM, credited_po
            ),
        ):
            b[field] = pts
        t = pred.promoted_b_playoff
        if t and playoff_promoted and t == playoff_promoted:
            b["promoted_b_playoff"] = PTS_PROMOTED_PLAYOFF
        elif t and t in directly:
            b["promoted_b_playoff"] = PTS_PROMOTED_PLAYOFF_WRONG_METHOD
        else:
            b["promoted_b_playoff"] = 0

    # ── Retrocesse B→C dirette (a insiemi) ────────────────────────────────────
    rel_bc = {x for x in (outcome.relegated_b_c_direct_1, outcome.relegated_b_c_direct_2, outcome.relegated_b_c_direct_3) if x}
    credited_bc: set[str] = set()
    for field, pts in zip(
        ("relegated_b_c_direct_1", "relegated_b_c_direct_2", "relegated_b_c_direct_3"),
        _score_set_group(
            [pred.relegated_b_c_direct_1, pred.relegated_b_c_direct_2, pred.relegated_b_c_direct_3],
            rel_bc, PTS_RELEGATED_BC, credited_bc,
        ),
    ):
        b[field] = pts

    # ── Playout Serie B (o alternativa "no playout") ──────────────────────────
    if outcome.playout_held is False:
        b["playout_b_1"] = 0
        b["playout_b_2"] = 0
        b["relegated_b_c_playout"] = (
            PTS_NO_PLAYOUTS
            if pred.relegated_b_c_playout and pred.relegated_b_c_playout == outcome.relegated_b_c_playout
            else 0
        )
    else:
        playout_set = {x for x in (outcome.playout_b_1, outcome.playout_b_2) if x}
        credited_pl: set[str] = set()
        for field, pts in zip(
            ("playout_b_1", "playout_b_2"),
            _score_set_group([pred.playout_b_1, pred.playout_b_2], playout_set, PTS_PLAYOUT_TEAM, credited_pl),
        ):
            b[field] = pts
        b["relegated_b_c_playout"] = (
            PTS_RELEGATED_BC_PLAYOUT
            if pred.relegated_b_c_playout and pred.relegated_b_c_playout == outcome.relegated_b_c_playout
            else 0
        )

    # ── Capocannoniere B ──────────────────────────────────────────────────────
    b["top_scorer_b"] = _score_top_scorer(
        pred.top_scorer_b, pred.top_scorer_b_goals, outcome.top_scorer_b, outcome.top_scorer_b_goals
    )

    # ── Coppe ─────────────────────────────────────────────────────────────────
    for field, actual in (
        ("coppa_italia_winner", outcome.coppa_italia_winner),
        ("champions_winner", outcome.champions_winner),
        ("europa_winner", outcome.europa_winner),
        ("conference_winner", outcome.conference_winner),
    ):
        guess = getattr(pred, field)
        b[field] = PTS_COPPA if guess and guess == actual else 0

    return b


async def score_tabellone(season_id: uuid.UUID, db: AsyncSession) -> list[TablePrediction]:
    """
    Calcola e salva i punti del tabellone per tutti i giocatori della stagione.
    Applica il cap al 50% sulle voci modificate post-mercato e somma la penalità
    immediata accumulata. Richiede che il SeasonOutcome sia già stato inserito.
    """
    outcome = await get_outcome(season_id, db)
    if outcome is None:
        raise TabelloneError("Risultati della stagione non ancora inseriti")

    preds = await list_tabelloni(season_id, db)
    for pred in preds:
        mod_result = await db.execute(
            select(TablePredictionModification.field_name).where(
                TablePredictionModification.prediction_id == pred.id
            )
        )
        modified_voices = set(mod_result.scalars().all())

        full = score_prediction(pred, outcome)
        # Cap al 50% (floor) sulle voci modificate; floor(x/2) == x // 2 per x >= 0
        capped = {
            voice: (pts // 2 if voice in modified_voices else pts) for voice, pts in full.items()
        }
        pred.points_breakdown = capped
        pred.total_points = sum(capped.values()) + pred.mercato_penalty
        pred.scored_at = _now()

    await db.flush()
    return preds
