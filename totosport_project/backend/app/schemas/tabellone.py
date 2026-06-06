import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TabelloneBase(BaseModel):
    """
    Campi condivisi del tabellone (pronostici del giocatore).
    Tutti opzionali → consente salvataggi parziali (bozze) e, per il PATCH di
    modifica, di inviare solo i campi che cambiano (`model_dump(exclude_unset=True)`).
    """

    # Serie A
    scudetto_team: Optional[str] = None
    scudetto_points_guess: Optional[int] = None
    relegated_a_1: Optional[str] = None
    relegated_a_2: Optional[str] = None
    relegated_a_3: Optional[str] = None
    top_scorer_a: Optional[str] = None
    top_scorer_a_goals: Optional[int] = None
    # Serie B
    promoted_b_direct_1: Optional[str] = None
    promoted_b_direct_2: Optional[str] = None
    promoted_b_first_points: Optional[int] = None
    playoff_b_1: Optional[str] = None
    playoff_b_2: Optional[str] = None
    playoff_b_3: Optional[str] = None
    playoff_b_4: Optional[str] = None
    playoff_b_5: Optional[str] = None
    playoff_b_6: Optional[str] = None
    promoted_b_playoff: Optional[str] = None
    relegated_b_c_direct_1: Optional[str] = None
    relegated_b_c_direct_2: Optional[str] = None
    relegated_b_c_direct_3: Optional[str] = None
    playout_b_1: Optional[str] = None
    playout_b_2: Optional[str] = None
    relegated_b_c_playout: Optional[str] = None
    top_scorer_b: Optional[str] = None
    top_scorer_b_goals: Optional[int] = None
    # Coppe
    coppa_italia_winner: Optional[str] = None
    champions_winner: Optional[str] = None
    europa_winner: Optional[str] = None
    conference_winner: Optional[str] = None


class TablePredictionCreate(TabelloneBase):
    """POST /tabellone — compila o sovrascrive il tabellone."""


class TablePredictionModify(TabelloneBase):
    """PATCH /tabellone/me — solo i campi inviati vengono modificati."""


class TablePredictionOut(TabelloneBase):
    id: uuid.UUID
    player_id: uuid.UUID
    season_id: uuid.UUID
    mercato_penalty: int
    total_points: Optional[int] = None
    points_breakdown: Optional[dict] = None
    scored_at: Optional[datetime] = None
    submitted_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    # Calcolato dal router in base allo stato stagione (True se in 'mercato')
    is_modifiable: bool = False

    model_config = {"from_attributes": True}


# ─── Season Outcome (risultati reali, lato admin) ────────────────────────────────


class SeasonOutcomeBase(BaseModel):
    # Serie A
    scudetto_team: Optional[str] = None
    scudetto_points: Optional[int] = None
    relegated_a_1: Optional[str] = None
    relegated_a_2: Optional[str] = None
    relegated_a_3: Optional[str] = None
    top_scorer_a: Optional[str] = None
    top_scorer_a_goals: Optional[int] = None
    # Serie B
    promoted_b_direct_1: Optional[str] = None
    promoted_b_direct_2: Optional[str] = None
    promoted_b_first_points: Optional[int] = None
    playoff_b_1: Optional[str] = None
    playoff_b_2: Optional[str] = None
    playoff_b_3: Optional[str] = None
    playoff_b_4: Optional[str] = None
    playoff_b_5: Optional[str] = None
    playoff_b_6: Optional[str] = None
    promoted_b_playoff: Optional[str] = None
    playoffs_held: Optional[bool] = None
    relegated_b_c_direct_1: Optional[str] = None
    relegated_b_c_direct_2: Optional[str] = None
    relegated_b_c_direct_3: Optional[str] = None
    playout_b_1: Optional[str] = None
    playout_b_2: Optional[str] = None
    relegated_b_c_playout: Optional[str] = None
    playout_held: Optional[bool] = None
    top_scorer_b: Optional[str] = None
    top_scorer_b_goals: Optional[int] = None
    # Coppe
    coppa_italia_winner: Optional[str] = None
    champions_winner: Optional[str] = None
    europa_winner: Optional[str] = None
    conference_winner: Optional[str] = None


class SeasonOutcomeCreate(SeasonOutcomeBase):
    """POST /admin/season-outcome — inserisce/aggiorna i risultati reali."""


class SeasonOutcomeOut(SeasonOutcomeBase):
    id: uuid.UUID
    season_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
