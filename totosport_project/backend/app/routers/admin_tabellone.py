from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import require_admin
from app.models.user import User
from app.schemas.tabellone import (
    SeasonOutcomeCreate,
    SeasonOutcomeOut,
    TablePredictionOut,
)
from app.services.season import get_current_season
from app.services.tabellone import (
    TabelloneError,
    get_outcome,
    list_tabelloni,
    score_tabellone,
    upsert_season_outcome,
)

router = APIRouter(prefix="/admin", tags=["admin", "tabellone"])


async def _require_current_season(db: AsyncSession):
    season = await get_current_season(db)
    if season is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Nessuna stagione attiva")
    return season


@router.get("/tabellone", response_model=list[TablePredictionOut])
async def all_tabelloni(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> list[TablePredictionOut]:
    season = await _require_current_season(db)
    preds = await list_tabelloni(season.id, db)
    return [TablePredictionOut.model_validate(p) for p in preds]


@router.post("/season-outcome", response_model=SeasonOutcomeOut)
async def set_outcome(
    body: SeasonOutcomeCreate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> SeasonOutcomeOut:
    season = await _require_current_season(db)
    outcome = await upsert_season_outcome(season.id, body.model_dump(exclude_unset=True), db)
    return SeasonOutcomeOut.model_validate(outcome)


@router.get("/season-outcome", response_model=SeasonOutcomeOut)
async def view_outcome(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> SeasonOutcomeOut:
    season = await _require_current_season(db)
    outcome = await get_outcome(season.id, db)
    if outcome is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Risultati stagione non inseriti")
    return SeasonOutcomeOut.model_validate(outcome)


@router.post("/tabellone/score", response_model=list[TablePredictionOut])
async def run_scoring(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> list[TablePredictionOut]:
    """Calcola e salva i punti del tabellone per tutti i giocatori della stagione."""
    season = await _require_current_season(db)
    try:
        preds = await score_tabellone(season.id, db)
    except TabelloneError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return [TablePredictionOut.model_validate(p) for p in preds]
