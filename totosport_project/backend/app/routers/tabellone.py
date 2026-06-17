from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models.season import SeasonStatus
from app.models.user import User
from app.schemas.tabellone import (
    TablePredictionCreate,
    TablePredictionModify,
    TablePredictionOut,
)
from app.services.season import get_current_season
from app.services.tabellone import (
    TabelloneAlreadyCompiled,
    TabelloneNotFound,
    TabelloneNotInMercato,
    TabelloneWindowClosed,
    get_tabellone,
    modify_tabellone,
    submit_tabellone,
)

router = APIRouter(prefix="/tabellone", tags=["tabellone"])


def _to_out(pred, season) -> TablePredictionOut:
    out = TablePredictionOut.model_validate(pred)
    out.is_modifiable = season.status == SeasonStatus.mercato
    return out


async def _require_current_season(db: AsyncSession):
    season = await get_current_season(db)
    if season is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Nessuna stagione attiva")
    return season


@router.post("", response_model=TablePredictionOut)
async def submit(
    body: TablePredictionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TablePredictionOut:
    season = await _require_current_season(db)
    try:
        pred = await submit_tabellone(
            current_user.id, season, body.model_dump(exclude_unset=True), db
        )
    except TabelloneWindowClosed as e:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(e))
    except TabelloneAlreadyCompiled as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e))
    return _to_out(pred, season)


@router.get("/me", response_model=TablePredictionOut)
async def my_tabellone(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TablePredictionOut:
    season = await _require_current_season(db)
    pred = await get_tabellone(current_user.id, season.id, db)
    if pred is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Tabellone non ancora compilato")
    return _to_out(pred, season)


@router.patch("/me", response_model=TablePredictionOut)
async def modify(
    body: TablePredictionModify,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TablePredictionOut:
    season = await _require_current_season(db)
    try:
        pred, _mods = await modify_tabellone(
            current_user.id, season, body.model_dump(exclude_unset=True), db
        )
    except (TabelloneNotInMercato, TabelloneWindowClosed) as e:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(e))
    except TabelloneNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    return _to_out(pred, season)
