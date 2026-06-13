import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import require_admin
from app.models.season import Season
from app.models.user import User
from app.schemas.leaderboard import SeasonBonusResult
from app.schemas.season import (
    SeasonCreate,
    SeasonOut,
    SeasonStatusUpdate,
    SeasonUpdate,
)
from app.services.leaderboard import finalize_season
from app.services.season import create_season, get_current_season, get_season, update_status

router = APIRouter(prefix="/admin/seasons", tags=["admin", "seasons"])


@router.post("", response_model=SeasonOut, status_code=status.HTTP_201_CREATED)
async def create(
    body: SeasonCreate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> Season:
    return await create_season(body.model_dump(exclude_unset=True), db)


@router.get("", response_model=list[SeasonOut])
async def list_seasons(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> list[Season]:
    result = await db.execute(select(Season).order_by(Season.created_at.desc()))
    return list(result.scalars().all())


@router.get("/current", response_model=SeasonOut)
async def current(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> Season:
    season = await get_current_season(db)
    if season is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Nessuna stagione attiva")
    return season


@router.patch("/{season_id}", response_model=SeasonOut)
async def update_deadlines(
    season_id: uuid.UUID,
    body: SeasonUpdate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> Season:
    season = await get_season(season_id, db)
    if season is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Stagione non trovata")
    fields = body.model_dump(exclude_unset=True)
    for key, value in fields.items():
        setattr(season, key, value)
    await db.flush()
    return season


@router.patch("/{season_id}/status", response_model=SeasonOut)
async def set_status(
    season_id: uuid.UUID,
    body: SeasonStatusUpdate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> Season:
    season = await get_season(season_id, db)
    if season is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Stagione non trovata")
    return await update_status(season, body.status, db)


@router.post("/{season_id}/finalize", response_model=SeasonBonusResult)
async def finalize(
    season_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> SeasonBonusResult:
    """Assegna i 3 bonus di fine stagione (+10) e porta la stagione a 'closed'."""
    season = await get_season(season_id, db)
    if season is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Stagione non trovata")
    return await finalize_season(season, db)
