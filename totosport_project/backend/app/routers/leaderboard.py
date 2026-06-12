import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models.user import User
from app.schemas.leaderboard import (
    LeaderboardEntry,
    MetricLeaderboardEntry,
    RoundLeaderboardEntry,
)
from app.services.leaderboard import (
    exacts_leaderboard,
    general_leaderboard,
    round_leaderboard,
    signs_leaderboard,
    tabellone_leaderboard,
)
from app.services.season import get_latest_season

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


async def _season_id(db: AsyncSession) -> uuid.UUID:
    season = await get_latest_season(db)
    if season is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Nessuna stagione")
    return season.id


@router.get("", response_model=list[LeaderboardEntry])
async def general(
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[LeaderboardEntry]:
    return await general_leaderboard(await _season_id(db), db)


@router.get("/rounds/{round_id}", response_model=list[RoundLeaderboardEntry])
async def by_round(
    round_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[RoundLeaderboardEntry]:
    return await round_leaderboard(round_id, db)


@router.get("/signs", response_model=list[MetricLeaderboardEntry])
async def signs(
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[MetricLeaderboardEntry]:
    return await signs_leaderboard(await _season_id(db), db)


@router.get("/exacts", response_model=list[MetricLeaderboardEntry])
async def exacts(
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[MetricLeaderboardEntry]:
    return await exacts_leaderboard(await _season_id(db), db)


@router.get("/tabellone", response_model=list[MetricLeaderboardEntry])
async def tabellone(
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> list[MetricLeaderboardEntry]:
    return await tabellone_leaderboard(await _season_id(db), db)
