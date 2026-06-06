import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import require_admin
from app.models.user import User
from app.routers.rounds import http_from_round_error
from app.schemas.round import MatchOut, MatchResultUpdate
from app.services.round import (
    RoundError,
    delete_match,
    get_match,
    get_round,
    set_match_result,
)

router = APIRouter(prefix="/matches", tags=["matches"])


async def _get_match_or_404(match_id: uuid.UUID, db: AsyncSession):
    match = await get_match(match_id, db)
    if match is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Partita non trovata")
    return match


@router.patch("/{match_id}/result", response_model=MatchOut)
async def set_result(
    match_id: uuid.UUID,
    body: MatchResultUpdate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> MatchOut:
    match = await _get_match_or_404(match_id, db)
    rnd = await get_round(match.round_id, db)
    try:
        match = await set_match_result(match, rnd, body.home_goals, body.away_goals, db)
    except RoundError as e:
        raise http_from_round_error(e)
    return MatchOut.model_validate(match)


@router.delete("/{match_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove(
    match_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> None:
    match = await _get_match_or_404(match_id, db)
    rnd = await get_round(match.round_id, db)
    try:
        await delete_match(match, rnd, db)
    except RoundError as e:
        raise http_from_round_error(e)
