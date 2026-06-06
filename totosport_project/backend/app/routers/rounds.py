import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import get_current_user, require_admin
from app.models.user import User, UserRole
from app.schemas.round import (
    MatchCreate,
    MatchOut,
    RoundCreate,
    RoundDetailOut,
    RoundOut,
    RoundStatusUpdate,
)
from app.services.round import (
    PLAYER_VISIBLE_STATUSES,
    MatchNotFound,
    RoundError,
    RoundNotFound,
    StagedFixtureNotFound,
    add_match,
    create_round,
    delete_round,
    get_round,
    list_matches,
    list_rounds,
    transition_status,
)
from app.services.season import get_current_season

router = APIRouter(prefix="/rounds", tags=["rounds"])


def http_from_round_error(e: RoundError) -> HTTPException:
    """Mappa le eccezioni di dominio in HTTP: not-found → 404, conflitti → 409."""
    if isinstance(e, (RoundNotFound, MatchNotFound, StagedFixtureNotFound)):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    return HTTPException(status.HTTP_409_CONFLICT, str(e))


async def _get_round_or_404(round_id: uuid.UUID, db: AsyncSession):
    rnd = await get_round(round_id, db)
    if rnd is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Giornata non trovata")
    return rnd


@router.get("", response_model=list[RoundOut])
async def list_all(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[RoundOut]:
    season = await get_current_season(db)
    if season is None:
        return []
    # L'admin vede tutte le giornate; il giocatore solo open + completed
    statuses = None if current_user.role == UserRole.admin else PLAYER_VISIBLE_STATUSES
    rounds = await list_rounds(season.id, db, statuses=statuses, limit=limit, offset=offset)
    return [RoundOut.model_validate(r) for r in rounds]


@router.get("/{round_id}", response_model=RoundDetailOut)
async def detail(
    round_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RoundDetailOut:
    rnd = await _get_round_or_404(round_id, db)
    if current_user.role != UserRole.admin and rnd.status not in PLAYER_VISIBLE_STATUSES:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Giornata non trovata")
    matches = await list_matches(rnd.id, db)
    out = RoundDetailOut.model_validate(rnd)
    out.matches = [MatchOut.model_validate(m) for m in matches]
    return out


@router.post("", response_model=RoundOut, status_code=status.HTTP_201_CREATED)
async def create(
    body: RoundCreate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> RoundOut:
    season = await get_current_season(db)
    if season is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Nessuna stagione attiva")
    rnd = await create_round(season.id, body.model_dump(exclude_unset=True), db)
    return RoundOut.model_validate(rnd)


@router.patch("/{round_id}/status", response_model=RoundOut)
async def update_status(
    round_id: uuid.UUID,
    body: RoundStatusUpdate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> RoundOut:
    rnd = await _get_round_or_404(round_id, db)
    try:
        rnd = await transition_status(rnd, body.status, db)
    except RoundError as e:
        raise http_from_round_error(e)
    return RoundOut.model_validate(rnd)


@router.delete("/{round_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove(
    round_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> None:
    rnd = await _get_round_or_404(round_id, db)
    try:
        await delete_round(rnd, db)
    except RoundError as e:
        raise http_from_round_error(e)


@router.get("/{round_id}/matches", response_model=list[MatchOut])
async def round_matches(
    round_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[MatchOut]:
    await _get_round_or_404(round_id, db)
    matches = await list_matches(round_id, db)
    return [MatchOut.model_validate(m) for m in matches]


@router.post("/{round_id}/matches", response_model=MatchOut, status_code=status.HTTP_201_CREATED)
async def add_round_match(
    round_id: uuid.UUID,
    body: MatchCreate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> MatchOut:
    rnd = await _get_round_or_404(round_id, db)
    try:
        match = await add_match(rnd, body.model_dump(exclude_unset=True), db)
    except RoundError as e:
        raise http_from_round_error(e)
    return MatchOut.model_validate(match)
