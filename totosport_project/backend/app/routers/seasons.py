from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models.user import User
from app.schemas.season import SeasonOut
from app.services.season import get_current_season

router = APIRouter(prefix="/seasons", tags=["seasons"])


@router.get("/current", response_model=SeasonOut)
async def current(
    db: AsyncSession = Depends(get_db),
    _current_user: User = Depends(get_current_user),
) -> SeasonOut:
    """Stagione corrente, in sola lettura — accessibile a tutti i giocatori.

    Serve al frontend per sapere lo stato (setup/active/mercato/closed) e le
    deadline del tabellone, così da scegliere la vista corretta.
    """
    season = await get_current_season(db)
    if season is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Nessuna stagione attiva")
    return SeasonOut.model_validate(season)
