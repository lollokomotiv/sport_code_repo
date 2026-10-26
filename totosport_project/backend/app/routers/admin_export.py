import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import require_admin
from app.models.user import User
from app.services.export import (
    build_general_workbook,
    build_round_workbook,
    general_filename,
    round_filename,
)
from app.services.round import get_round
from app.services.season import get_latest_season, get_season

router = APIRouter(prefix="/admin/export", tags=["admin", "export"])

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _xlsx(content: bytes, filename: str) -> Response:
    return Response(
        content=content,
        media_type=XLSX_MEDIA_TYPE,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            # Il frontend legge il nome file da qui: senza questo header CORS
            # non lo vede, essendo Vercel e Render su domini diversi.
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )


@router.get("/leaderboards")
async def leaderboards_xlsx(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> Response:
    """Classifica generale + speciali della stagione corrente (admin)."""
    season = await get_latest_season(db)
    if season is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Nessuna stagione")
    content = await build_general_workbook(season, db)
    return _xlsx(content, general_filename(season))


@router.get("/rounds/{round_id}")
async def round_xlsx(
    round_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> Response:
    """Punti e schedine di una singola giornata (admin)."""
    rnd = await get_round(round_id, db)
    if rnd is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Giornata non trovata")
    # La stagione del nome file è quella DELLA giornata, non la corrente: così
    # un export di una stagione passata non prende il nome di quella in corso.
    season = await get_season(rnd.season_id, db)
    if season is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Stagione della giornata non trovata")
    content = await build_round_workbook(rnd, db)
    return _xlsx(content, round_filename(season, rnd))
