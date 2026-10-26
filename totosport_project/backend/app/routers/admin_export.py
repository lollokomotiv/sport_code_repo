from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import require_admin
from app.models.user import User
from app.services.export import build_leaderboards_workbook, workbook_filename
from app.services.season import get_latest_season

router = APIRouter(prefix="/admin/export", tags=["admin", "export"])

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@router.get("/leaderboards")
async def leaderboards_xlsx(
    include_schedine: bool = True,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> Response:
    """Classifiche della stagione corrente in un unico file Excel (admin)."""
    season = await get_latest_season(db)
    if season is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Nessuna stagione")
    content = await build_leaderboards_workbook(season, db, include_schedine=include_schedine)
    filename = workbook_filename(season)
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
