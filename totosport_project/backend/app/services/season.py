"""Helper di dominio per le stagioni (creazione, stagione corrente, transizioni)."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.season import Season, SeasonStatus


async def get_current_season(db: AsyncSession) -> Season | None:
    """La stagione 'corrente' = la più recente non ancora chiusa."""
    result = await db.execute(
        select(Season)
        .where(Season.status != SeasonStatus.closed)
        .order_by(Season.created_at.desc())
    )
    return result.scalars().first()


async def get_season(season_id: uuid.UUID, db: AsyncSession) -> Season | None:
    result = await db.execute(select(Season).where(Season.id == season_id))
    return result.scalar_one_or_none()


async def create_season(data: dict, db: AsyncSession) -> Season:
    season = Season(**data)
    db.add(season)
    await db.flush()
    return season


async def update_status(season: Season, new_status: SeasonStatus, db: AsyncSession) -> Season:
    season.status = new_status
    await db.flush()
    return season
