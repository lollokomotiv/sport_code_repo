import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import require_admin
from app.models.user import User
from app.schemas.prediction import (
    MatchPredictionOut,
    RoundPredictionOut,
    RoundPredictionsBundle,
)
from app.services.prediction import (
    list_match_predictions,
    list_round_match_predictions,
    list_round_total_goals_predictions,
)
from app.services.round import get_round

router = APIRouter(prefix="/admin/predictions", tags=["admin", "predictions"])


async def _require_round_after_deadline(round_id: uuid.UUID, db: AsyncSession):
    rnd = await get_round(round_id, db)
    if rnd is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Giornata non trovata")
    if rnd.deadline is not None and datetime.now(timezone.utc) < rnd.deadline:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Le previsioni dei giocatori sono visibili solo dopo la deadline",
        )
    return rnd


@router.get("/{round_id}", response_model=RoundPredictionsBundle)
async def round_predictions(
    round_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> RoundPredictionsBundle:
    await _require_round_after_deadline(round_id, db)
    match_preds = await list_round_match_predictions(round_id, db)
    round_preds = await list_round_total_goals_predictions(round_id, db)
    return RoundPredictionsBundle(
        match_predictions=[MatchPredictionOut.model_validate(p) for p in match_preds],
        round_predictions=[RoundPredictionOut.model_validate(p) for p in round_preds],
    )


@router.get("/{round_id}/{match_id}", response_model=list[MatchPredictionOut])
async def match_predictions(
    round_id: uuid.UUID,
    match_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> list[MatchPredictionOut]:
    await _require_round_after_deadline(round_id, db)
    preds = await list_match_predictions(match_id, db)
    return [MatchPredictionOut.model_validate(p) for p in preds]
