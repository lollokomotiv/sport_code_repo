import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models.user import User
from app.schemas.prediction import (
    MatchPredictionCreate,
    MatchPredictionOut,
    RoundPredictionCreate,
    RoundPredictionOut,
    RoundPredictionsBundle,
)
from app.services.prediction import (
    CompetitionNotInRound,
    MatchNotFound,
    PredictionsClosed,
    RoundNotFound,
    get_match_prediction_history,
    get_my_match_predictions,
    get_my_round_predictions,
    submit_match_prediction,
    submit_round_prediction,
)

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/match", response_model=MatchPredictionOut)
async def submit_match(
    body: MatchPredictionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> MatchPredictionOut:
    try:
        pred = await submit_match_prediction(current_user.id, body.model_dump(), db)
    except MatchNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    except PredictionsClosed as e:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(e))
    return MatchPredictionOut.model_validate(pred)


@router.post("/round-goals", response_model=RoundPredictionOut)
async def submit_round_goals(
    body: RoundPredictionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RoundPredictionOut:
    try:
        pred = await submit_round_prediction(current_user.id, body.model_dump(), db)
    except RoundNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    except PredictionsClosed as e:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(e))
    except CompetitionNotInRound as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e))
    return RoundPredictionOut.model_validate(pred)


@router.get("/me", response_model=RoundPredictionsBundle)
async def my_predictions(
    round_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RoundPredictionsBundle:
    match_preds = await get_my_match_predictions(current_user.id, round_id, db)
    round_preds = await get_my_round_predictions(current_user.id, round_id, db)
    return RoundPredictionsBundle(
        match_predictions=[MatchPredictionOut.model_validate(p) for p in match_preds],
        round_predictions=[RoundPredictionOut.model_validate(p) for p in round_preds],
    )


@router.get("/me/history", response_model=list[MatchPredictionOut])
async def my_history(
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[MatchPredictionOut]:
    preds = await get_match_prediction_history(current_user.id, db, limit=limit, offset=offset)
    return [MatchPredictionOut.model_validate(p) for p in preds]
