import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import get_current_user
from app.models.user import User
from app.schemas.prediction import (
    MatchPredictionCreate,
    MatchPredictionOut,
    PlayerPredictions,
    RoundPredictionCreate,
    RoundPredictionOut,
    RoundPredictionsBundle,
    RoundPredictionsView,
)
from app.services.prediction import (
    CompetitionNotInRound,
    MatchNotFound,
    PredictionsClosed,
    RoundNotFound,
    SignResultMismatch,
    get_match_prediction_history,
    get_my_match_predictions,
    get_my_round_predictions,
    list_round_predictions_grouped,
    submit_match_prediction,
    submit_round_prediction,
)
from app.services.round import get_round, predictions_visible

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
    except SignResultMismatch as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
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


@router.get("/round/{round_id}", response_model=RoundPredictionsView)
async def round_all_predictions(
    round_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> RoundPredictionsView:
    """
    Schedine di TUTTI i giocatori per la giornata — visibili a qualunque utente
    autenticato SOLO a finestra chiusa (deadline superata o giornata completata).
    Prima della deadline: 403 (nessuna fuga di previsioni altrui).
    """
    rnd = await get_round(round_id, db)
    if rnd is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Giornata non trovata")
    if not predictions_visible(rnd):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Le schedine degli altri sono visibili solo dopo la deadline",
        )
    grouped = await list_round_predictions_grouped(round_id, db)
    return RoundPredictionsView(players=[PlayerPredictions.model_validate(g) for g in grouped])
