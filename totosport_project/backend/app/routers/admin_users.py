import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies.auth import require_admin
from app.models.user import User
from app.schemas.user import UserOut, UserPatch
from app.services.auth import hash_password

router = APIRouter(prefix="/admin/users", tags=["admin"])


async def _get_user_or_404(user_id: uuid.UUID, db: AsyncSession) -> User:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Utente non trovato")
    return user


@router.get("", response_model=list[UserOut])
async def list_users(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
) -> list[User]:
    result = await db.execute(select(User).order_by(User.username))
    return list(result.scalars().all())


@router.patch("/{user_id}", response_model=UserOut)
async def update_user(
    user_id: uuid.UUID,
    body: UserPatch,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
) -> User:
    user = await _get_user_or_404(user_id, db)

    # username/email: verifica che non siano già usati da un ALTRO utente
    if body.username is not None or body.email is not None:
        clash = await db.execute(
            select(User).where(
                User.id != user_id,
                or_(
                    User.username == (body.username or user.username),
                    User.email == (body.email or user.email),
                ),
            )
        )
        if clash.scalar_one_or_none():
            raise HTTPException(status.HTTP_409_CONFLICT, "Username o email già in uso")

    if body.username is not None:
        user.username = body.username
    if body.email is not None:
        user.email = body.email
    if body.password is not None:
        user.password_hash = hash_password(body.password)
    if body.is_active is not None:
        if user.id == admin.id and body.is_active is False:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Non puoi disattivare il tuo account")
        user.is_active = body.is_active

    await db.flush()
    return user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
) -> None:
    """Eliminazione definitiva dell'utente (e, in cascata, dei suoi dati)."""
    user = await _get_user_or_404(user_id, db)
    if user.id == admin.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Non puoi eliminare il tuo account")
    await db.delete(user)
    await db.flush()
