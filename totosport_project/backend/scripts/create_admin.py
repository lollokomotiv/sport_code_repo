"""
Crea il primo utente admin da CLI.

Uso:
    cd backend
    python scripts/create_admin.py --username admin --email admin@example.com --password <pwd>
"""

import argparse
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import select

# Riusa engine/sessione dell'app: così eredita la config SSL (DB_SSL=true per Neon).
from app.database import AsyncSessionLocal, engine
from app.models.user import User, UserRole
from app.services.auth import hash_password


async def create_admin(username: str, email: str, password: str) -> None:
    async with AsyncSessionLocal() as session:
        existing = await session.execute(
            select(User).where((User.username == username) | (User.email == email))
        )
        if existing.scalar_one_or_none():
            print(f"Errore: username '{username}' o email '{email}' già in uso.")
            await engine.dispose()
            sys.exit(1)

        admin = User(
            username=username,
            email=email,
            password_hash=hash_password(password),
            role=UserRole.admin,
        )
        session.add(admin)
        await session.commit()
        print(f"Admin '{username}' creato con successo.")

    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crea utente admin")
    parser.add_argument("--username", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    asyncio.run(create_admin(args.username, args.email, args.password))
