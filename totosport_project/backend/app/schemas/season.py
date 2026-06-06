import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.models.season import SeasonStatus


class SeasonCreate(BaseModel):
    name: str
    tabellone_deadline: Optional[datetime] = None
    modification_deadline: Optional[datetime] = None


class SeasonOut(BaseModel):
    id: uuid.UUID
    name: str
    status: SeasonStatus
    tabellone_deadline: Optional[datetime]
    modification_deadline: Optional[datetime]
    created_at: datetime

    model_config = {"from_attributes": True}


class SeasonStatusUpdate(BaseModel):
    status: SeasonStatus
