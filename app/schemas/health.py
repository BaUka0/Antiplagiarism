from typing import Literal, Optional

from pydantic import BaseModel

HealthStatus = Literal["ok", "degraded", "error"]


class HealthComponent(BaseModel):
    status: HealthStatus
    ready: bool
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: HealthStatus
    service: str
    db: HealthComponent
    model: HealthComponent
