from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.schemas.health import HealthResponse
from app.services.health import build_health_report

router = APIRouter()

@router.get("/health", summary="Readiness check", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Check whether the API, database, and plagiarism model are ready.
    """
    return await build_health_report(db)
