from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.plagiarism import PlagiarismService


async def check_database(db: AsyncSession) -> dict:
    try:
        await db.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "ready": True,
            "detail": "database connection is healthy",
        }
    except Exception as exc:
        return {
            "status": "error",
            "ready": False,
            "detail": str(exc),
        }


def check_model() -> dict:
    service = PlagiarismService._instance
    if service and getattr(service, "_initialized", False):
        device = getattr(service, "device", "unknown")
        hidden_size = getattr(service, "hidden_size", 0)
        return {
            "status": "ok",
            "ready": True,
            "detail": f"model loaded on {device} (hidden_size={hidden_size})",
        }

    return {
        "status": "degraded",
        "ready": False,
        "detail": f"model service is not preloaded yet ({settings.MODEL_ID})",
    }


async def build_health_report(db: AsyncSession) -> dict:
    db_health = await check_database(db)
    model_health = check_model()

    if db_health["status"] == "error":
        overall = "error"
    elif model_health["status"] == "ok":
        overall = "ok"
    else:
        overall = "degraded"

    return {
        "status": overall,
        "service": settings.PROJECT_NAME,
        "db": db_health,
        "model": model_health,
    }
