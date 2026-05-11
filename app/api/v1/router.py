from fastapi import APIRouter
from app.api.v1.routes import health, checks, documents

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(checks.router, prefix="/checks", tags=["checks"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])