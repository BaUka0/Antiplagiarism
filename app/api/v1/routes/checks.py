from contextlib import suppress
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.files import build_upload_path, remove_file, write_stream_to_path
from app.core.config import settings
from app.db.session import get_db
from app.schemas.checks import CheckResponse, MatchResult
from app.services.plagiarism import PlagiarismService, get_plagiarism_service

router = APIRouter()

async def _run_check_upload(
    *,
    file: UploadFile,
    db: AsyncSession,
    service: PlagiarismService,
) -> CheckResponse:
    file_path = build_upload_path(settings.UPLOAD_DIR, file.filename, kind="checks")
    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        write_stream_to_path(file.file, file_path)
        return await service.check_file(str(file_path), file.filename, db)
    finally:
        remove_file(file_path)
        with suppress(Exception):
            await file.close()


@router.post("/", response_model=CheckResponse)
async def check_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    service: PlagiarismService = Depends(get_plagiarism_service),
):
    try:
        return await _run_check_upload(file=file, db=db, service=service)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {e}")


@router.get("/{check_id}", response_model=CheckResponse)
async def get_check(
    check_id: int,
    db: AsyncSession = Depends(get_db),
    service: PlagiarismService = Depends(get_plagiarism_service),
):
    try:
        return await service.get_check_report(check_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{check_id}/matches", response_model=List[MatchResult])
async def get_check_matches(
    check_id: int,
    db: AsyncSession = Depends(get_db),
    service: PlagiarismService = Depends(get_plagiarism_service),
):
    try:
        return await service.get_check_matches(check_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
