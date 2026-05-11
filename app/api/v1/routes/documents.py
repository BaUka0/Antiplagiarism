from contextlib import suppress
from typing import Optional, List

from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.files import build_upload_path, remove_file, write_stream_to_path
from app.core.config import settings
from app.db.session import get_db
from app.services.plagiarism import get_plagiarism_service, PlagiarismService
from app.schemas.documents import DocumentRead
from app.repositories.documents import DocumentRepository

router = APIRouter()

async def _store_document_upload(
    *,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    year: Optional[int],
    source: Optional[str],
    db: AsyncSession,
    service: PlagiarismService,
) -> DocumentRead:
    file_path = None
    try:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        file_path = build_upload_path(settings.UPLOAD_DIR, file.filename, kind="documents")
        write_stream_to_path(file.file, file_path)

        repo = DocumentRepository(db)
        doc = await repo.create_document(
            filename=file.filename,
            original_path=str(file_path),
            year=year,
            source=source,
        )
        await db.commit()
        await db.refresh(doc)

        background_tasks.add_task(
            service.index_document_background,
            doc.id,
            str(file_path),
            file.filename,
            year,
            source,
        )

        return doc
    except IntegrityError:
        await db.rollback()
        remove_file(file_path)
        raise HTTPException(
            status_code=409,
            detail="A document with the same filename and year already exists.",
        )
    except HTTPException:
        await db.rollback()
        remove_file(file_path)
        raise
    except Exception as e:
        await db.rollback()
        remove_file(file_path)
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        with suppress(Exception):
            await file.close()


@router.post("/", response_model=DocumentRead)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    year: Optional[int] = Form(None),
    source: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    service: PlagiarismService = Depends(get_plagiarism_service),
):
    try:
        return await _store_document_upload(
            background_tasks=background_tasks,
            file=file,
            year=year,
            source=source,
            db=db,
            service=service,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

@router.get("/", response_model=List[DocumentRead])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    repo = DocumentRepository(db)
    return await repo.list_documents(skip=skip, limit=limit)

@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db)
):
    repo = DocumentRepository(db)
    doc = await repo.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db)
):
    repo = DocumentRepository(db)
    doc = await repo.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    await repo.delete_document(document_id)
    await db.commit()
    remove_file(doc.original_path)
    return {"status": "deleted"}
