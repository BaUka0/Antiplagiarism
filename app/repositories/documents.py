from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete
from app.db.models import Document, DocumentText, Chunk
import numpy as np

class DocumentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_document(self, filename: str, original_path: Optional[str] = None, source: Optional[str] = None, year: Optional[int] = None) -> Document:
        db_doc = Document(
            filename=filename,
            original_path=original_path,
            source=source,
            year=year,
            status="processing"
        )
        self.session.add(db_doc)
        await self.session.flush()
        return db_doc

    async def update_document(self, doc_id: int, **kwargs) -> Optional[Document]:
        result = await self.session.execute(select(Document).where(Document.id == doc_id))
        db_doc = result.scalar_one_or_none()
        if db_doc:
            for key, value in kwargs.items():
                setattr(db_doc, key, value)
            await self.session.flush()
        return db_doc

    async def save_document_text(self, doc_id: int, raw_text: str, body_text: str, clean_text: str) -> DocumentText:
        db_text = DocumentText(
            document_id=doc_id,
            raw_text=raw_text,
            body_text=body_text,
            clean_text=clean_text
        )
        self.session.add(db_text)
        await self.session.flush()
        return db_text

    async def save_chunks(self, doc_id: int, chunks: List[dict], embeddings: np.ndarray):
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            db_chunk = Chunk(
                document_id=doc_id,
                chunk_idx=i,
                token_start=chunk['token_start'],
                token_end=chunk['token_end'],
                text_preview=chunk.get('text_preview', ""),
                embedding=emb
            )
            self.session.add(db_chunk)
        await self.session.flush()

    async def get_document(self, doc_id: int) -> Optional[Document]:
        result = await self.session.execute(select(Document).where(Document.id == doc_id))
        return result.scalar_one_or_none()

    async def list_documents(self, skip: int = 0, limit: int = 100) -> List[Document]:
        result = await self.session.execute(select(Document).offset(skip).limit(limit))
        return result.scalars().all()

    async def delete_document(self, doc_id: int):
        await self.session.execute(delete(Document).where(Document.id == doc_id))
        await self.session.flush()
