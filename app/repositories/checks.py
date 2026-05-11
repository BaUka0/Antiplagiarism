from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Check, CheckMatch, CheckMatchChunk


class CheckRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_check(
        self,
        filename: str,
        status: str = "processing",
        total_pages: int = 0,
        clean_word_count: int = 0,
        n_chunks: int = 0,
        suspect_count: int = 0,
        top_score: float = 0.0,
        error_message: Optional[str] = None,
    ) -> Check:
        db_check = Check(
            filename=filename,
            status=status,
            total_pages=total_pages,
            clean_word_count=clean_word_count,
            n_chunks=n_chunks,
            suspect_count=suspect_count,
            top_score=top_score,
            error_message=error_message,
        )
        self.session.add(db_check)
        await self.session.flush()
        return db_check

    async def update_check(self, check_id: int, **kwargs) -> Optional[Check]:
        result = await self.session.execute(select(Check).where(Check.id == check_id))
        db_check = result.scalar_one_or_none()
        if db_check:
            for key, value in kwargs.items():
                setattr(db_check, key, value)
            await self.session.flush()
        return db_check

    async def add_match(
        self,
        check_id: int,
        corpus_doc_idx: int,
        matched_document_id: Optional[int],
        matched_filename: str,
        matched_year: Optional[int],
        matched_query_chunks: int,
        matched_corpus_chunks: int,
        n_query_chunks: int,
        n_corpus_chunks: int,
        overlap_query: float,
        overlap_corpus: float,
        score: float,
        max_similarity: float,
        mean_top_similarity: Optional[float],
    ) -> CheckMatch:
        db_match = CheckMatch(
            check_id=check_id,
            corpus_doc_idx=corpus_doc_idx,
            matched_document_id=matched_document_id,
            matched_filename=matched_filename,
            matched_year=matched_year,
            matched_query_chunks=matched_query_chunks,
            matched_corpus_chunks=matched_corpus_chunks,
            n_query_chunks=n_query_chunks,
            n_corpus_chunks=n_corpus_chunks,
            overlap_query=overlap_query,
            overlap_corpus=overlap_corpus,
            score=score,
            max_similarity=max_similarity,
            mean_top_similarity=mean_top_similarity,
        )
        self.session.add(db_match)
        await self.session.flush()
        return db_match

    async def add_match_chunk(
        self,
        check_match_id: int,
        query_chunk_idx: int,
        corpus_chunk_global: int,
        corpus_chunk_idx: int,
        similarity: float,
        query_token_start: Optional[int] = None,
        query_token_end: Optional[int] = None,
        corpus_token_start: Optional[int] = None,
        corpus_token_end: Optional[int] = None,
    ) -> CheckMatchChunk:
        db_chunk = CheckMatchChunk(
            check_match_id=check_match_id,
            query_chunk_idx=query_chunk_idx,
            corpus_chunk_global=corpus_chunk_global,
            corpus_chunk_idx=corpus_chunk_idx,
            similarity=similarity,
            query_token_start=query_token_start,
            query_token_end=query_token_end,
            corpus_token_start=corpus_token_start,
            corpus_token_end=corpus_token_end,
        )
        self.session.add(db_chunk)
        await self.session.flush()
        return db_chunk

    async def get_check(self, check_id: int) -> Optional[Check]:
        stmt = (
            select(Check)
            .where(Check.id == check_id)
            .options(selectinload(Check.matches).selectinload(CheckMatch.match_chunks))
        )
        result = await self.session.execute(stmt)
        db_check = result.scalar_one_or_none()
        if db_check:
            db_check.matches.sort(key=lambda match: match.score, reverse=True)
            for match in db_check.matches:
                match.match_chunks.sort(key=lambda chunk: chunk.similarity, reverse=True)
        return db_check
