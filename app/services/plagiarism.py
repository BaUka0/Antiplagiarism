from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from transformers import AutoModel, AutoTokenizer

from app.core.config import settings
from app.core.files import remove_file
from app.db.models import Chunk, Document
from app.repositories.checks import CheckRepository
from app.repositories.documents import DocumentRepository
from app.services.chunking import chunk_document
from app.services.embeddings import embed_chunks_list
from app.services.ocr import ocr_page
from app.services.pdf_extractor import extract_pages_pymupdf, page_is_gap
from app.services.preprocessing import clean_text, extract_body
from app.services.similarity import calculate_similarity, rank_matches


@dataclass
class CorpusSnapshot:
    embeddings: np.ndarray
    chunks_meta: pd.DataFrame
    df_corpus: pd.DataFrame


class PlagiarismService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PlagiarismService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model {settings.MODEL_ID} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID)
        self.model = AutoModel.from_pretrained(settings.MODEL_ID).to(self.device).eval()
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 0) or 0)

        self._set_snapshot(self._empty_snapshot())
        self._initialized = True

    def _empty_snapshot(self) -> CorpusSnapshot:
        return CorpusSnapshot(
            embeddings=np.zeros((0, self.hidden_size), dtype=np.float32),
            chunks_meta=pd.DataFrame(
                columns=[
                    "document_id",
                    "chunk_db_id",
                    "doc_idx",
                    "chunk_idx",
                    "token_start",
                    "token_end",
                    "text_preview",
                ]
            ),
            df_corpus=pd.DataFrame(
                columns=["document_id", "filename", "year", "source", "status"]
            ),
        )

    def _set_snapshot(self, snapshot: CorpusSnapshot) -> None:
        self.corpus_snapshot = snapshot
        self.embeddings = snapshot.embeddings
        self.chunks_meta = snapshot.chunks_meta
        self.df_corpus = snapshot.df_corpus

    async def load_corpus_from_db(self, db: AsyncSession) -> CorpusSnapshot:
        doc_stmt = select(Document).where(Document.status == "ready").order_by(Document.id)
        doc_result = await db.execute(doc_stmt)
        documents = doc_result.scalars().all()

        if not documents:
            snapshot = self._empty_snapshot()
            self._set_snapshot(snapshot)
            return snapshot

        doc_index_map: dict[int, int] = {}
        doc_records: list[dict] = []
        for idx, doc in enumerate(documents):
            doc_index_map[doc.id] = idx
            doc_records.append(
                {
                    "document_id": doc.id,
                    "filename": doc.filename,
                    "year": doc.year,
                    "source": doc.source,
                    "status": doc.status,
                }
            )

        chunk_stmt = (
            select(Chunk)
            .where(Chunk.document_id.in_(list(doc_index_map.keys())))
            .order_by(Chunk.document_id, Chunk.chunk_idx, Chunk.id)
        )
        chunk_result = await db.execute(chunk_stmt)
        chunks = chunk_result.scalars().all()

        chunk_records: list[dict] = []
        embeddings: list[np.ndarray] = []
        for chunk in chunks:
            doc_idx = doc_index_map.get(chunk.document_id)
            if doc_idx is None:
                continue

            chunk_records.append(
                {
                    "document_id": chunk.document_id,
                    "chunk_db_id": chunk.id,
                    "doc_idx": doc_idx,
                    "chunk_idx": chunk.chunk_idx,
                    "token_start": chunk.token_start,
                    "token_end": chunk.token_end,
                    "text_preview": chunk.text_preview or "",
                }
            )
            embeddings.append(np.asarray(chunk.embedding, dtype=np.float32))

        emb_array = (
            np.vstack(embeddings).astype(np.float32, copy=False)
            if embeddings
            else np.zeros((0, self.hidden_size), dtype=np.float32)
        )

        snapshot = CorpusSnapshot(
            embeddings=emb_array,
            chunks_meta=pd.DataFrame(
                chunk_records,
                columns=[
                    "document_id",
                    "chunk_db_id",
                    "doc_idx",
                    "chunk_idx",
                    "token_start",
                    "token_end",
                    "text_preview",
                ],
            ),
            df_corpus=pd.DataFrame(
                doc_records,
                columns=["document_id", "filename", "year", "source", "status"],
            ),
        )
        self._set_snapshot(snapshot)
        return snapshot

    async def index_document_background(
        self,
        doc_id: int,
        file_path: str,
        filename: str,
        year: Optional[int] = None,
        source: Optional[str] = None,
    ):
        """
        Background task version of indexing.
        """
        from app.db.session import async_session

        try:
            async with async_session() as db:
                repo = DocumentRepository(db)
                try:
                    pages_info = extract_pages_pymupdf(file_path)
                    self._apply_ocr_to_pages(file_path, filename, pages_info)

                    raw_text = "\n".join([pg.get("text", "") for pg in pages_info])
                    total_pages = len(pages_info)

                    body = extract_body([pg.get("text", "") for pg in pages_info])
                    if not body.strip():
                        body = raw_text

                    cleaned = clean_text(body)
                    wc = len(cleaned.split())

                    await repo.save_document_text(doc_id, raw_text, body, cleaned)

                    doc_chunks = chunk_document(cleaned, self.tokenizer)
                    embeddings = embed_chunks_list(doc_chunks, self.model, self.tokenizer, device=self.device)

                    await repo.save_chunks(doc_id, doc_chunks, embeddings)

                    await repo.update_document(
                        doc_id,
                        status="ready",
                        total_pages=total_pages,
                        text_len=len(cleaned),
                        clean_word_count=wc,
                    )
                    await db.commit()
                    await self.load_corpus_from_db(db)
                    print(f"Successfully indexed {filename}")
                except Exception as e:
                    print(f"Error in background indexing {filename}: {e}")
                    await repo.update_document(doc_id, status="error")
                    await db.commit()
        finally:
            remove_file(file_path)

    def _apply_ocr_to_pages(self, file_path: str, filename: str, pages_info: list[dict]) -> list[dict]:
        for i, p in enumerate(pages_info):
            gap, reason = page_is_gap(p)
            if gap:
                print(f"      - OCR page {i} for {filename} ({reason})...")
                try:
                    ocr_text = ocr_page(file_path, i)
                    if ocr_text.strip():
                        p["text"] = ocr_text
                        p["is_ocr"] = True
                except Exception as ocr_e:
                    print(f"OCR failed for {filename} page {i}: {ocr_e}")
        return pages_info

    async def _persist_check_matches(
        self,
        check_id: int,
        results: list[dict],
        snapshot: CorpusSnapshot,
        repo: CheckRepository,
    ) -> None:
        for result in results:
            doc_idx = int(result["doc_idx"])
            matched_document_id = result.get("matched_document_id")
            if matched_document_id is None and "document_id" in snapshot.df_corpus.columns:
                if 0 <= doc_idx < len(snapshot.df_corpus):
                    matched_document_id = snapshot.df_corpus.iloc[doc_idx].get("document_id")
            if matched_document_id is not None:
                matched_document_id = int(matched_document_id)

            match = await repo.add_match(
                check_id=check_id,
                corpus_doc_idx=doc_idx,
                matched_document_id=matched_document_id,
                matched_filename=str(result["filename"]),
                matched_year=result.get("year"),
                matched_query_chunks=int(result["matched_q"]),
                matched_corpus_chunks=int(result["matched_c"]),
                n_query_chunks=int(result["n_chunks_q"]),
                n_corpus_chunks=int(result["n_chunks_c"]),
                overlap_query=float(result["overlap_q"]),
                overlap_corpus=float(result["overlap_c"]),
                score=float(result["score"]),
                max_similarity=float(result["max_sim"]),
                mean_top_similarity=result.get("mean_top_similarity"),
            )

            for tm in result.get("top_matches", []):
                await repo.add_match_chunk(
                    check_match_id=match.id,
                    query_chunk_idx=int(tm["query_chunk"]),
                    corpus_chunk_global=int(tm["corpus_chunk_global"]),
                    corpus_chunk_idx=int(tm["corpus_chunk_idx"]),
                    similarity=float(tm["sim"]),
                    query_token_start=tm.get("query_token_start"),
                    query_token_end=tm.get("query_token_end"),
                    corpus_token_start=tm.get("corpus_token_start"),
                    corpus_token_end=tm.get("corpus_token_end"),
                )

    def _build_check_response(self, check, results: list[dict]) -> dict:
        return {
            "check_id": check.id,
            "status": check.status,
            "filename": check.filename,
            "n_pages": check.total_pages,
            "n_words_clean": check.clean_word_count,
            "n_chunks": check.n_chunks,
            "suspect_count": check.suspect_count,
            "top_score": check.top_score,
            "all_results": results,
            "created_at": check.created_at,
            "completed_at": check.completed_at,
            "error_message": check.error_message,
        }

    def _build_check_response_from_orm(self, check) -> dict:
        results = []
        for match in sorted(check.matches, key=lambda item: item.score, reverse=True):
            top_matches = []
            for chunk in sorted(match.match_chunks, key=lambda item: item.similarity, reverse=True):
                top_matches.append(
                    {
                        "query_chunk": chunk.query_chunk_idx,
                        "corpus_chunk_global": chunk.corpus_chunk_global,
                        "corpus_chunk_idx": chunk.corpus_chunk_idx,
                        "sim": chunk.similarity,
                        "query_token_start": chunk.query_token_start,
                        "query_token_end": chunk.query_token_end,
                        "corpus_token_start": chunk.corpus_token_start,
                        "corpus_token_end": chunk.corpus_token_end,
                    }
                )

            results.append(
                {
                    "doc_idx": match.corpus_doc_idx,
                    "matched_document_id": match.matched_document_id,
                    "filename": match.matched_filename,
                    "year": match.matched_year,
                    "matched_q": match.matched_query_chunks,
                    "matched_c": match.matched_corpus_chunks,
                    "n_chunks_q": match.n_query_chunks,
                    "n_chunks_c": match.n_corpus_chunks,
                    "overlap_q": match.overlap_query,
                    "overlap_c": match.overlap_corpus,
                    "score": match.score,
                    "max_sim": match.max_similarity,
                    "mean_top_similarity": match.mean_top_similarity,
                    "top_matches": top_matches,
                }
            )

        return {
            "check_id": check.id,
            "status": check.status,
            "filename": check.filename,
            "n_pages": check.total_pages,
            "n_words_clean": check.clean_word_count,
            "n_chunks": check.n_chunks,
            "suspect_count": check.suspect_count,
            "top_score": check.top_score,
            "all_results": results,
            "created_at": check.created_at,
            "completed_at": check.completed_at,
            "error_message": check.error_message,
        }

    async def check_file(self, file_path: str, filename: str, db: AsyncSession) -> dict:
        """
        Runs the full plagiarism check pipeline on a single PDF file.
        Persists the check result in the database.
        """
        checks_repo = CheckRepository(db)
        check = await checks_repo.create_check(filename=filename, status="processing")

        try:
            snapshot = await self.load_corpus_from_db(db)

            pages_info = extract_pages_pymupdf(file_path)
            self._apply_ocr_to_pages(file_path, filename, pages_info)
            pages_text = [pg.get("text", "") for pg in pages_info]

            if not pages_text:
                raise ValueError("No text could be extracted from the PDF.")

            body = extract_body(pages_text)
            if not body.strip():
                raise ValueError("Extracted body is empty.")

            cleaned = clean_text(body)
            wc = len(cleaned.split())
            if wc < 100:
                raise ValueError("Too little text after cleaning (less than 100 words).")

            doc_chunks = chunk_document(cleaned, self.tokenizer)
            if len(doc_chunks) < settings.SKIP_INTRO_CHUNKS + 1:
                raise ValueError("Not enough chunks for analysis.")

            query_emb = embed_chunks_list(doc_chunks, self.model, self.tokenizer, device=self.device)

            skip_intro = settings.SKIP_INTRO_CHUNKS
            query_idx = list(range(skip_intro, len(doc_chunks)))
            if not query_idx:
                query_idx = list(range(len(doc_chunks)))

            q_emb = query_emb[query_idx]

            if len(snapshot.embeddings) == 0:
                results = []
            else:
                sim_matrix = calculate_similarity(q_emb, snapshot.embeddings)
                results = rank_matches(
                    sim_matrix=sim_matrix,
                    query_idx=query_idx,
                    chunks_meta=snapshot.chunks_meta,
                    df_corpus=snapshot.df_corpus,
                    match_threshold=settings.MATCH_THRESHOLD,
                    skip_intro=skip_intro,
                    top_k=settings.TOP_K_MATCHES,
                    query_chunks=doc_chunks,
                )

            suspects = [r for r in results if r["score"] >= settings.SUSPECT_OVERLAP]
            top_score = float(max((r["score"] for r in results), default=0.0))

            await self._persist_check_matches(check.id, results, snapshot, checks_repo)
            check = await checks_repo.update_check(
                check.id,
                status="completed",
                total_pages=len(pages_info),
                clean_word_count=wc,
                n_chunks=len(doc_chunks),
                suspect_count=len(suspects),
                top_score=top_score,
                completed_at=datetime.now(timezone.utc),
                error_message=None,
            )
            await db.commit()
            return self._build_check_response(check, results)
        except ValueError as e:
            await checks_repo.update_check(
                check.id,
                status="failed",
                error_message=str(e),
                completed_at=datetime.now(timezone.utc),
            )
            await db.commit()
            raise
        except Exception as e:
            await checks_repo.update_check(
                check.id,
                status="failed",
                error_message=str(e),
                completed_at=datetime.now(timezone.utc),
            )
            await db.commit()
            raise

    async def get_check_report(self, check_id: int, db: AsyncSession) -> dict:
        repo = CheckRepository(db)
        check = await repo.get_check(check_id)
        if not check:
            raise ValueError("Check not found")
        return self._build_check_response_from_orm(check)

    async def get_check_matches(self, check_id: int, db: AsyncSession) -> list[dict]:
        report = await self.get_check_report(check_id, db)
        return report["all_results"]


def get_plagiarism_service() -> PlagiarismService:
    return PlagiarismService()
