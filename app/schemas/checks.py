from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

class MatchChunkInfo(BaseModel):
    query_chunk: int
    corpus_chunk_global: int
    corpus_chunk_idx: int
    sim: float
    query_token_start: Optional[int] = None
    query_token_end: Optional[int] = None
    corpus_token_start: Optional[int] = None
    corpus_token_end: Optional[int] = None

class MatchResult(BaseModel):
    doc_idx: int
    matched_document_id: Optional[int] = None
    filename: str
    year: Optional[int] = None
    matched_q: int
    matched_c: int
    n_chunks_q: int
    n_chunks_c: int
    overlap_q: float
    overlap_c: float
    score: float
    max_sim: float
    mean_top_similarity: Optional[float] = None
    top_matches: List[MatchChunkInfo]

class CheckResponse(BaseModel):
    check_id: Optional[int] = None
    status: Optional[str] = None
    filename: str
    n_pages: int
    n_words_clean: int
    n_chunks: int
    suspect_count: int
    top_score: Optional[float] = None
    all_results: List[MatchResult]
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
