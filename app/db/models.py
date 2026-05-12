import datetime
from typing import List, Optional
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import numpy as np

class Base(DeclarativeBase):
    pass

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), index=True)
    original_path: Mapped[Optional[str]] = mapped_column(String(1024))
    source: Mapped[Optional[str]] = mapped_column(String(100))
    year: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(50), default="processing")
    total_pages: Mapped[int] = mapped_column(Integer, default=0)
    text_len: Mapped[int] = mapped_column(Integer, default=0)
    clean_word_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    __table_args__ = (
        UniqueConstraint("filename", "year", name="uq_document_filename_year"),
    )

    texts: Mapped["DocumentText"] = relationship(back_populates="document", cascade="all, delete-orphan", uselist=False)
    chunks: Mapped[List["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")

class DocumentText(Base):
    __tablename__ = "document_texts"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), unique=True)
    raw_text: Mapped[str] = mapped_column(Text)
    body_text: Mapped[str] = mapped_column(Text)
    clean_text: Mapped[str] = mapped_column(Text)

    document: Mapped["Document"] = relationship(back_populates="texts")

class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_idx: Mapped[int] = mapped_column(Integer)
    token_start: Mapped[int] = mapped_column(Integer)
    token_end: Mapped[int] = mapped_column(Integer)
    text_preview: Mapped[str] = mapped_column(String(1024))
    embedding: Mapped[np.ndarray] = mapped_column(Vector(768))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped["Document"] = relationship(back_populates="chunks")

class Check(Base):
    __tablename__ = "checks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), default="processing")
    total_pages: Mapped[int] = mapped_column(Integer, default=0)
    clean_word_count: Mapped[int] = mapped_column(Integer, default=0)
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    suspect_count: Mapped[int] = mapped_column(Integer, default=0)
    top_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    matches: Mapped[List["CheckMatch"]] = relationship(back_populates="check", cascade="all, delete-orphan")

class CheckMatch(Base):
    __tablename__ = "check_matches"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    check_id: Mapped[int] = mapped_column(ForeignKey("checks.id", ondelete="CASCADE"), index=True)
    corpus_doc_idx: Mapped[int] = mapped_column(Integer)
    matched_document_id: Mapped[Optional[int]] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"), index=True)
    matched_filename: Mapped[str] = mapped_column(String(255))
    matched_year: Mapped[Optional[int]] = mapped_column(Integer)
    matched_query_chunks: Mapped[int] = mapped_column(Integer)
    matched_corpus_chunks: Mapped[int] = mapped_column(Integer)
    n_query_chunks: Mapped[int] = mapped_column(Integer)
    n_corpus_chunks: Mapped[int] = mapped_column(Integer)
    overlap_query: Mapped[float] = mapped_column(Float)
    overlap_corpus: Mapped[float] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)
    max_similarity: Mapped[float] = mapped_column(Float)
    mean_top_similarity: Mapped[Optional[float]] = mapped_column(Float)

    check: Mapped["Check"] = relationship(back_populates="matches")
    match_chunks: Mapped[List["CheckMatchChunk"]] = relationship(back_populates="check_match", cascade="all, delete-orphan")

class CheckMatchChunk(Base):
    __tablename__ = "check_match_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    check_match_id: Mapped[int] = mapped_column(ForeignKey("check_matches.id", ondelete="CASCADE"), index=True)
    query_chunk_idx: Mapped[int] = mapped_column(Integer)
    corpus_chunk_global: Mapped[int] = mapped_column(Integer)
    corpus_chunk_idx: Mapped[int] = mapped_column(Integer)
    similarity: Mapped[float] = mapped_column(Float)
    query_token_start: Mapped[Optional[int]] = mapped_column(Integer)
    query_token_end: Mapped[Optional[int]] = mapped_column(Integer)
    corpus_token_start: Mapped[Optional[int]] = mapped_column(Integer)
    corpus_token_end: Mapped[Optional[int]] = mapped_column(Integer)

    check_match: Mapped["CheckMatch"] = relationship(back_populates="match_chunks")
