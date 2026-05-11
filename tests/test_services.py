import numpy as np
import pandas as pd
import pytest

import app.services.preprocessing as preprocessing
from app.services.chunking import chunk_document
from app.services.preprocessing import clean_text
from app.services.similarity import rank_matches


def test_clean_text_removes_noise_and_stopwords(monkeypatch):
    monkeypatch.setattr(preprocessing, "STOP", {"the", "and", "of"})

    text = "The quick, brown fox and 123!"
    assert clean_text(text) == "quick brown fox 123"


def test_chunk_document_creates_overlapping_windows():
    class DummyTokenizer:
        cls_token_id = 101
        sep_token_id = 102
        pad_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return list(range(len(text.split())))

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(f"t{item}" for item in ids)

    tokenizer = DummyTokenizer()
    text = " ".join(f"w{i}" for i in range(1000))

    chunks = chunk_document(text, tokenizer)

    assert len(chunks) == 3
    assert chunks[0]["token_start"] == 0
    assert chunks[1]["token_start"] == 448
    assert chunks[2]["token_start"] == 896


def test_rank_matches_includes_document_and_token_metadata():
    sim_matrix = np.array([[0.95, 0.20, 0.10]], dtype=np.float32)
    query_idx = [0]

    chunks_meta = pd.DataFrame(
        [
            {"doc_idx": 0, "chunk_idx": 0, "token_start": 10, "token_end": 20},
            {"doc_idx": 0, "chunk_idx": 1, "token_start": 21, "token_end": 30},
            {"doc_idx": 0, "chunk_idx": 2, "token_start": 31, "token_end": 40},
        ]
    )
    df_corpus = pd.DataFrame(
        [{"document_id": 42, "filename": "match.pdf", "year": 2024}]
    )
    query_chunks = [
        {"token_start": 1, "token_end": 11},
    ]

    results = rank_matches(
        sim_matrix=sim_matrix,
        query_idx=query_idx,
        chunks_meta=chunks_meta,
        df_corpus=df_corpus,
        match_threshold=0.9,
        top_k=2,
        skip_intro=0,
        query_chunks=query_chunks,
    )

    assert len(results) == 1
    match = results[0]
    assert match["matched_document_id"] == 42
    assert match["year"] == 2024
    assert match["mean_top_similarity"] == pytest.approx(0.575)
    assert match["top_matches"][0]["query_token_start"] == 1
    assert match["top_matches"][0]["corpus_token_start"] == 10
