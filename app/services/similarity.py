import numpy as np
import pandas as pd

def calculate_similarity(query_emb: np.ndarray, corpus_emb: np.ndarray) -> np.ndarray:
    """
    Calculates the cosine similarity between query embeddings and corpus embeddings
    using dot product (assuming embeddings are normalized).
    """
    return query_emb @ corpus_emb.T

def rank_matches(sim_matrix: np.ndarray, 
                 query_idx: list[int], 
                 chunks_meta: pd.DataFrame,
                 df_corpus: pd.DataFrame,
                 match_threshold: float,
                 top_k: int = 10,
                 skip_intro: int = 2,
                 query_chunks: list[dict] | None = None) -> list[dict]:
    """
    Computes scores and ranks documents based on chunk-level similarity.
    """
    def _int_or_none(value):
        if value is None:
            return None
        if pd.isna(value):
            return None
        return int(value)

    results = []
    for doc_id in df_corpus.index:
        # Filter corpus chunks
        corpus_idx = chunks_meta.index[
            (chunks_meta['doc_idx'] == doc_id) &
            (chunks_meta['chunk_idx'] >= skip_intro)
        ].to_numpy()
        
        if len(corpus_idx) < 3:
            continue

        sub = sim_matrix[:, corpus_idx]
        best_q = sub.max(axis=1)
        best_c = sub.max(axis=0)

        matched_q = int((best_q > match_threshold).sum())
        matched_c = int((best_c > match_threshold).sum())
        overlap_q = matched_q / len(query_idx)
        overlap_c = matched_c / len(corpus_idx)
        score = max(overlap_q, overlap_c)
        max_sim = float(sub.max())

        year_value = None
        if 'year' in df_corpus.columns:
            year_value = _int_or_none(df_corpus.loc[doc_id, 'year'])

        matched_document_id = None
        if 'document_id' in df_corpus.columns:
            matched_document_id = _int_or_none(df_corpus.loc[doc_id, 'document_id'])

        if matched_q >= 1 or matched_c >= 1:
            flat = sub.flatten()
            k = min(top_k, len(flat))
            top_idx = np.argpartition(flat, -k)[-k:]
            # Sort the top k elements in descending order
            top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
            matches = []
            for idx in top_idx:
                qi = int(idx // sub.shape[1])
                ci = int(idx % sub.shape[1])
                query_chunk_idx = int(query_idx[qi])
                query_meta = None
                if query_chunks is not None and 0 <= query_chunk_idx < len(query_chunks):
                    query_meta = query_chunks[query_chunk_idx]
                corpus_row = chunks_meta.iloc[int(corpus_idx[ci])]
                matches.append({
                    'query_chunk': query_chunk_idx,
                    'corpus_chunk_global': int(corpus_idx[ci]),
                    'corpus_chunk_idx': int(corpus_row['chunk_idx']),
                    'sim': float(flat[idx]),
                    'query_token_start': _int_or_none(query_meta.get('token_start')) if query_meta else None,
                    'query_token_end': _int_or_none(query_meta.get('token_end')) if query_meta else None,
                    'corpus_token_start': _int_or_none(corpus_row['token_start']),
                    'corpus_token_end': _int_or_none(corpus_row['token_end']),
                })

            mean_top_similarity = float(np.mean([m['sim'] for m in matches])) if matches else None

            results.append({
                'doc_idx'    : doc_id,
                'matched_document_id': matched_document_id,
                'filename'   : df_corpus.loc[doc_id, 'filename'],
                'year'       : year_value,
                'matched_q'  : matched_q,
                'matched_c'  : matched_c,
                'n_chunks_q' : len(query_idx),
                'n_chunks_c' : len(corpus_idx),
                'overlap_q'  : overlap_q,
                'overlap_c'  : overlap_c,
                'score'      : score,
                'max_sim'    : max_sim,
                'mean_top_similarity': mean_top_similarity,
                'top_matches': matches,
            })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results
