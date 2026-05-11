def chunk_document(text: str, tokenizer, max_len: int = 512, overlap: int = 64):
    """
    Splits text into chunks of token IDs, managing overlap.
    """
    if not text or not text.strip():
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    stride = max_len - overlap
    window = max_len - 2  # reserve space for CLS/SEP
    step = stride
    chunks = []
    for start in range(0, len(ids), step):
        window_ids = ids[start:start + window]
        if len(window_ids) < 16:
            break
        
        # Decode preview for database
        preview = tokenizer.decode(window_ids[:20], skip_special_tokens=True)
        
        chunks.append({
            'token_start': start, 
            'token_end': start + len(window_ids), 
            'ids': window_ids,
            'text_preview': preview
        })
        if start + window >= len(ids):
            break
    return chunks