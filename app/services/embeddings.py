import torch
import torch.nn.functional as F
import numpy as np

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def build_batch(chunk_ids_list, tokenizer, max_len=512):
    batch_ids = []
    _CLS_ID = tokenizer.cls_token_id
    _SEP_ID = tokenizer.sep_token_id
    _PAD_ID = tokenizer.pad_token_id
    if _PAD_ID is None:
        _PAD_ID = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    for ids in chunk_ids_list:
        full = list(ids)
        if _CLS_ID is not None:
            full = [_CLS_ID] + full
        if _SEP_ID is not None:
            full = full + [_SEP_ID]
        full = full[:max_len]
        batch_ids.append(full)
        
    L = max(len(x) for x in batch_ids)
    input_ids = torch.full((len(batch_ids), L), _PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(batch_ids), L), dtype=torch.long)
    for i, ids in enumerate(batch_ids):
        n_tok = len(ids)
        input_ids[i, :n_tok] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :n_tok] = 1
    return input_ids, attention_mask

@torch.no_grad()
def embed_chunks_list(chunks: list[dict], model, tokenizer, batch_size=16, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    n = len(chunks)
    if n == 0:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
        
    hidden = model.config.hidden_size
    out = np.zeros((n, hidden), dtype=np.float32)
    use_amp = device.type == 'cuda'
    ids_list = [c['ids'] for c in chunks]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        input_ids, attention_mask = build_batch(ids_list[start:end], tokenizer)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                pooled = mean_pool(output.last_hidden_state, attention_mask)
                pooled = F.normalize(pooled, p=2, dim=1)
        else:
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = mean_pool(output.last_hidden_state, attention_mask)
            pooled = F.normalize(pooled, p=2, dim=1)

        out[start:end] = pooled.float().cpu().numpy()
    return out