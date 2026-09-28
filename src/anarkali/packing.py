"""Torch-free packing of state, question and options into one encoder sequence."""
import json


def pack_row(row, tokenizer, max_tokens=512):
    """Keep the entire question/options; explicitly budget state with head+tail.

Returns truncation metadata. Short schemas leaving <32 state tokens are rejected
rather than turning a state-dependent decision into an option-only prediction.
"""
    if not 40 <= max_tokens <= tokenizer.model_max_length:
        raise ValueError('invalid packed token limit')
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError('packed architecture requires CLS/SEP/PAD tokenizer')
    if len(row['candidates']) < 2:
        raise ValueError('at least two candidates required')
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    question = encode(row['question'])
    options = [encode(c['text']) for c in row['candidates']]
    if any(not ids for ids in options):
        raise ValueError('empty candidate text')
    schema_size = 3 + len(question) + sum(len(x)+1 for x in options)
    budget = max_tokens - schema_size
    if budget < 32:
        raise ValueError('schema too long; needs at least 32 tokens of state budget')
    state = encode(json.dumps(row['state'], ensure_ascii=False, sort_keys=True, separators=(',', ':')))
    original_length = len(state)
    if len(state) > budget:
        left = (budget+1)//2
        state = state[:left] + state[-(budget-left):]
    ids = [tokenizer.cls_token_id] + question + [tokenizer.sep_token_id] + state + [tokenizer.sep_token_id]
    spans = []
    for option in options:
        start = len(ids)
        ids.extend(option)
        spans.append((start, len(ids)))
        ids.append(tokenizer.sep_token_id)
    return ids, spans, {'input_tokens':len(ids), 'state_tokens':original_length,
                        'state_tokens_dropped':original_length-len(state)}
