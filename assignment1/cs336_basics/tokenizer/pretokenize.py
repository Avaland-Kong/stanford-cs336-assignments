import regex as re


PAT =  re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def segment_by_special_tokens(
    text : str, 
    special_tokens : list[str]
) -> list[tuple[int, int, bool]]:
    """
    Split text into segments by the given special tokens. 
    Each segment is returned as (start, end, is_special), where [start, end) is 
    its string indices range in text, is_special marks a special-token segment.

    Notes:
      - Segments are contiguous and together cover [0, len(text)).
      - A special-token segment is exactly one token from special_tokens,
        matched with longest-first priority when tokens overlap.
      - Ordinary segments never contain a special token and are never empty.
      - If text is empty, returns [].
    """
    if not text:
        return []
    if not special_tokens:
        return [(0, len(text), False)]

    toks = sorted([tok for tok in special_tokens if tok], key=len, reverse=True)
    if not toks:
        return [(0, len(text), False)]
    
    pat = re.compile("|".join(re.escape(tok) for tok in toks))

    result = []
    prev_end = 0
    for m in pat.finditer(text):
        start, end = m.start(), m.end()
        if start > prev_end:
            result.append((prev_end, start, False))
        result.append((start, end, True))
        prev_end = end

    if prev_end < len(text):
        result.append((prev_end, len(text), False))

    return result

def pre_tokenize(text : str) -> list[str]:
    """
    Split text into a list of pre-tokens using the regex pattern PAT.

    Notes:
      - Special tokens are not treated specially
    """
    return [m.group() for m in PAT.finditer(text)]

    




