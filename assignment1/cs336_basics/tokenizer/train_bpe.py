from cs336_basics.pretokenization_example import find_chunk_boundaries
from .pretokenize import segment_by_special_tokens, pre_tokenize
from collections import Counter

def count_words(
    pretokens : list[str],
    word_counts : Counter 
) -> None:
    for pt in pretokens:
        if not pt:
            continue
        pt_bytes = pt.encode("utf-8")
        word = tuple(bytes([x]) for x in pt_bytes)
        word_counts[word] += 1

def count_pairs(
    word_counts : Counter,
    pair_counts : Counter,
    pair_to_words : dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
) -> None:
    for word, freq in word_counts.items():
        if freq == 0:
            continue
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
            pair_to_words.setdefault(pair, set()).add(word)

def upgrade_counts(
    merged_pair : tuple[bytes, bytes],
    word_counts : Counter,
    pair_counts : Counter,
    pair_to_words : dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
) -> None:

    p_left, p_right = merged_pair
    merged = p_left + p_right

    affected_words = list(pair_to_words.get(merged_pair, ()))
    affect_pairs = set()

    for word in affected_words:
        freq = word_counts[word]

        new_word = []
        i = 0
        while (i < len(word)):
            if i < len(word) - 1 and word[i] == p_left and word[i+1] == p_right:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)

        
        for pair in zip(word, word[1:]):
            pair_word_set = pair_to_words[pair]
            pair_word_set.discard(word)
            pair_counts[pair] -= freq
            affect_pairs.add(pair)
        
        for pair in zip(new_word, new_word[1:]):
            pair_to_words.setdefault(pair, set()).add(new_word)
            pair_counts[pair] += freq
            affect_pairs.add(pair)
        
        del word_counts[word]
        word_counts[new_word] += freq

    for pair in affect_pairs:
        if pair_counts[pair] <= 0:
            del pair_counts[pair]
            del pair_to_words[pair]

def merge(
    word_counts : Counter,
    pair_counts : Counter,
    pair_to_words : dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
) -> tuple[bytes, bytes] | None:
    if not pair_counts:
        return None

    max_freq = max(pair_counts.values())
    merged_pair = max(pair for pair, freq in pair_counts.items() if freq == max_freq)
    upgrade_counts(merged_pair, word_counts, pair_counts, pair_to_words)

    return merged_pair

def train_bpe(
    input_path : str,
    vocab_size : int,
    special_tokens : list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i : bytes([i]) for i in range(256)}
    next_id = 256

    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1
    
    word_counts = Counter()
    pair_counts = Counter()
    pair_to_words = {}

    

    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, 1, b"<|endoftext|>")
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            for s, e, is_special_token in segment_by_special_tokens(chunk, special_tokens):
                if is_special_token:
                    continue
                count_words(pre_tokenize(chunk[s : e]), word_counts)

    count_pairs(word_counts, pair_counts, pair_to_words)

    merges = []
    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        merged_pair = merge(word_counts, pair_counts, pair_to_words)
        if not merged_pair:
            break
        merges.append(merged_pair)
        vocab[next_id] = merged_pair[0] + merged_pair[1]
        next_id += 1
    
    return vocab, merges
        
        


    



