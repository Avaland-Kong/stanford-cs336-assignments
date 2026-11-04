import multiprocessing as mp
from collections import Counter
import queue


from cs336_basics.pretokenization_example import find_chunk_boundaries
from .pretokenize import segment_by_special_tokens, pre_tokenize


_SENTINEL = "___DONE___"
_SINGLE_BYTES = tuple(bytes([i]) for i in range(256))


def count_words(
    pretokens : list[str],
    word_counts : Counter 
) -> None:
    """
    Convert each pre-token to a tuple of single-byte units and increment its frequency.

    Notes:
      - Modifies word_counts in place; returns None.
    """
    for pt in pretokens:
        if not pt:
            continue
        pt_bytes = pt.encode("utf-8")
        word = tuple(_SINGLE_BYTES[x] for x in pt_bytes)
        word_counts[word] += 1

def count_pairs(
    word_counts : Counter,
    pair_counts : Counter,
    pair_to_words : dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
) -> None:
    """
    Count weighted frequencies of adjacent pairs and build the reverse index.

    Notes:
      - Modifies pair_counts and pair_to_words in place; returns None.
    """
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
    """Merge merged_pair into a single unit in every word containing it, updating all three tables.

    Notes:
      - Modifies all three tables in place; returns None.
    """
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
    """
    Pick the most frequent pair (lexicographically largest on ties), merge it, and return it.

    Notes:
      - Returns None when no mergeable pairs left.
    """
    if not pair_counts:
        return None

    max_freq = max(pair_counts.values())
    merged_pair = max(pair for pair, freq in pair_counts.items() if freq == max_freq)
    upgrade_counts(merged_pair, word_counts, pair_counts, pair_to_words)

    return merged_pair

def _count_worker(
    task_q, 
    result_q, 
    special_tokens : list[str]
) -> None:
    """
    Consume chunk tasks from task_q, compute local word_counts, and put them into result_q.
    
    Notes:
      - Loops until it receives the sentinel, then puts the sentinel into result_q and returns.
    """
    while True:
        task = task_q.get()
        if task == _SENTINEL:
            result_q.put(_SENTINEL)
            break

        input_path, start, end= task 
        word_counts = Counter()
        
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            for s, e, is_special_token in segment_by_special_tokens(chunk, special_tokens):
                if is_special_token:
                    continue
                count_words(pre_tokenize(chunk[s : e]), word_counts)        

        result_q.put(word_counts)



def train_bpe(
    input_path : str,
    vocab_size : int,
    special_tokens : list[str],
    chunk_num : int = 256,
    worker_num : int = 8
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train byte-level BPE on a file and return (vocab, merges).

    Notes:
      - vocab_size counts the 256 base bytes, special tokens, and all merge results.
    """
    vocab = {i : bytes([i]) for i in range(256)}
    next_id = 256

    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1
    
    word_counts = Counter()
    pair_counts = Counter()
    pair_to_words = {}

    # Parallelize pre-token frequency counting
    split_special_token = special_tokens[0].encode("utf-8") if special_tokens else b'<|endoftext|>'
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, chunk_num, split_special_token)
    
    count_task_q = mp.Queue()
    count_result_q = mp.Queue()
    
    count_workers = [mp.Process(target=_count_worker, args=(count_task_q, count_result_q, special_tokens)) for _ in range(worker_num)]
    try:
        for worker in count_workers:
            try:
                worker.start()
            except Exception as e:
                raise RuntimeError(
                    f"Count_worker start failed: {e}"
                )


        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            count_task_q.put((input_path, start, end))
        
        for _ in range(worker_num):
            count_task_q.put(_SENTINEL)

        done_count = 0
        while done_count < worker_num:
            try:
                result = count_result_q.get(timeout=1)
            except queue.Empty:
                for worker in count_workers:
                    if worker.exitcode is not None and worker.exitcode != 0:
                        raise RuntimeError(
                            f"Count_worker crashed: {worker.exitcode}"
                        )
                continue
                
            if result == _SENTINEL:
                done_count += 1
                continue
            
            word_counts += result
        
        for worker in count_workers:
            worker.join()
    except Exception as e:
        for w in count_workers:
            if w.is_alive():
                w.terminate() 
        for w in count_workers:
            if w.pid is not None:
                w.join()
        raise


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
        
        


    



