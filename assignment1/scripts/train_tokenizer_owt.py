from datetime import timedelta
from pathlib import Path
import pickle
import threading
import time
import psutil
import os

from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.tokenizer.train_bpe import train_bpe


_CHUNK_NUM = 256
_WORKER_NUM = 8
_peak_tree_rss_mb = 0.0
_stop_sampling = threading.Event()

def _sample_process_tree_rss(
    interval: float = 0.1
) -> None:
    global _peak_tree_rss_mb
    me = psutil.Process(os.getpid())

    while not _stop_sampling.is_set():
        total = 0
        try:
            total += me.memory_info().rss
        except psutil.NoSuchProcess:
            break

        for child in me.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        
        _peak_tree_rss_mb = max(_peak_tree_rss_mb, total / 1024 / 1024)
        time.sleep(interval)

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]

    sampler = threading.Thread(target=_sample_process_tree_rss, daemon=True)
    sampler.start()

    start_time = time.perf_counter()
    vocab, merges = train_bpe("data/raw_text/owt_valid.txt", 10000, special_tokens, _CHUNK_NUM, _WORKER_NUM)
    elapsed_time = time.perf_counter() - start_time

    _stop_sampling.set()
    sampler.join()
    
    print(f"""
    =============================================================================
    Training time: {str(timedelta(seconds=elapsed_time))}
    Peak Rss: {_peak_tree_rss_mb:.2f} MB
    =============================================================================
    """)

    out_dir = Path("data/tokenizer")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open("data/tokenizer/owt_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    
    with open("data/tokenizer/owt_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    
    tokenizer = Tokenizer.from_files(
        vocab_filepath="data/tokenizer/owt_vocab.pkl",
        merges_filepath="data/tokenizer/owt_merges.pkl",
        special_tokens=special_tokens
    )

    longest_token_bytes = b''
    longest_token_len = 0
    for id, tok in tokenizer.vocab.items():
        if len(tok) > longest_token_len:
            longest_token_bytes = tok
            longest_token_len = len(tok)

    print(f"""
    ================================================================================
    Vocab size: {len(tokenizer.vocab)}
    Merges num: {len(tokenizer.merges)}
    longest token: <|{longest_token_bytes.decode("utf-8", errors="replace")}|>
    ================================================================================
    """)