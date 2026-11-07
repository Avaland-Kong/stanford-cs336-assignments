import time

from cs336_basics.tokenizer.tokenizer import Tokenizer


_txt_path = "data/raw_text/owt_valid.txt"
_vocab_path = "data/tokenizer/owt_vocab.pkl"
_merges_path = "data/tokenizer/owt_merges.pkl"

special_tokens = ["<|endoftext|>"]
tokenizer = Tokenizer.from_files(
        vocab_filepath=_vocab_path,
        merges_filepath=_merges_path,
        special_tokens=special_tokens
    )

with open(_txt_path, "rb") as f:
    text_bytes = f.read()
    total_bytes = len(text_bytes)
    text = text_bytes.decode("utf-8")
    start_time = time.perf_counter()
    token_list = tokenizer.encode(text)
    elapsed_time = time.perf_counter() - start_time


throughput = total_bytes / elapsed_time

print(f"""
============================================================================
Throughput: {throughput / 1024 / 1024} MiB/s
============================================================================
""")


