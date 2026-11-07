import numpy as np

from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries

_txt_path = "data/raw_text/TinyStoriesV2-GPT4-valid.txt"
_vocab_path = "data/tokenizer/tinystories_vocab.pkl"
_merges_path = "data/tokenizer/tinystories_merges.pkl"
_save_path = "data/tokens/tinystories_valid.bin"
_batch_size = 100000
_chuck_num = 256

special_tokens = ["<|endoftext|>"]
tokenizer = Tokenizer.from_files(
        vocab_filepath=_vocab_path,
        merges_filepath=_merges_path,
        special_tokens=special_tokens
    )

with open(_txt_path, "rb") as text_file, open(_save_path, "wb") as tokens_file:
    buf = []
    split_special_token = special_tokens[0].encode("utf-8") if special_tokens else b'<|endoftext|>'
    chunk_boundaries = find_chunk_boundaries(text_file, _chuck_num, split_special_token)
    
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        text_file.seek(start)
        text_chunk = text_file.read(end - start).decode("utf-8", errors="ignore")
        for tok in tokenizer.encode(text_chunk):
            buf.append(tok)
            if len(buf) >= _batch_size:
                arr = np.array(buf, dtype=np.uint16)
                tokens_file.write(arr.tobytes())
                buf.clear()
            
        if buf:
            arr = np.array(buf, dtype=np.uint16)
            tokens_file.write(arr.tobytes())
            buf.clear()


