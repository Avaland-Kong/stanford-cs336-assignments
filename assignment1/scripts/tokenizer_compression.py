from cs336_basics.tokenizer.tokenizer import Tokenizer

_txt_path = "data/tiny_text/owt.txt"
_vocab_path = "data/tokenizer/tinystories_vocab.pkl"
_merges_path = "data/tokenizer/tinystories_merges.pkl"

special_tokens = ["<|endoftext|>"]
tokenizer = Tokenizer.from_files(
        vocab_filepath=_vocab_path,
        merges_filepath=_merges_path,
        special_tokens=special_tokens
    )

with open(_txt_path, "rb") as f:
    text_bytes = f.read()
    total_bytes = len(text_bytes)
    token_list = tokenizer.encode(text_bytes.decode("utf-8"))
    total_tokens = len(token_list)

compression_rate = total_bytes/total_tokens

print(f"""
============================================================================
Compression Rate: {compression_rate}
============================================================================
""")
