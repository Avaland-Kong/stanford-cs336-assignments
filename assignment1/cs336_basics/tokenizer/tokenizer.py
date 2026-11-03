from typing import Iterable, Iterator
from .pretokenize import pre_tokenize, segment_by_special_tokens

class Tokenizer:
    """
    Byte-level BPE tokenizer: text ↔ token ids.
    """
    def __init__(
        self,
        vocab : dict[int, bytes],
        merges : list[tuple[bytes, bytes]],
        special_tokens : list[str] | None = None
    ) -> None:
        """
        Store vocab/merges/special_tokens and build derived tables for encode/decode.

        Notes:
          - Special tokens not already in vocab are appended at the end, in order.
          - Assumes vocab ids are contiguous 0..len(vocab)-1 with no holes;
        """
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens

        self._bytes_to_id = {b : i for i, b in vocab.items()}
        self._merge_ranks = {pair : i for i, pair in enumerate(merges)}

        next_id = len(vocab)
        if special_tokens:
            for token_str in special_tokens:
                token_bytes = token_str.encode("utf-8")
                if token_bytes in self._bytes_to_id:
                    continue
                self.vocab[next_id] = token_bytes
                self._bytes_to_id[token_bytes] = next_id
                next_id += 1


    def _encode_merge_loop(
        self,
        word : tuple[bytes, ...]
    ) -> tuple[bytes, ...]:
        """
        Repeatedly merge the lowest-rank adjacent pair until none can be merged.

        Notes:
          - Ranks come from the training-time merge order; smaller means higher priority.
        """
        while len(word) > 1:
            lowest_rank = float('inf')
            merge_loc = -1
            for i, pair in enumerate(zip(word, word[1:])):
                rank = self._merge_ranks.get(pair, float('inf'))
                if rank < lowest_rank:
                    merge_loc = i
                    lowest_rank = rank
                        
            if merge_loc != -1:
                i = 0
                new_word = []
                while i < len(word):
                    if i == merge_loc:
                        new_word.append(word[i] + word[i+1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = tuple(new_word)
            else:
                break
        
        return word

    def encode(
        self,
        text : str
    ) -> list[int]:
        """
        Encode text into a list of token ids.

        Notes:
          - Special tokens map directly to ids, bypassing pre-tokenization and BPE.
        """
        result = []
        for start, end, is_special_token in segment_by_special_tokens(text, self.special_tokens):
            if is_special_token:
                token = text[start : end].encode("utf-8")
                result.append(self._bytes_to_id[token])
            else:
                for pt in pre_tokenize(text[start : end]):
                    if not pt:
                        continue
                    pt_bytes = pt.encode("utf-8")
                    word = self._encode_merge_loop(tuple(bytes([x]) for x in pt_bytes))
                    for token in word:
                        result.append(self._bytes_to_id[token])

        return result 

    def decode(
        self,
        ids : list[int]
    ) -> str:
        """
        Decode a list of token ids back into text.

        Notes:
          - errors="replace" substitutes invalid byte sequences with U+FFFD instead of raising.
        """
        return b''.join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")

    def encode_iterable(
        self, 
        iterable: Iterable[str]
    ) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
        


