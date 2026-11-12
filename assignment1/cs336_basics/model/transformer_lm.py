import torch
import torch.nn as nn

from cs336_basics.model.embedding import Embedding
from cs336_basics.model.transformer_block import TransformerBlock
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.linear import Linear
from cs336_basics.model.attention import softmax

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerBlock(d_model, num_heads, d_ff, context_length, theta, device=device, dtype=dtype))
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def reset_parameter(self):
        self.token_embeddings.reset_parameters()
        for block in self.transformer_blocks:
            block.reset_parameters()
        self.ln_final.reset_parameters()
        self.lm_head.reset_parameters()

    def forward(
        self,
        in_indices: torch.Tensor,
        start_position: int | None = None,
        kv_caches: list[tuple[torch.Tensor,torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        x = self.token_embeddings(in_indices)
        
        if kv_caches is not None:
            for block, kv_cache in zip(self.transformer_blocks, kv_caches):
                x = block(x, start_position, kv_cache)
            x = x[..., -1:, :]
        else: 
            for block in self.transformer_blocks:
                x = block(x)
        
        x = self.ln_final(x)
        return self.lm_head(x)
