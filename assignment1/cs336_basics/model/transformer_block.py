import torch
import torch.nn as nn

from cs336_basics.model.rope import RoPE
from cs336_basics.model.attention import MHA
from cs336_basics.model.swiglu import SwiGLU
from cs336_basics.model.rmsnorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = RoPE(theta, d_model//num_heads, max_seq_len, device=device)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MHA(d_model, num_heads, rope=self.rope, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def reset_parameter(self):
        self.ln1.reset_parameters()
        self.ln2.reset_parameters()
        self.attn.reset_parameter()
        self.ffn.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        start_position: int | None = None,
        kv_cache: tuple[torch.Tensor,torch.Tensor] | None = None,
    ) -> torch.Tensor:
        
        if kv_cache is not None:
            token_positions = torch.arange(start_position, start_position + x.shape[-2], device=x.device)
            y = x + self.attn(self.ln1(x), token_positions, start_position, kv_cache)
        else:
            token_positions = torch.arange(0, x.shape[-2], device=x.device)
            y = x + self.attn(self.ln1(x), token_positions)
        return y + self.ffn(self.ln2(y))