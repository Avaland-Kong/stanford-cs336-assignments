import math

import torch
import torch.nn as nn

from cs336_basics.model.rope import RoPE
from cs336_basics.model.linear import Linear

def split_heads(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int
) -> torch.Tensor:
    return x.reshape(*x.shape[:-1], num_heads, head_dim).transpose(-3, -2)

def softmax(
    input: torch.Tensor,
    dim: int
) -> torch.Tensor:
    max = input.max(dim=dim, keepdim=True).values
    input_exp = (input - max).exp()
    sum_exp = input_exp.sum(dim=dim, keepdim=True)
    return input_exp / sum_exp


def sdpa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    d_k = K.shape[-1]
    score = Q@K.transpose(-2,-1) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(~mask, -float('inf'))
    return softmax(score, dim=-1)@V


class MHA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RoPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        """
        Initialize the MHA block.

        Notes:
            - Uses a fused w_qkv projection and supports optional RoPE.
            - rope: Optional Rotary Position Embedding module. If None, no positional encoding is applied. 
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_qkv = d_model // num_heads
        self.w_qkv = Linear(self.d_model, 3 * self.d_model, device=device, dtype=dtype)
        self.w_o = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.rope = rope

        self.reset_parameter()

    
    def reset_parameter(self):
        self.w_qkv.reset_parameters()
        self.w_o.reset_parameters()

    
    
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        start_position: int | None = None,
        kv_cache: tuple[torch.Tensor,torch.Tensor] | None = None,
    ) -> torch.Tensor: 
        """
        Notes:
            - x: Input features, shape (..., sequence_length, d_model). 
            - token_positions: Optional absolute positions, shape (sequence_length,). 
              If None or no RoPE module, RoPE is skipped. 
        """
        Q, K, V = self.w_qkv(x).split(self.d_model, dim=-1)
        Q = split_heads(Q, self.num_heads, self.d_qkv)
        K = split_heads(K, self.num_heads, self.d_qkv)
        V = split_heads(V, self.num_heads, self.d_qkv)
        if token_positions is not None:
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)
        
        causal_mask = torch.ones(Q.shape[-2], K.shape[-2], device=Q.device, dtype=torch.bool).tril()
        
        if kv_cache is not None:
            
            k_cache, v_cache = kv_cache
            l_new = K.shape[-2]
            k_cache[..., start_position : start_position + l_new, :] = K
            v_cache[..., start_position : start_position + l_new, :] = V

            K = k_cache[..., 0 : start_position + l_new, :]
            V = v_cache[..., 0 : start_position + l_new, :]
            
            if l_new == 1:
                causal_mask = None

        sdpa_result = sdpa(Q, K, V, mask=causal_mask).transpose(-3,-2).reshape(x.shape)
        return self.w_o(sdpa_result)
