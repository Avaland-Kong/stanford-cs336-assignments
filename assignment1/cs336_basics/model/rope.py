import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None  
    ) -> None:
        """
        Initialize the RoPElayer

        Notes:
          - Encodes positions by rotating vectors in 2D planes, 
            ensuring attention scores only depend on relative distances. 
          - Precomputes and caches sin/cos tables for efficiency.
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        i = torch.arange(0, max_seq_len, 1, dtype=torch.float32, device=device)
        freq =  theta ** -((torch.arange(0, d_k, 2, dtype=torch.float32, device=device)) / d_k)
        angle = torch.outer(i, freq)

        self.register_buffer("rotary_factor_buf", torch.polar(torch.ones_like(angle), angle), persistent=False)

    
    def forward(
        self,
        x : torch.Tensor,
        token_positions : torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Rotary Position Embedding (RoPE) via complex multiplication.

        Notes:
          - This function ONLY operates on the last dimension (d_k) and assumes it is even.
          - The caller MUST ensure the shape of `token_positions` can broadcast correctly 
            with `x`. If `x` has a head dimension (e.g., [B, H, S, D]), `token_positions` 
            must be properly expanded (e.g., [B, 1, S]) to avoid dimension mismatch errors.
        """
        x_dtype = x.dtype
        x = x.to(torch.float32)
        rotary_factor = self.rotary_factor_buf[token_positions] 
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        
        return  torch.view_as_real(x_complex * rotary_factor).flatten(-2).to(x_dtype)


        
        

    
