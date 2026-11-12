import torch
import torch.nn as nn

from cs336_basics.model.linear import Linear

def SiLU(
    x: torch.Tensor
) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        """
        Initialize the SwiGLU feed-forward block.

        Notes:
          - w1 and w3 are gated projections to d_ff, and w2 projects the gated result back to d_model.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

        self.reset_parameters()

    
    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.w3.reset_parameters()

    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the SwiGLU transformation to the input.

        Notes:
          - Computes w2( SiLU(w1 x) ⊙ w3 x ) and returns a tensor of the same shape as x.
        """
        gate = SiLU(self.w1(x))
        content =  self.w3(x)
        return self.w2(content * gate)


    