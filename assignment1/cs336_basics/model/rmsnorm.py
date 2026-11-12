import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype:  torch.dtype | None = None
    ) -> None:
        """
        Initialize the RMSNorm layer.
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.ones_(self.weight)

    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply RMSNorm along the last dimension of the input.

        Notes:
          - Computation is done in float32 for stability, and the output is cast back to the input dtype.
        """
        x_dtype = x.dtype
        x = x.to(torch.float32)
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x * inv_rms * self.weight.to(torch.float32)
        return result.to(x_dtype)



