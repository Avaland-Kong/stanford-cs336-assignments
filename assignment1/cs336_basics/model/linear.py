import math
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype  | None = None
    ) -> None:
        """
        Initialize the linear layer.
        
        Notes:
          - Weights are initialized using Xavier initialization and truncated to the range [-3*sigma, 3*sigma].
          - This layer has no bias parameter.
        """
        super().__init__()
        self.d_in = in_features
        self.d_out = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        mean = 0
        std = math.sqrt(2/(self.d_in + self.d_out))
        nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=-3*std, b=3*std)
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        mapping input features from `in_features` dimensions to `out_features` dimensions.
        """
        y = x @ self.weight.T
        return y

    