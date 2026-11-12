import math
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(
        self,
        num_embedding: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        """
        Initialize the embedding layer.

        Notes:
          -Embedding_weight are initialized by random normal sampling: mean=0, std=1, truncated to [-3, 3].
        """
        super().__init__()
        self.num_embeddings = num_embedding
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embedding, embedding_dim, device=device, dtype=dtype))

        self.reset_parameters()


    def reset_parameters(self):
        mean = 0
        std = math.sqrt(1)
        nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=-3, b=3)

    def forward(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert input token_id into word vectors.
        """
        return self.weight[token_ids]