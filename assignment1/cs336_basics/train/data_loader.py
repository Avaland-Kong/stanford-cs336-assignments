import numpy as np
import torch

def load_data(
    x: np.typing.NDArray,
    batch_size: int,
    context_length: int,
    device: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(x) - context_length, (batch_size, 1))
    offsets = torch.arange(0, context_length)
    inputs = torch.from_numpy(x[starts + offsets]).to(device)
    labels = torch.from_numpy(x[starts + 1 + offsets]).to(device)
    return inputs, labels
    