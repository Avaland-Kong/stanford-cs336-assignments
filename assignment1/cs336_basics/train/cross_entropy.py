import torch
import torch.nn as nn

from cs336_basics.model.attention import softmax

def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    max = logits.max(-1, keepdim=True).values
    sum_exp = (logits - max).exp().sum(-1, keepdim=True)
    loss = -(logits.gather(-1, targets.unsqueeze(-1)) - max - sum_exp.log())
    return loss.mean()

def perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    return cross_entropy(logits, targets).exp()