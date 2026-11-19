import math
from typing import Iterable

from regex import P
import torch
import torch.nn as nn

def gradient_clipping(
    params : Iterable[nn.Parameter],
    max_l2_norm : float
) -> None:
    l2_norm_pow = 0
    for param in params:   
        if param.grad is None:
            continue
        l2_norm_pow += torch.norm(param.grad, 2)  ** 2
    
    l2_norm = math.sqrt(l2_norm_pow)
    
    if l2_norm >= max_l2_norm:
        for param in params:
            if param.grad is None:
                continue
            param.grad.mul_(max_l2_norm / (l2_norm + 1e-6))

