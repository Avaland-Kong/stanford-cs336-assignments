from math import sqrt
from typing import Iterable
import torch
import torch.nn as nn

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8
    ) -> None:
        defaults = {
            "lr" : lr, 
            "weight_decay" : weight_decay,
            "betas" : betas,
            "eps" : eps
        }
        super().__init__(params, defaults)
    
    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)
                
                lr_t = lr * (sqrt(1 - beta2 ** state["t"])) / (1 - beta1 ** state["t"])
                
                param.data = param.data - lr * weight_decay * param.data
                
                state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * grad ** 2

                param.data = param.data - lr_t * state["m"] / torch.sqrt(state["v"] + eps)

                state["t"] = state["t"] + 1


