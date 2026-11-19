import math

from math import pi as PI

def lr_cosine_schedule(
    t: int,
    max_learning_rate: float,
    min_learning_rate: float,
    T_warmup: int,
    T_cosine_cycle: int,
) -> float:
    if t < T_warmup:
        return t / T_warmup * max_learning_rate
    elif T_warmup <= t <= T_cosine_cycle:
        cosine_weight = 1 / 2 * (1 + math.cos((t - T_warmup) / (T_cosine_cycle - T_warmup) * PI))
        return min_learning_rate + cosine_weight * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate
