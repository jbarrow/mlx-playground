from mlx.optimizers import Optimizer, OptimizerState
from typing import List

import mlx.core as mx


class AdamW(Optimizer):
    r"""Implementation of the AdamW optimizer."""

    def __init__(
        self,
        learning_rate: float,
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

    def apply_single(
        self, gradient: mx.array, parameter: mx.array, state: OptimizerState
    ):
        """Performs the AdamW parameter update and stores :math:`v` and
        :math:`m` in the optimizer state."""
        lr = self.learning_rate
        b1, b2 = self.betas
        eps = self.eps
        wd = self.weight_decay

        m = state.get("m", gradient)
        v = state.get("v", mx.square(gradient))
        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * mx.square(gradient)
        state["m"] = m
        state["v"] = v

        return parameter - lr * (m / (mx.sqrt(v) + eps) + wd * parameter)
