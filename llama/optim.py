from mlx.optimizers import Optimizer, OptimizerState
from typing import List

import mlx.core as mx


class AdamW(Optimizer):
    r"""Implementation of the AdamW optimizer [1].

    Following the above convention, in contrast with [1], we do not use bias
    correction in the first and second moments for AdamW. We update the weights 
    with a weight_decay (λ) value:

    .. math::

        m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
        v_{t+1} &= \beta_2 v_t + (1 - \beta_2) g_t^2 \\
        \hat{m}_{t+1} &= \frac{m_t}{(1 - \beta_1^t)}
        \hat{v}_{t+1} &= \frac{v_t}{(1 - \beta_1^t)}
        w_{t+1} &= w_t - \alpha (\frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1} + \epsilon}} + \lambda w_t)

    [1]: Loshchilov, I. and Hutter, F., 2019. Decoupled weight decay 
    regularization. ICLR 2019.
    """

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
        t = state.get("t", 1)
        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * mx.square(gradient)
        state["m"] = m
        state["v"] = v
        state["t"] = t + 1

        m_hat = m / (1. - b1 ** t)
        v_hat = v / (1. - b2 ** t)

        return parameter - lr * (m_hat / (mx.sqrt(v_hat) + eps) + wd * parameter)
