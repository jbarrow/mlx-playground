import mlx.nn as nn
import mlx.core as mx


class RMSNorm(nn.Module):
    pass


class Llama(nn.Module):
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.linear = nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"]),
            nn.ReLU(),
            nn.Linear(config["d_model"], config["vocab_size"]),
        )

    def __call__(self, idx: mx.array):
        x = self.embedding(idx)
        return self.linear(x)

    def loss(self, x, y):
        logits = self(x)
        losses = nn.losses.cross_entropy(logits, y)
        mx.simplify(losses)

        return mx.mean(losses)
