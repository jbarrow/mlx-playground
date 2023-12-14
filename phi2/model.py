from typing import Optional
from dataclasses import dataclass
from mlx.utils import tree_flatten

import mlx.core as mx
import mlx.nn as nn
import numpy


@dataclass
class ModelArgs:
    max_sequence_length: int = 2048
    num_vocab: int = 51200
    model_dim: int = 2560
    num_heads: int = 32
    num_layers: int = 32
    rotary_dim: int = 32


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dims: int, num_heads: int, mlp_dims: Optional[int] = None):
        super().__init__()
        mlp_dims = mlp_dims or dims * 4
        self.self_attention = nn.MultiHeadAttention(dims, num_heads, bias=True)
        self.ln = nn.LayerNorm(dims)
        self.fc1 = nn.Linear(dims, mlp_dims)
        self.fc2 = nn.Linear(mlp_dims, dims)

    def __call__(self, x, memory, x_mask, memory_mask):
        y = self.ln(x)
        y = self.self_attention(y, y, y, x_mask)
        x = x + y

        y = self.ln(x)
        y = self.linear1(y)
        y = mx.maximum(y, 0)
        y = self.linear2(y)
        x = x + y

        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self, num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None
    ):
        super().__init__()
        self.h = [
            TransformerDecoderLayer(dims, num_heads, mlp_dims)
            for i in range(num_layers)
        ]

    def __call__(self, x, memory, x_mask, memory_mask):
        for layer in self.h:
            x = layer(x, memory, x_mask, memory_mask)
        return x


class Phi2(nn.Module):
    def __init__(self, config: ModelArgs):
        self.wte = nn.Embedding(config.num_vocab, config.model_dim)
        self.transformer = TransformerDecoder(
            num_layers=config.num_layers,
            dims=config.model_dim,
            num_heads=config.num_heads,
        )

        self.lm_head = LanguageModelingHead(config)

    def __call__(
        self,
        input_ids: mx.array,
        positions: mx.array,
        attention_mask: mx.array = None,
    ) -> tuple[mx.array, mx.array]:
        text = self.wte(input_ids)
        position = self.wpe(positions)

        x = text + position

        if attention_mask is not None:
            # convert 0's to -infs, 1's to 0's, and make it broadcastable
            attention_mask = mx.log(attention_mask)
            attention_mask = mx.expand_dims(attention_mask, (1, 2))

        y = self.decoder(x, attention_mask)
        return self.lm_head(y)


class LanguageModelingHead(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        self.ln = nn.LayerNorm(config.model_dim)
        self.linear = nn.Linear(config.model_dim, config.num_vocab)

    def __call__(self, inputs, mask):
        return self.linear(self.ln(inputs))


if __name__ == "__main__":
    new_model = Phi2(ModelArgs())
    new_weights = dict([(k, v.shape) for k, v in tree_flatten(new_model.parameters())])

    # print(set(weights.keys()) ^ set(new_weights.keys()))
