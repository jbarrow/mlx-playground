from dataclasses import dataclass

import mlx.nn as nn
import mlx.core as mx
import math


@dataclass
class ModelArgs:
    block_size: int = 16
    vocab_size: int = 65
    n_layers: int = 4
    n_heads: int = 8
    dims: int = 256
    intermediate_size: int = 128
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10_000
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_heads

        # if self.intermediate_size is None:
        #     hidden_dim = 4 * self.dims
        #     n_hidden = int(2 * hidden_dim / 3)
        #     self.intermediate_size = find_multiple(n_hidden, 256)

        self.head_dim = self.dims // self.n_heads


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dims, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.dims, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.intermediate_size, config.dims, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # compute the SwiGLU activation of x
        a = self.w1(x)
        b = self.w2(x)
        return self.w3(a * mx.sigmoid(a) * b)


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.num_heads = config.n_heads

        self.rope = nn.RoPE(config.dims // config.n_heads, traditional=True)
        self.query_proj = nn.Linear(config.dims, config.dims, bias=False)
        self.key_proj = nn.Linear(config.dims, config.dims, bias=False)
        self.value_proj = nn.Linear(config.dims, config.dims, bias=False)
        self.out_proj = nn.Linear(config.dims, config.dims, bias=False)

    def __call__(self, queries, keys, values, mask=None, cache=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        # Extract some shapes
        num_heads = self.num_heads
        B, L, D = queries.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rope(queries)
        keys = self.rope(keys)

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Note that we return the keys and values to possibly be used as a cache
        return self.out_proj(values_hat)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()

        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = nn.RMSNorm(config.dims, config.norm_eps)
        self.attention_norm = nn.RMSNorm(config.dims, config.norm_eps)

    def __call__(self, x, mask=None):
        y = self.attention_norm(x)
        y = self.attention(y, y, y, mask)
        x = x + y

        y = self.ffn_norm(x)
        y = self.feed_forward(y)
        x = x + y

        return x


class Llama(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.dims)
        self.attention = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.norm = nn.RMSNorm(config.dims)
        self.out_proj = nn.Linear(config.dims, config.vocab_size, bias=False)

    def __call__(self, idx: mx.array):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(idx.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)

        x = self.embedding(idx)
        for encoding_layer in self.attention:
            x = encoding_layer(x, mask)
        x = self.norm(x)

        return self.out_proj(x)

    def loss(self, x, y):
        logits = self(x)
        losses = nn.losses.cross_entropy(logits, y)
        mx.simplify(losses)

        return mx.mean(losses)
