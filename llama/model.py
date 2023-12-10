from dataclasses import dataclass
from typing import Optional, Tuple

import mlx.nn as nn
import mlx.core as mx


@dataclass
class ModelArgs:
    block_size: int = 16
    vocab_size: int = 65
    n_layers: int = 4
    n_heads: int = 8
    dims: int = 256
    intermediate_size: int = 512
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10_000
    norm_eps: float = 1e-5
    n_kv_heads: int = 4

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
        self.w2 = nn.Linear(config.intermediate_size, config.dims, bias=False)
        self.w3 = nn.Linear(config.dims, config.intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config

        self.n_heads: int = config.n_heads
        self.n_kv_heads: int = config.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(config.dims, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(
            config.dims, config.n_kv_heads * config.head_dim, bias=False
        )
        self.wv = nn.Linear(
            config.dims, config.n_kv_heads * config.head_dim, bias=False
        )
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dims, bias=False)
        self.rope = nn.RoPE(config.head_dim, traditional=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        def repeat(a):
            a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
            return a.reshape([B, self.n_heads, L, -1])

        keys, values = map(repeat, (keys, values))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), (keys, values)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()

        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = nn.RMSNorm(config.dims, config.norm_eps)
        self.attention_norm = nn.RMSNorm(config.dims, config.norm_eps)

    def __call__(self, x, mask=None):
        y = self.attention_norm(x)
        y, _ = self.attention(y, mask)
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
