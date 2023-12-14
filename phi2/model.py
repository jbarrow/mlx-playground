from typing import Optional
from transformers import AutoModelForCausalLM
from dataclasses import dataclass
from mlx.utils import tree_flatten

import mlx.core as mx
import mlx.nn as nn


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


def split_attention_matrix(state_dict, key) -> dict:
    # "transformer.h.0.mixer"
    _, model_dim = state_dict[key + ".weight"].shape
    # (3 * model_dim, model_dim)
    Wqkv_weight_key = key + ".weight"
    Wq_weight = state_dict[Wqkv_weight_key][:model_dim, :]
    Wk_weight = state_dict[Wqkv_weight_key][model_dim : 2 * model_dim, :]
    Wv_weight = state_dict[Wqkv_weight_key][2 * model_dim :, :]

    # (3 * model_dim)
    Wqkv_bias_key = key + ".bias"
    Wq_bias = state_dict[Wqkv_bias_key][:model_dim]
    Wk_bias = state_dict[Wqkv_bias_key][model_dim : 2 * model_dim]
    Wv_bias = state_dict[Wqkv_bias_key][2 * model_dim :]

    out_key = key.replace("mixer.Wqkv", "self_attention")

    return {
        out_key + ".query_proj.weight": Wq_weight,
        out_key + ".query_proj.bias": Wq_bias,
        out_key + ".key_proj.weight": Wk_weight,
        out_key + ".key_proj.bias": Wk_bias,
        out_key + ".value_proj.weight": Wv_weight,
        out_key + ".value_proj.bias": Wv_bias,
    }


def replace_key(key: str) -> str:
    if "wte.weight" in key:
        key = "wte.weight"

    if ".mlp" in key:
        key = key.replace(".mlp", "")

    if ".mixer.out_proj" in key:
        key = key.replace(".mixer", ".self_attention")

    return key


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", torch_dtype="auto", trust_remote_code=True
    )
    state_dict = model.state_dict()
    keys = list(state_dict.keys())

    for key in keys:
        if ".mixer.Wqkv.weight" not in key:
            continue
        key_stub = key.rstrip(".weight")
        state_dict.update(split_attention_matrix(state_dict, key_stub))

        del state_dict[key_stub + ".weight"]
        del state_dict[key_stub + ".bias"]

    weights = dict([(replace_key(k), v.shape) for k, v in state_dict.items()])

    new_model = Phi2(ModelArgs())
    new_weights = dict([(k, v.shape) for k, v in tree_flatten(new_model.parameters())])

    print(set(weights.keys()) ^ set(new_weights.keys()))
