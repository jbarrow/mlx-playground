from typing import Optional
from transformers import AutoModelForCausalLM
from dataclasses import dataclass
from pprint import pprint

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelArgs:
    max_sequence_length: int = 1024
    num_vocab: int = 50257
    model_dim: int = 768
    mlp_dim: int = 3072
    num_heads: int = 12


class Gpt2(nn.Module):
    def __init__(self, config: ModelArgs):
        self.wte = nn.Embedding(config.num_vocab, config.model_dim)
        self.wpe = nn.Embedding(config.max_sequence_length, config.model_dim)
        self.decoder = nn.TransformerDecoder(
            num_layers=config.num_layers,
            dims=config.model_dim,
            num_heads=config.num_heads,
            mlp_dims=config.mlp_dim
        )
        
        self.lm_head = nn.Linear(config.model_dim, config.num_vocab)

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


def replace_key(key: str) -> str:

    return key


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    new_model = Gpt2(ModelArgs())

    pprint([(k, v.shape) for k, v in model.state_dict().items()])
