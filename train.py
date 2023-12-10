"""
Super simple train.py, getting started without any tokenizers,
and with a very simple training loop.
"""
from llama.model import Llama, ModelArgs
from llama.optim import AdamW
from mlx.utils import tree_flatten
from tqdm import tqdm

import mlx.optimizers as optim
import mlx.core as mx
import mlx.nn as nn


lines = open("./data/example.txt", "r").read()

vocab = sorted(list(set(lines)))
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}

CONFIG = {
    "context_length": 16,
    "batch_size": 32,
    "steps": 1000,
    "learning_rate": 0.001,
}


def encode(s):
    return [stoi[ch] for ch in s]


def decode(l):
    return "".join([itos[i] for i in l])


def get_batches(
    data: mx.array, split: str, batch_size: int, context_window: int, config=CONFIG
) -> tuple[mx.array, mx.array]:
    train = data[: int(0.8 * len(data))]
    val = data[int(0.8 * len(data)) : int(0.9 * len(data))]
    test = data[int(0.9 * len(data)) :]

    batch_data = train
    if split == "val":
        batch_data = val

    if split == "test":
        batch_data = test

    ixs = mx.random.randint(
        0, batch_data.shape[0] - context_window - 1, shape=(batch_size,)
    ).tolist()

    # create B x C tensors of x and y
    x = mx.concatenate(
        [mx.expand_dims(batch_data[ix : ix + context_window], 0) for ix in ixs], axis=0
    )
    y = mx.concatenate(
        [mx.expand_dims(batch_data[ix + 1 : ix + context_window + 1], 0) for ix in ixs],
        axis=0,
    )

    return x, y


def evaluate_loss(model, config=CONFIG) -> dict[str, mx.array]:
    out = {}

    mx.eval(model.parameters())
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(
                dataset, split, config["batch_size"], config["context_length"], config
            )
            loss = model.loss(xb, yb)
            losses.append(loss.item())
        out[split] = mx.mean(mx.array(losses)).item()
    return out


def train(model: nn.Module, optimizer, config=CONFIG):
    losses = []

    loss_and_grad_fn = nn.value_and_grad(model, model.loss)
    pbar = tqdm(range(config["steps"]))

    for step in pbar:
        xs, ys = get_batches(
            dataset, "train", config["batch_size"], config["context_length"]
        )

        loss, grads = loss_and_grad_fn(xs, ys)
        model.update(optimizer.apply_gradients(grads, model))

        mx.simplify(loss, model.parameters())
        # mx.eval(loss, model.parameters())
        losses.append(loss.item())

        pbar.set_description(f"loss: ({loss.item():.2f})")

    print(evaluate_loss(model))


if __name__ == "__main__":
    dataset = mx.array(encode(lines))

    args = ModelArgs()
    model = Llama(args)

    nparams = sum(x.size for k, x in tree_flatten(model.parameters()))
    print(f"training a model with {nparams} trainable params")

    optimizer = AdamW(
        learning_rate=CONFIG["learning_rate"], weight_decay=0.1, betas=[0.9, 0.95]
    )
    # optimizer = optim.Adam(learning_rate=CONFIG["learning_rate"])

    train(model, optimizer)
