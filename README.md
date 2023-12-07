# 🍏🦙 `mlxllama`

Run llama on your Macbooks' GPU!
Working towards a fast reimplementation of llama-2 in [mlx](https://ml-explore.github.io/mlx/build/html/index.html).

The aim is that the only dependencies are:
- `mlx`
- `sentencepiece`
- `tqdm`

With an optional dev dependency of:
- `transformers` for downloading and converting weights

## Project Goals

These will be checked off as they're completed.
This project will be considered complete once these goals are achieved.

- [ ] model reimplementation in MLX
- [x] AdamW implementation
- [ ] learning rate scheduling 
- [ ] conversion script for HF format to MLX format
- [ ] Add [gpt-fast](https://github.com/pytorch-labs/gpt-fast) optimizations
- [ ] LoRA for fine-tuning

## Installation

```
pip install -Ue .
```

## Usage

🚧 TBD

Something akin to the following:

```sh
python train.py <train> <val> <other params>
```

```sh
python generate.py <model> <prompt>
```

## Acknowledgements

Some great resources:

- [Brian Kitano's LLaMa from Scratch](https://blog.briankitano.com/llama-from-scratch/)
- [PyTorch lab's `gpt-fast`](https://github.com/pytorch-labs/gpt-fast)
