# mlxllama

Run llama on your Macbooks' GPU!
Working towards a fast reimplementation of llama-2 in mlx.

## Project Goals

These will be checked off as they're completed.
This project will be considered complete once these goals are achieved.

The aim is that the only dependencies are:
- `mlx`
- `sentencepiece`
- `tqdm`

With an optional dev dependency of:
- `transformers` for downloading and converting weights

1. [ ] model reimplementation in MLX
2. [ ] conversion script for HF format to MLX format
3. [ ] Add [https://github.com/pytorch-labs/gpt-fast](gpt-fast) optimizations
4. [ ] LoRA for fine-tuning

## Installation

```
pip install -Ue .
```

## Usage

🚧 TBD