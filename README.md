# 🍏🤖 `mlx-playground`

Run fast transformer decoders  on your Macbooks' GPU!
Working towards a fast reimplementation of GPT-2 and Llama-like models in [mlx](https://ml-explore.github.io/mlx/build/html/index.html).

The aim is that the only dependencies are:
- `mlx`
- `sentencepiece`
- `tqdm`
- `numpy`

With an optional dev dependency of:
- `transformers` for downloading and converting weights

## Project Goals

These will be checked off as they're completed.
This project will be considered complete once these goals are achieved.

- [ ] Phi-2 reimplementation in MLX
- [ ] GPT-2 reimplementation in MLX
- [x] ~~llama reimplementation~~ (train your own makemore llama w/ `python train.py`!)
- [ ] conversion script for HF format to MLX format
- [ ] speculative decoding
- [x] ~~AdamW implementation~~ [merged!](https://github.com/ml-explore/mlx/pull/72)
- [ ] learning rate scheduling 

## Installation

```
poetry install --no-root
```

## Phi-2

To download and convert the model:

```sh 
python phi2/convert.py
```

That will fill in `weights/phi-2.npz`.

🚧 (Not yet done) To run the model:

```sh
python phi2/generate.py
```

## Acknowledgements

Some great resources:

- [Brian Kitano's LLaMa from Scratch](https://blog.briankitano.com/llama-from-scratch/)
- [PyTorch lab's `gpt-fast`](https://github.com/pytorch-labs/gpt-fast)
