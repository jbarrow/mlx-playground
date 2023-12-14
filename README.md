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

## Accomplishments

- [x] ~~makemore llama reimplementation~~ (train your own w/ `python train.py`!)
- [x] [BERT merged into `mlx-examples`](https://github.com/ml-explore/mlx-examples/pull/43)
- [x] [Phi-2 merged into `mlx-examples`](https://github.com/ml-explore/mlx-examples/pull/97)
- [x] [AdamW merged into `mlx`](https://github.com/ml-explore/mlx/pull/72)

## Remaining Goals

This project will be considered complete once these goals are achieved.

- [ ] finetune BERT
- [ ] GPT-2 reimplementation and loading in MLX
- [ ] speculative decoding
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
