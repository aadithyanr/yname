# Model

`yname` uses a deliberately small character-level multilayer perceptron. It is
inspired by the neural probabilistic language model in
[Bengio et al. (2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf),
but implemented directly in NumPy for this dataset.

```text
previous 10 characters
        ↓
character embeddings ── industry embedding (conditional model only)
        ↓
160-unit tanh hidden layer
        ↓
next-character probabilities
```

## Files

- `tiny_mlp.py` is a small educational implementation using twelve familiar
  startup names.
- `train.py` cleans the YC directory, creates stratified splits, trains both
  models, evaluates them, and writes compressed NumPy archives.
- `generate.py` samples filtered names from the trained archives.
- `artifacts/` contains the model snapshots and training report used by the
  browser app.

## Generate locally

```bash
python model/generate.py --model plain --count 20

python model/generate.py \
  --model conditional \
  --category Fintech \
  --creativity high \
  --count 20
```

No pretrained weights or language-model APIs are used.
