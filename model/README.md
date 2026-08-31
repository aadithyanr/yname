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
- `train.py` loads the cleaned training corpus, creates stratified splits,
  trains both models, evaluates them, and writes compressed NumPy archives.
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

## Startup idea model

The experimental idea generator is a dependency-free hybrid token model trained
on YC company one-liners:

```text
industry ── solution-fragment distribution ─┐
                                            ├─ filtered startup one-liner
industry ── audience-fragment distribution ─┘

previous 2 tokens ── interpolated industry/global backoff ── next token
```

Low and medium creativity recombine solution and audience fragments from
different descriptions. High creativity can also sample the token model for
less predictable output. Exact matches, close lexical copies, malformed ideas,
and near-duplicates within a generated batch are rejected.

- `idea_model.py` implements tokenization, training, sampling, serialization,
  novelty scoring, and validation.
- `train_ideas.py` trains the model and writes a compressed JSON artifact.
- `generate_ideas.py` generates ideas by industry from that artifact.
- `evaluate_ideas.py` measures held-out token/context coverage and generated
  output novelty.

```bash
python model/train_ideas.py
python model/generate_ideas.py --category Fintech --count 20
python model/evaluate_ideas.py
```

The compressed artifact is plain JSON inside gzip so a later browser export can
reuse it without a Python runtime. `web/scripts/export_models.py` exports the
solution fragments, audience fragments, and novelty index as a compact browser
asset; the browser does not download the larger token backoff tables.
