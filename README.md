# yname

Generate a YC startup name before someone else does.

A tiny character-level neural network trained from scratch on 6,194 companies
from the public YC directory. No LLM, API, or backend—the model runs entirely
in your browser.

<img width="1200" alt="yname social preview" src="web/public/og.png" />

## How it works

The model sees the previous ten characters of a company name and predicts the
next one. It uses learned character embeddings, one `tanh` hidden layer, and a
softmax output. A second model adds a small industry embedding so generation
can be conditioned on categories such as B2B, Fintech, or Healthcare.

- Plain model: 46,700 parameters
- Industry-conditioned model: 49,068 parameters
- Training set: 6,090 cleaned unique names
- Runtime: TypeScript matrix inference in the browser

Generated candidates are rejected when they duplicate or closely resemble an
existing YC company name.

## Repository structure

```text
data/                 directory snapshot and cleaned training corpus
model/                NumPy trainer, generator, reports, and model archives
web/                  Vite + React browser application
```

## Run the site

```bash
cd web
npm install
npm run dev
```

## Train the models

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r model/requirements.txt

python model/fetch_data.py
python model/train.py \
  --data data/yc_companies.json \
  --output-dir model/artifacts
```

Export freshly trained weights for the browser:

```bash
cd web
npm run export-models
```

## Deploy

Import the repository into Vercel. The root `vercel.json` installs and builds
the app from `web/` and serves the static output from `web/dist/`.

The source directory is maintained by
[yc-oss](https://yc-oss.github.io/api/companies/all.json). The snapshot used by
the live model was scraped with [Context.dev](https://context.dev/).

This is an unofficial project and is not affiliated with or endorsed by
Y Combinator.

## License

[MIT](LICENSE)
