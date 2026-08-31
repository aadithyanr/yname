# Browser app

The web app generates both YC-style startup names and startup ideas locally:

- Name mode loads exported `float32` weights and runs character inference in
  TypeScript.
- Idea mode loads a compact phrase model and recombines industry-conditioned
  solution and audience fragments with exact- and near-copy filtering.

There is no server-side generation endpoint or language-model API.

```bash
npm install
npm run dev
```

Use `npm run export-models` after retraining from the repository root. A
production build is written to `dist/` with `npm run build`.
