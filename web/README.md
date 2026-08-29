# Browser app

The web app loads exported `float32` model weights and runs character inference
directly in TypeScript. There is no server-side generation endpoint.

```bash
npm install
npm run dev
```

Use `npm run export-models` after retraining from the repository root. A
production build is written to `dist/` with `npm run build`.
