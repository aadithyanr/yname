export type ModelKind = "plain" | "conditional";

type ParameterMetadata = {
  shape: number[];
  byteOffset: number;
  length: number;
};

export type ModelManifest = {
  version: number;
  kind: ModelKind;
  architecture: {
    context_size: number;
    embedding_size: number;
    hidden_size: number;
    category_embedding_size: number;
    category_count: number;
    vocab_size: number;
    conditional: boolean;
  };
  tokens: string[];
  categories: string[];
  parameters: Record<string, ParameterMetadata>;
  training: {
    sourceRecords: number;
    uniqueCleanedNames: number;
    bestStep: number;
    testLoss: number;
  };
};

export type NameModel = {
  manifest: ModelManifest;
  parameters: Record<string, Float32Array>;
  tokenToId: Map<string, number>;
};

const modelPromises = new Map<ModelKind, Promise<NameModel>>();

export function loadModel(kind: ModelKind): Promise<NameModel> {
  const existing = modelPromises.get(kind);
  if (existing) return existing;

  const promise = Promise.all([
    fetch(`/models/${kind}.json`).then((response) => {
      if (!response.ok) throw new Error(`could not load ${kind} model metadata`);
      return response.json() as Promise<ModelManifest>;
    }),
    fetch(`/models/${kind}.bin`).then((response) => {
      if (!response.ok) throw new Error(`could not load ${kind} model weights`);
      return response.arrayBuffer();
    }),
  ]).then(([manifest, buffer]) => {
    const parameters: Record<string, Float32Array> = {};
    for (const [name, metadata] of Object.entries(manifest.parameters)) {
      parameters[name] = new Float32Array(
        buffer,
        metadata.byteOffset,
        metadata.length,
      );
    }
    return {
      manifest,
      parameters,
      tokenToId: new Map(manifest.tokens.map((token, index) => [token, index])),
    };
  });

  modelPromises.set(kind, promise);
  return promise;
}

export function forward(
  model: NameModel,
  context: number[],
  categoryId: number,
): Float32Array {
  const { architecture } = model.manifest;
  const {
    context_size: contextSize,
    embedding_size: embeddingSize,
    hidden_size: hiddenSize,
    category_embedding_size: categoryEmbeddingSize,
    vocab_size: vocabSize,
    conditional,
  } = architecture;
  const { char_embedding: charEmbedding, w1, b1, w2, b2 } = model.parameters;
  const inputSize = contextSize * embeddingSize + categoryEmbeddingSize;
  const features = new Float32Array(inputSize);

  for (let position = 0; position < contextSize; position += 1) {
    const tokenId = context[position];
    const sourceOffset = tokenId * embeddingSize;
    const targetOffset = position * embeddingSize;
    for (let dimension = 0; dimension < embeddingSize; dimension += 1) {
      features[targetOffset + dimension] = charEmbedding[sourceOffset + dimension];
    }
  }

  if (conditional) {
    const categoryEmbedding = model.parameters.category_embedding;
    const sourceOffset = categoryId * categoryEmbeddingSize;
    const targetOffset = contextSize * embeddingSize;
    for (let dimension = 0; dimension < categoryEmbeddingSize; dimension += 1) {
      features[targetOffset + dimension] = categoryEmbedding[sourceOffset + dimension];
    }
  }

  // Walk row-major weights contiguously. This is materially faster than
  // striding through one output column at a time in JavaScript.
  const hidden = new Float64Array(hiddenSize);
  for (let output = 0; output < hiddenSize; output += 1) {
    hidden[output] = b1[output];
  }
  for (let input = 0; input < inputSize; input += 1) {
    const feature = features[input];
    const weightOffset = input * hiddenSize;
    for (let output = 0; output < hiddenSize; output += 1) {
      hidden[output] += feature * w1[weightOffset + output];
    }
  }
  for (let output = 0; output < hiddenSize; output += 1) {
    hidden[output] = Math.tanh(hidden[output]);
  }

  const logits = new Float32Array(vocabSize);
  const categoryOutput = conditional ? model.parameters.category_output : null;
  for (let output = 0; output < vocabSize; output += 1) {
    logits[output] = b2[output];
  }
  for (let input = 0; input < hiddenSize; input += 1) {
    const hiddenValue = hidden[input];
    const weightOffset = input * vocabSize;
    for (let output = 0; output < vocabSize; output += 1) {
      logits[output] += hiddenValue * w2[weightOffset + output];
    }
  }
  for (let output = 0; output < vocabSize; output += 1) {
    if (categoryOutput) {
      logits[output] += categoryOutput[categoryId * vocabSize + output];
    }
  }
  return logits;
}
