import { forward, type NameModel } from "./model";

export type Creativity = "sensible" | "yc" | "unhinged";

export type GeneratedName = {
  name: string;
  score: number;
  nearestKnownName: string;
  nearestKnownSimilarity: number;
};

type KnownNamesPayload = {
  names: string[];
  canonicalNames: string[];
};

const START = "<START>";
const END = "<END>";
const GENERIC_DESCRIPTORS = new Set([
  "ai",
  "bio",
  "bioscience",
  "biosciences",
  "health",
  "industries",
  "industry",
  "labs",
  "medical",
  "robotics",
  "systems",
  "technologies",
  "technology",
]);

const BLOCKED_FRAGMENTS = [
  "fuck",
  "shit",
  "bitch",
  "cunt",
  "dick",
  "cock",
  "pussy",
  "nigger",
  "nigga",
  "faggot",
  "slut",
  "whore",
];

const TEMPERATURES: Record<Creativity, number[]> = {
  sensible: [0.66, 0.72, 0.78, 0.84],
  yc: [0.72, 0.84, 0.96, 1.08],
  unhinged: [0.88, 1, 1.12, 1.24],
};

function canonicalKey(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function trigrams(value: string): Set<string> {
  const padded = `^^${canonicalKey(value)}$$`;
  const grams = new Set<string>();
  for (let index = 0; index <= padded.length - 3; index += 1) {
    grams.add(padded.slice(index, index + 3));
  }
  return grams;
}

function editDistance(left: string, right: string): number {
  if (left.length < right.length) return editDistance(right, left);
  let previous = Array.from({ length: right.length + 1 }, (_, index) => index);
  for (let leftIndex = 1; leftIndex <= left.length; leftIndex += 1) {
    const current = [leftIndex];
    for (let rightIndex = 1; rightIndex <= right.length; rightIndex += 1) {
      current.push(
        Math.min(
          current[current.length - 1] + 1,
          previous[rightIndex] + 1,
          previous[rightIndex - 1] +
            (left[leftIndex - 1] === right[rightIndex - 1] ? 0 : 1),
        ),
      );
    }
    previous = current;
  }
  return previous[previous.length - 1];
}

function similarity(left: string, right: string): number {
  if (left === right) return 1;
  const distance = editDistance(left, right);
  return 1 - distance / Math.max(left.length, right.length, 1);
}

function descriptiveRoot(value: string): string {
  const words = value.toLowerCase().split(/\s+/).filter(Boolean);
  while (words.length > 1 && GENERIC_DESCRIPTORS.has(words[words.length - 1])) {
    words.pop();
  }
  return words.join(" ");
}

function descriptorIsMisspelled(value: string): boolean {
  for (const word of value.toLowerCase().split(/\s+/)) {
    if (GENERIC_DESCRIPTORS.has(word) || word.length < 4) continue;
    for (const descriptor of GENERIC_DESCRIPTORS) {
      if (
        Math.abs(word.length - descriptor.length) <= 1 &&
        editDistance(word, descriptor) === 1
      ) {
        return true;
      }
    }
  }
  return false;
}

function looksUsable(value: string): boolean {
  if (value.length < 3 || value.length > 20) return false;
  if (!/[a-z0-9]/.test(value[0]) || !/[a-z0-9]/.test(value[value.length - 1])) {
    return false;
  }
  if ((value.match(/[a-z]/g) ?? []).length < 2) return false;
  if (/(.)\1\1/.test(value)) return false;
  if (/[ ./'-]{2}/.test(value)) return false;
  if ((value.match(/ /g) ?? []).length > 2) return false;
  const letters = value.replace(/[^a-z]/g, "");
  if (letters.length >= 5 && !/[aeiouy]/.test(letters)) return false;
  if (/[bcdfghjklmnpqrstvwxz]{6}/.test(letters)) return false;
  if (BLOCKED_FRAGMENTS.some((fragment) => letters.includes(fragment))) return false;
  if (GENERIC_DESCRIPTORS.has(value) || descriptorIsMisspelled(value)) return false;
  return true;
}

function mulberry32(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let result = value;
    result = Math.imul(result ^ (result >>> 15), result | 1);
    result ^= result + Math.imul(result ^ (result >>> 7), result | 61);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
}

function logSumExp(values: Float32Array): number {
  let maximum = -Infinity;
  for (const value of values) maximum = Math.max(maximum, value);
  let total = 0;
  for (const value of values) total += Math.exp(value - maximum);
  return maximum + Math.log(total);
}

function sampleToken(
  logits: Float32Array,
  temperature: number,
  random: () => number,
  blockedIds: Set<number>,
): number {
  let maximum = -Infinity;
  for (let index = 0; index < logits.length; index += 1) {
    if (!blockedIds.has(index)) maximum = Math.max(maximum, logits[index] / temperature);
  }
  const weights = new Float64Array(logits.length);
  let total = 0;
  for (let index = 0; index < logits.length; index += 1) {
    if (blockedIds.has(index)) continue;
    weights[index] = Math.exp(logits[index] / temperature - maximum);
    total += weights[index];
  }
  let cursor = random() * total;
  for (let index = 0; index < weights.length; index += 1) {
    cursor -= weights[index];
    if (cursor <= 0) return index;
  }
  return weights.length - 1;
}

function sampleName(
  model: NameModel,
  categoryId: number,
  temperature: number,
  random: () => number,
): { name: string; logProbability: number } {
  const startId = model.tokenToId.get(START)!;
  const endId = model.tokenToId.get(END)!;
  const context = Array(model.manifest.architecture.context_size).fill(startId);
  const characters: string[] = [];
  let logProbability = 0;

  for (let step = 0; step < 22; step += 1) {
    const logits = forward(model, context, categoryId);
    const blocked = new Set([startId]);
    if (characters.length < 3) blocked.add(endId);
    const tokenId = sampleToken(logits, temperature, random, blocked);
    logProbability += logits[tokenId] - logSumExp(logits);
    if (tokenId === endId) break;
    characters.push(model.manifest.tokens[tokenId]);
    context.shift();
    context.push(tokenId);
  }
  const name = characters.join("").trim();
  return { name, logProbability: logProbability / Math.max(1, characters.length + 1) };
}

export class NameIndex {
  readonly names: Set<string>;
  readonly roots: Set<string>;
  readonly canonicalToName: Map<string, string>;
  readonly trigramIndex: Map<string, Set<string>>;
  readonly lengthIndex: Map<number, Set<string>>;

  constructor(payload: KnownNamesPayload) {
    this.names = new Set(payload.names);
    this.roots = new Set(payload.names.map(descriptiveRoot));
    this.canonicalToName = new Map();
    this.trigramIndex = new Map();
    this.lengthIndex = new Map();
    for (let index = 0; index < payload.canonicalNames.length; index += 1) {
      const canonical = payload.canonicalNames[index];
      this.canonicalToName.set(canonical, canonical);
      const lengthGroup = this.lengthIndex.get(canonical.length) ?? new Set();
      lengthGroup.add(canonical);
      this.lengthIndex.set(canonical.length, lengthGroup);
      for (const gram of trigrams(canonical)) {
        const group = this.trigramIndex.get(gram) ?? new Set();
        group.add(canonical);
        this.trigramIndex.set(gram, group);
      }
    }
  }

  nearest(value: string): { name: string; similarity: number } {
    const canonical = canonicalKey(value);
    const pool = new Set<string>();
    for (const gram of trigrams(value)) {
      for (const candidate of this.trigramIndex.get(gram) ?? []) pool.add(candidate);
    }
    if (canonical.length <= 5) {
      for (let length = Math.max(1, canonical.length - 2); length <= canonical.length + 2; length += 1) {
        for (const candidate of this.lengthIndex.get(length) ?? []) pool.add(candidate);
      }
    }
    let nearestName = "";
    let nearestScore = 0;
    for (const candidate of pool) {
      const score = similarity(canonical, candidate);
      if (score > nearestScore) {
        nearestScore = score;
        nearestName = this.canonicalToName.get(candidate) ?? candidate;
      }
    }
    return { name: nearestName, similarity: nearestScore };
  }
}

let indexPromise: Promise<NameIndex> | null = null;

export function loadNameIndex(): Promise<NameIndex> {
  if (!indexPromise) {
    indexPromise = fetch("/models/known-names.json")
      .then((response) => {
        if (!response.ok) throw new Error("could not load known-name index");
        return response.json() as Promise<KnownNamesPayload>;
      })
      .then((payload) => new NameIndex(payload));
  }
  return indexPromise;
}

export function generateNames(options: {
  model: NameModel;
  index: NameIndex;
  category: string;
  creativity: Creativity;
  count: number;
  seed: number;
}): GeneratedName[] {
  const { model, index, category, creativity, count, seed } = options;
  const random = mulberry32(seed);
  const categoryId = Math.max(0, model.manifest.categories.indexOf(category));
  const temperatures = TEMPERATURES[creativity];
  const attempts = 110;
  const candidates = new Map<string, GeneratedName>();

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const temperature = temperatures[attempt % temperatures.length];
    const sampled = sampleName(model, categoryId, temperature, random);
    const key = sampled.name.toLowerCase();
    const root = descriptiveRoot(key);
    if (
      candidates.has(key) ||
      index.names.has(key) ||
      index.roots.has(root) ||
      !looksUsable(key)
    ) {
      continue;
    }
    const nearest = index.nearest(key);
    const threshold = canonicalKey(key).length <= 5 ? 0.8 : 0.84;
    if (nearest.similarity >= threshold) continue;

    const words = key.split(/\s+/);
    const descriptorCount = words.filter((word) => GENERIC_DESCRIPTORS.has(word)).length;
    const score =
      sampled.logProbability -
      Math.abs(key.length - 8) * 0.018 -
      descriptorCount * 0.28 -
      Math.max(0, words.length - 1) * 0.06;
    candidates.set(key, {
      name: key,
      score,
      nearestKnownName: nearest.name,
      nearestKnownSimilarity: nearest.similarity,
    });
  }

  const pool = [...candidates.values()].sort((left, right) => right.score - left.score);
  const selected: GeneratedName[] = [];
  while (pool.length > 0 && selected.length < count) {
    let bestIndex = 0;
    let bestScore = -Infinity;
    for (let index = 0; index < Math.min(pool.length, 100); index += 1) {
      const candidateGrams = trigrams(pool[index].name);
      let maximumSimilarity = 0;
      for (const prior of selected) {
        const priorGrams = trigrams(prior.name);
        const intersection = [...candidateGrams].filter((gram) => priorGrams.has(gram)).length;
        const union = new Set([...candidateGrams, ...priorGrams]).size;
        maximumSimilarity = Math.max(maximumSimilarity, union ? intersection / union : 0);
      }
      const adjustedScore = pool[index].score - maximumSimilarity * 0.35;
      if (adjustedScore > bestScore) {
        bestScore = adjustedScore;
        bestIndex = index;
      }
    }
    selected.push(pool.splice(bestIndex, 1)[0]);
  }
  return selected;
}

export function randomSeed(): number {
  const value = new Uint32Array(1);
  crypto.getRandomValues(value);
  return value[0];
}
