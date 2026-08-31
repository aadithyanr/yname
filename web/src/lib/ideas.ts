import type { Creativity } from "./generator";

type WeightedEntry = [value: string, count: number];

type TemplateGroup = {
  solutions: WeightedEntry[];
  audiences: WeightedEntry[];
};

type IdeaModelPayload = {
  version: number;
  categories: string[];
  knownIdeas: string[];
  globalTemplates: TemplateGroup;
  categoryTemplates: Record<string, TemplateGroup>;
};

export type GeneratedIdea = {
  idea: string;
  nearestKnownIdea: string;
  nearestKnownSimilarity: number;
};

const SETTINGS: Record<
  Creativity,
  { temperature: number; topK: number | null; categoryWeight: number }
> = {
  sensible: { temperature: 0.72, topK: 80, categoryWeight: 16 },
  yc: { temperature: 0.92, topK: 500, categoryWeight: 12 },
  unhinged: { temperature: 1.16, topK: null, categoryWeight: 3 },
};

function tokens(value: string): string[] {
  return value
    .toLowerCase()
    .match(/[a-z0-9]+(?:['-][a-z0-9]+)*|[&+/]|[.,!?;:()]/g) ?? [];
}

function normalizeIdea(value: string): string {
  return tokens(value)
    .filter((token) => /[a-z0-9]/.test(token))
    .join(" ");
}

function displayIdea(solution: string, audience: string): string {
  let value = `${solution} for ${audience}`
    .replace(/\s+([.,!?;:)])/g, "$1")
    .replace(/([(])\s+/g, "$1")
    .replace(/\s*([/+])\s*/g, "$1")
    .replace(/\s+/g, " ")
    .replace(/^[,;:\s-]+|[,;:\s-]+$/g, "");
  if (!value) return "";
  value = value[0].toUpperCase() + value.slice(1);
  if (!/[.!?]$/.test(value)) value += ".";
  return value;
}

function shingles(value: string, requestedSize = 3): Set<string> {
  const words = normalizeIdea(value).split(" ").filter(Boolean);
  if (words.length === 0) return new Set();
  const size = Math.min(requestedSize, words.length);
  const result = new Set<string>();
  for (let index = 0; index <= words.length - size; index += 1) {
    result.add(words.slice(index, index + size).join("\u001f"));
  }
  return result;
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

function ideaSimilarity(left: string, right: string): number {
  const leftKey = normalizeIdea(left);
  const rightKey = normalizeIdea(right);
  if (leftKey === rightKey) return 1;
  const leftShingles = shingles(leftKey);
  const rightShingles = shingles(rightKey);
  let intersection = 0;
  for (const shingle of leftShingles) {
    if (rightShingles.has(shingle)) intersection += 1;
  }
  const union = new Set([...leftShingles, ...rightShingles]).size;
  const jaccard = union ? intersection / union : 0;
  const sequence =
    1 - editDistance(leftKey, rightKey) / Math.max(leftKey.length, rightKey.length, 1);
  return Math.max(jaccard, sequence * 0.82);
}

function looksUsable(value: string): boolean {
  const words = normalizeIdea(value).split(" ").filter(Boolean);
  if (words.length < 5 || words.length > 22) return false;
  if (new Set(words).size / words.length < 0.45) return false;
  const bigrams = words.slice(0, -1).map((word, index) => `${word}\u001f${words[index + 1]}`);
  if (new Set(bigrams).size !== bigrams.length) return false;
  if (/[.,!?;:]{2,}|[,;:]\./.test(value)) return false;
  if ((value.match(/[.!?](?:\s|$)/g) ?? []).length > 1) return false;
  return /[a-z]/i.test(value);
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

class IdeaIndex {
  readonly ideas: string[];
  readonly exact: Set<string>;
  readonly bigramIndex: Map<string, Set<number>>;

  constructor(ideas: string[]) {
    const unique = new Map<string, string>();
    for (const idea of ideas) {
      const key = normalizeIdea(idea);
      if (key && !unique.has(key)) unique.set(key, idea);
    }
    this.ideas = [...unique.values()];
    this.exact = new Set(unique.keys());
    this.bigramIndex = new Map();
    for (let index = 0; index < this.ideas.length; index += 1) {
      for (const bigram of shingles(this.ideas[index], 2)) {
        const group = this.bigramIndex.get(bigram) ?? new Set<number>();
        group.add(index);
        this.bigramIndex.set(bigram, group);
      }
    }
  }

  contains(value: string): boolean {
    return this.exact.has(normalizeIdea(value));
  }

  nearest(value: string): { idea: string; similarity: number } {
    const candidates = new Set<number>();
    for (const bigram of shingles(value, 2)) {
      for (const index of this.bigramIndex.get(bigram) ?? []) candidates.add(index);
    }
    let nearestIdea = "";
    let nearestSimilarity = 0;
    for (const index of candidates) {
      const similarity = ideaSimilarity(value, this.ideas[index]);
      if (similarity > nearestSimilarity) {
        nearestIdea = this.ideas[index];
        nearestSimilarity = similarity;
      }
    }
    return { idea: nearestIdea, similarity: nearestSimilarity };
  }
}

export class IdeaModel {
  readonly payload: IdeaModelPayload;
  readonly index: IdeaIndex;

  constructor(payload: IdeaModelPayload) {
    this.payload = payload;
    this.index = new IdeaIndex(payload.knownIdeas);
  }

  private distribution(
    kind: keyof TemplateGroup,
    category: string,
    categoryWeight: number,
  ): WeightedEntry[] {
    const counts = new Map<string, number>(this.payload.globalTemplates[kind]);
    const categoryEntries = this.payload.categoryTemplates[category]?.[kind] ?? [];
    for (const [value, count] of categoryEntries) {
      counts.set(value, (counts.get(value) ?? 0) + count * categoryWeight);
    }
    return [...counts].sort(
      (left, right) => right[1] - left[1] || left[0].localeCompare(right[0]),
    );
  }

  private static sample(
    distribution: WeightedEntry[],
    random: () => number,
    temperature: number,
    topK: number | null,
  ): string {
    const candidates = topK === null ? distribution : distribution.slice(0, topK);
    const exponent = 1 / Math.max(temperature, 0.000001);
    const weights = candidates.map(([, count]) => Math.pow(count, exponent));
    let cursor = random() * weights.reduce((total, weight) => total + weight, 0);
    for (let index = 0; index < candidates.length; index += 1) {
      cursor -= weights[index];
      if (cursor <= 0) return candidates[index][0];
    }
    return candidates.at(-1)?.[0] ?? "";
  }

  generate(options: {
    category: string;
    creativity: Creativity;
    count: number;
    seed: number;
  }): GeneratedIdea[] {
    const { category, creativity, count, seed } = options;
    const settings = SETTINGS[creativity];
    const random = mulberry32(seed);
    const effectiveCategory = category === "any industry" ? "" : category;
    const solutions = this.distribution(
      "solutions",
      effectiveCategory,
      settings.categoryWeight,
    );
    const audiences = this.distribution(
      "audiences",
      effectiveCategory,
      settings.categoryWeight,
    );
    const results: GeneratedIdea[] = [];
    const seen = new Set<string>();

    for (let attempt = 0; attempt < Math.max(200, count * 500); attempt += 1) {
      const solution = IdeaModel.sample(
        solutions,
        random,
        settings.temperature,
        settings.topK,
      );
      const audience = IdeaModel.sample(
        audiences,
        random,
        settings.temperature,
        settings.topK,
      );
      const idea = displayIdea(solution, audience);
      const key = normalizeIdea(idea);
      if (!key || seen.has(key) || this.index.contains(idea) || !looksUsable(idea)) continue;
      const nearest = this.index.nearest(idea);
      if (nearest.similarity >= 0.72) continue;
      if (results.some((result) => ideaSimilarity(idea, result.idea) >= 0.66)) continue;
      seen.add(key);
      results.push({
        idea,
        nearestKnownIdea: nearest.idea,
        nearestKnownSimilarity: nearest.similarity,
      });
      if (results.length >= count) break;
    }
    return results;
  }
}

let modelPromise: Promise<IdeaModel> | null = null;

export function loadIdeaModel(): Promise<IdeaModel> {
  if (!modelPromise) {
    modelPromise = fetch("/models/ideas.json")
      .then((response) => {
        if (!response.ok) throw new Error("could not load the idea model");
        return response.json() as Promise<IdeaModelPayload>;
      })
      .then((payload) => new IdeaModel(payload));
  }
  return modelPromise;
}
