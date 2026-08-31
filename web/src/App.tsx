import { useEffect, useRef, useState } from "react";
import {
  generateNames,
  loadNameIndex,
  randomSeed,
  type Creativity,
} from "./lib/generator";
import {
  checkNameDomains,
  type DomainResult,
} from "./lib/domains";
import { loadModel } from "./lib/model";
import { loadIdeaModel } from "./lib/ideas";

type GeneratorMode = "names" | "ideas";

const CATEGORIES = [
  "any industry",
  "B2B",
  "Consumer",
  "Education",
  "Fintech",
  "Healthcare",
  "Industrials",
  "Real Estate and Construction",
  "Other",
];

const CREATIVITY: Array<{ value: Creativity; label: string }> = [
  { value: "sensible", label: "sensible" },
  { value: "yc", label: "yc mode" },
  { value: "unhinged", label: "unhinged" },
];

function ChevronIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden="true">
      <path d="m4.5 6.25 3.5 3.5 3.5-3.5" />
    </svg>
  );
}

function CopyIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="8" y="8" width="11" height="11" rx="2" />
      <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m5 12.5 4.2 4.2L19 7" />
    </svg>
  );
}

function NameUnderline() {
  return (
    <svg
      className="name-underline"
      viewBox="0 0 420 28"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <path d="M8 17c82-13 161-10 232-3 61 6 116 5 172-4" />
      <path d="M21 23c93-8 183-5 268-1 45 2 82 0 112-4" />
    </svg>
  );
}

function BatchArrow() {
  return (
    <svg className="batch-arrow" viewBox="0 0 52 42" aria-hidden="true">
      <path d="M4 6c8 24 21 29 39 23" />
      <path d="m35 22 9 7-8 8" />
    </svg>
  );
}

function readInitialState(): {
  mode: GeneratorMode;
  category: string;
  creativity: Creativity;
  seed: number;
} {
  const params = new URLSearchParams(window.location.search);
  const mode: GeneratorMode = params.get("mode") === "ideas" ? "ideas" : "names";
  const requestedCategory = params.get("category") ?? "any industry";
  const category = CATEGORIES.includes(requestedCategory)
    ? requestedCategory
    : "any industry";
  const requestedCreativity = params.get("vibe") as Creativity | null;
  const creativity = CREATIVITY.some((option) => option.value === requestedCreativity)
    ? requestedCreativity!
    : "yc";
  const parsedSeed = Number(params.get("seed"));
  const seed = Number.isSafeInteger(parsedSeed) && parsedSeed >= 0 ? parsedSeed : randomSeed();
  return { mode, category, creativity, seed };
}

export default function App() {
  const initial = useRef(readInitialState());
  const [mode, setMode] = useState<GeneratorMode>(initial.current.mode);
  const [category, setCategory] = useState(initial.current.category);
  const [creativity, setCreativity] = useState<Creativity>(initial.current.creativity);
  const [results, setResults] = useState<string[]>([]);
  const [isGenerating, setIsGenerating] = useState(true);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [domainResults, setDomainResults] = useState<DomainResult[]>([]);
  const [isCheckingDomains, setIsCheckingDomains] = useState(false);
  const requestId = useRef(0);
  const domainRequestId = useRef(0);

  async function runGeneration(
    nextMode = mode,
    nextCategory = category,
    nextCreativity = creativity,
    seed = randomSeed(),
  ) {
    const currentRequest = ++requestId.current;
    domainRequestId.current += 1;
    setIsGenerating(true);
    setIsCheckingDomains(false);
    setDomainResults([]);
    setError("");
    setMessage("");
    try {
      let generated: string[];
      if (nextMode === "ideas") {
        const ideaModel = await loadIdeaModel();
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        generated = ideaModel.generate({
          category: nextCategory,
          creativity: nextCreativity,
          count: 5,
          seed,
        }).map((result) => result.idea);
      } else {
        const kind = nextCategory === "any industry" ? "plain" : "conditional";
        const [nameModel, index] = await Promise.all([loadModel(kind), loadNameIndex()]);
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
        generated = generateNames({
          model: nameModel,
          index,
          category: nextCategory,
          creativity: nextCreativity,
          count: 5,
          seed,
        }).map((result) => result.name);
      }
      if (currentRequest !== requestId.current) return;
      if (generated.length === 0) throw new Error("the batch came back empty");
      setResults(generated);
      const params = new URLSearchParams();
      params.set("seed", String(seed));
      if (nextMode === "ideas") params.set("mode", "ideas");
      if (nextCategory !== "any industry") params.set("category", nextCategory);
      if (nextCreativity !== "yc") params.set("vibe", nextCreativity);
      window.history.replaceState(null, "", `${window.location.pathname}?${params}`);
    } catch (caught) {
      if (currentRequest !== requestId.current) return;
      setError(caught instanceof Error ? caught.message : "something went wrong");
    } finally {
      if (currentRequest === requestId.current) setIsGenerating(false);
    }
  }

  useEffect(() => {
    void runGeneration(
      initial.current.mode,
      initial.current.category,
      initial.current.creativity,
      initial.current.seed,
    );
    // Restore the shareable URL state only once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const activeName = mode === "names" ? results[0] ?? "" : "";

  useEffect(() => {
    if (mode !== "names" || !activeName || isGenerating) return;

    const currentRequest = ++domainRequestId.current;
    setIsCheckingDomains(true);
    setDomainResults([]);

    void checkNameDomains(activeName, category).then((checks) => {
      if (currentRequest === domainRequestId.current) setDomainResults(checks);
    }).finally(() => {
      if (currentRequest === domainRequestId.current) setIsCheckingDomains(false);
    });

    return () => {
      if (currentRequest === domainRequestId.current) domainRequestId.current += 1;
    };
  }, [activeName, category, isGenerating, mode]);

  function handleMode(nextMode: GeneratorMode) {
    if (nextMode === mode) return;
    setMode(nextMode);
    void runGeneration(nextMode, category, creativity);
  }

  function handleCategory(nextCategory: string) {
    setCategory(nextCategory);
    void runGeneration(mode, nextCategory, creativity);
  }

  function handleCreativity(nextCreativity: Creativity) {
    setCreativity(nextCreativity);
    void runGeneration(mode, category, nextCreativity);
  }

  async function copyText(value: string) {
    try {
      await navigator.clipboard.writeText(value);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = value;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
  }

  async function copyResult() {
    if (!results[0]) return;
    await copyText(results[0]);
    setMessage(`${mode === "names" ? "name" : "idea"} copied`);
    window.setTimeout(() => setMessage(""), 1600);
  }

  function promote(result: string) {
    domainRequestId.current += 1;
    setDomainResults([]);
    setIsCheckingDomains(false);
    setResults((current) => [
      result,
      ...current.filter((item) => item !== result),
    ]);
  }

  const displayText = error
    ? error
    : isGenerating
      ? mode === "names" ? "naming..." : "thinking..."
      : results[0] ?? "";
  const textSizeClass = mode === "ideas"
    ? " is-idea"
    : displayText.length > 18
      ? " is-very-long"
      : displayText.length > 13
        ? " is-long"
        : "";

  return (
    <main className={`page-shell is-${mode}`}>
      <span className="backdrop-letter" aria-hidden="true">y</span>
      <span className="backdrop-word" aria-hidden="true">
        {mode === "names" ? "name?" : "idea?"}
      </span>

      <section className="studio-card">
        <header className="site-header">
          <a className="brand" href="/" aria-label="yname home">
            <span className="brand-mark">y</span>
            <span>yname</span>
          </a>
          <div className="header-meta">
            <a className="maker-link" href="https://aadithyanr.dev">
              by aadithyan ↗
            </a>
          </div>
        </header>

        <section className="generator">
          <div className="mode-switch" role="group" aria-label="generator mode">
            <button
              type="button"
              className={mode === "names" ? "is-active" : ""}
              aria-pressed={mode === "names"}
              onClick={() => handleMode("names")}
              disabled={isGenerating}
            >
              names
            </button>
            <button
              type="button"
              className={mode === "ideas" ? "is-active" : ""}
              aria-pressed={mode === "ideas"}
              onClick={() => handleMode("ideas")}
              disabled={isGenerating}
            >
              ideas
            </button>
          </div>
          <p className="generator-heading">
            {mode === "names"
              ? "the yc startup name generator · trained on 6,194 company names"
              : "the yc startup idea generator · trained on 5,411 company one-liners"}
          </p>

          <div className={`name-stage${mode === "ideas" ? " is-idea" : ""}`}>
            <p className="name-intro">
              {mode === "names"
                ? "your next startup is called"
                : "your next startup could be"}
            </p>
            <div className="name-lockup">
              <div className="name-content">
                <div className="name-text">
                  <h1
                    className={`${isGenerating ? "is-generating" : ""}${textSizeClass}`}
                    aria-live="polite"
                  >
                    {displayText}
                  </h1>
                  <NameUnderline />
                </div>
                <button
                  type="button"
                  className={`name-copy${message ? " is-copied" : ""}`}
                  aria-label={message || `copy ${mode === "names" ? "name" : "idea"}`}
                  data-tooltip={message || `copy ${mode === "names" ? "name" : "idea"}`}
                  onClick={() => void copyResult()}
                  disabled={!results[0] || isGenerating}
                >
                  {message ? <CheckIcon /> : <CopyIcon />}
                </button>
              </div>
            </div>

            <div className="domain-area" aria-live="polite" aria-busy={isCheckingDomains}>
              {domainResults.length > 0 && (
                <div className="domain-results" aria-label="domain availability">
                  {domainResults.map((result) => (
                    <span
                      className="domain-result"
                      key={result.domain}
                      aria-label={`${result.domain} ${
                        result.availability === "available"
                          ? "likely available"
                          : result.availability
                      }`}
                      title={
                        result.availability === "available"
                          ? "Likely available — confirm before purchasing"
                          : result.availability
                      }
                    >
                      <span>{result.domain}</span>
                      <span
                        className={`domain-status is-${result.availability}`}
                        aria-hidden="true"
                      >
                        {result.availability === "available"
                          ? "✓"
                          : result.availability === "taken"
                            ? "×"
                            : "?"}
                      </span>
                    </span>
                  ))}
                </div>
              )}
              {mode === "ideas" && !isGenerating && !error && (
                <span className="idea-note">generated locally · no llm · no backend</span>
              )}
            </div>
          </div>

          <div className="control-stack">
            <div
              className="controls"
              aria-label={`${mode === "names" ? "name" : "idea"} controls`}
            >
              <label className="select-control">
                <span className="sr-only">industry</span>
                <span className="select-shell">
                  <select
                    value={category}
                    aria-label="industry"
                    onChange={(event) => handleCategory(event.target.value)}
                    disabled={isGenerating}
                  >
                    {CATEGORIES.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                  <ChevronIcon />
                </span>
              </label>

              <span className="control-divider" aria-hidden="true" />

              <label className="select-control style-control">
                <span className="sr-only">style</span>
                <span className="select-shell">
                  <select
                    value={creativity}
                    aria-label="style"
                    onChange={(event) => handleCreativity(event.target.value as Creativity)}
                    disabled={isGenerating}
                  >
                    {CREATIVITY.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <ChevronIcon />
                </span>
              </label>

              <button
                type="button"
                className="generate-action"
                onClick={() => void runGeneration(mode, category, creativity)}
                disabled={isGenerating}
              >
                {isGenerating ? "making..." : "another"}
              </button>

            </div>
          </div>

          <div className="batch-strip">
            <div className="batch-heading">
              <span>also in<br />the batch</span>
              <BatchArrow />
            </div>
            <div
              className={`alternatives${mode === "ideas" ? " is-ideas" : ""}`}
              aria-label={`more ${mode}`}
            >
              {results.slice(1, 5).map((result, index) => (
                <button
                  type="button"
                  key={result}
                  onClick={() => promote(result)}
                  disabled={isGenerating}
                >
                  <span className="alternative-number">0{index + 1}</span>
                  <span className="alternative-name">{result}</span>
                </button>
              ))}
            </div>
          </div>
        </section>

      </section>

      <a
        className="context-cookie"
        href="https://context.dev"
        aria-label="built using Context.dev"
      >
        <img src="https://webdog.ai/fortunecookie.png" alt="built using Context.dev" />
      </a>
    </main>
  );
}
