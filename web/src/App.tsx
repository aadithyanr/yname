import { useEffect, useRef, useState } from "react";
import {
  generateNames,
  loadNameIndex,
  randomSeed,
  type Creativity,
  type GeneratedName,
} from "./lib/generator";
import {
  checkNameDomains,
  type DomainResult,
} from "./lib/domains";
import { loadModel } from "./lib/model";

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
    <svg className="name-underline" viewBox="0 0 420 28" preserveAspectRatio="none" aria-hidden="true">
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
  category: string;
  creativity: Creativity;
  seed: number;
} {
  const params = new URLSearchParams(window.location.search);
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
  return { category, creativity, seed };
}

export default function App() {
  const initial = useRef(readInitialState());
  const [category, setCategory] = useState(initial.current.category);
  const [creativity, setCreativity] = useState<Creativity>(initial.current.creativity);
  const [results, setResults] = useState<GeneratedName[]>([]);
  const [isGenerating, setIsGenerating] = useState(true);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [domainResults, setDomainResults] = useState<DomainResult[]>([]);
  const [isCheckingDomains, setIsCheckingDomains] = useState(false);
  const requestId = useRef(0);
  const domainRequestId = useRef(0);

  async function runGeneration(
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
      const kind = nextCategory === "any industry" ? "plain" : "conditional";
      const [model, index] = await Promise.all([loadModel(kind), loadNameIndex()]);
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      const generated = generateNames({
        model,
        index,
        category: nextCategory,
        creativity: nextCreativity,
        count: 5,
        seed,
      });
      if (currentRequest !== requestId.current) return;
      if (generated.length === 0) throw new Error("the batch came back empty");
      setResults(generated);
      const params = new URLSearchParams();
      params.set("seed", String(seed));
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
      initial.current.category,
      initial.current.creativity,
      initial.current.seed,
    );
    // Restore the shareable URL state only once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const activeName = results[0]?.name ?? "";

  useEffect(() => {
    if (!activeName || isGenerating) return;

    const currentRequest = ++domainRequestId.current;
    setIsCheckingDomains(true);
    setDomainResults([]);

    void checkNameDomains(activeName).then((checks) => {
      if (currentRequest === domainRequestId.current) setDomainResults(checks);
    }).finally(() => {
      if (currentRequest === domainRequestId.current) setIsCheckingDomains(false);
    });

    return () => {
      if (currentRequest === domainRequestId.current) domainRequestId.current += 1;
    };
  }, [activeName, isGenerating]);

  function handleCategory(nextCategory: string) {
    setCategory(nextCategory);
    void runGeneration(nextCategory, creativity);
  }

  function handleCreativity(nextCreativity: Creativity) {
    setCreativity(nextCreativity);
    void runGeneration(category, nextCreativity);
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

  async function copyName() {
    if (!results[0]) return;
    await copyText(results[0].name);
    setMessage("name copied");
    window.setTimeout(() => setMessage(""), 1600);
  }

  function promote(result: GeneratedName) {
    domainRequestId.current += 1;
    setDomainResults([]);
    setIsCheckingDomains(false);
    setResults((current) => [
      result,
      ...current.filter((item) => item.name !== result.name),
    ]);
  }

  const displayName = error
    ? error
    : isGenerating
      ? "naming..."
      : results[0]?.name ?? "";
  const nameSizeClass = displayName.length > 18
    ? " is-very-long"
    : displayName.length > 13
      ? " is-long"
      : "";

  return (
    <main className="page-shell">
      <span className="backdrop-letter" aria-hidden="true">y</span>
      <span className="backdrop-word" aria-hidden="true">name?</span>

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
            <p className="dataset-credit">
              dataset scraped with <a href="https://context.dev">context.dev</a>
            </p>
          </div>
        </header>

        <section className="generator">
          <p className="generator-heading">
            the yc startup name generator · trained on 6,194 company names
          </p>

          <div className="name-stage">
            <p className="name-intro">your next startup is called</p>
            <div className="name-lockup">
              <div className="name-content">
                <div className="name-text">
                  <h1 className={`${isGenerating ? "is-generating" : ""}${nameSizeClass}`} aria-live="polite">
                    {displayName}
                  </h1>
                  <NameUnderline />
                </div>
                <button
                  type="button"
                  className={`name-copy${message ? " is-copied" : ""}`}
                  aria-label={message ? "name copied" : "copy name"}
                  data-tooltip={message || "copy name"}
                  onClick={() => void copyName()}
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
            </div>
          </div>

          <div className="control-stack">
            <div className="controls" aria-label="name controls">
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
                onClick={() => void runGeneration()}
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
            <div className="alternatives" aria-label="more names">
              {results.slice(1, 5).map((result, index) => (
                <button type="button" key={result.name} onClick={() => promote(result)} disabled={isGenerating}>
                  <span className="alternative-number">0{index + 1}</span>
                  <span className="alternative-name">{result.name}</span>
                </button>
              ))}
            </div>
          </div>
        </section>

      </section>
    </main>
  );
}
