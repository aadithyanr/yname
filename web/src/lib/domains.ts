export type DomainAvailability = "available" | "taken" | "unknown";

export type DomainResult = {
  domain: string;
  availability: DomainAvailability;
};

const RDAP_REGISTRIES = {
  verisign: "https://rdap.verisign.com/com/v1/",
  identityDigital: "https://rdap.identitydigital.services/rdap/",
  google: "https://pubapi.registry.google/rdap/",
} as const;

const TLD_ENDPOINTS: Record<string, string> = {
  com: RDAP_REGISTRIES.verisign,
  ai: RDAP_REGISTRIES.identityDigital,
  dev: RDAP_REGISTRIES.google,
  app: RDAP_REGISTRIES.google,
  finance: RDAP_REGISTRIES.identityDigital,
  money: RDAP_REGISTRIES.identityDigital,
  care: RDAP_REGISTRIES.identityDigital,
  clinic: RDAP_REGISTRIES.identityDigital,
  academy: RDAP_REGISTRIES.identityDigital,
  school: RDAP_REGISTRIES.identityDigital,
  estate: RDAP_REGISTRIES.identityDigital,
  properties: RDAP_REGISTRIES.identityDigital,
  solutions: RDAP_REGISTRIES.identityDigital,
  engineering: RDAP_REGISTRIES.identityDigital,
  industries: RDAP_REGISTRIES.identityDigital,
  life: RDAP_REGISTRIES.identityDigital,
};

const CATEGORY_TLDS: Record<string, string[]> = {
  Fintech: ["com", "ai", "finance", "money"],
  Healthcare: ["com", "ai", "care", "clinic"],
  Education: ["com", "ai", "academy", "school"],
  "Real Estate and Construction": ["com", "ai", "estate", "properties"],
  B2B: ["com", "ai", "dev", "solutions"],
  Industrials: ["com", "ai", "engineering", "industries"],
  Consumer: ["com", "ai", "app", "life"],
};

const DEFAULT_TLDS = ["com", "ai", "dev"];

function toDomainStem(name: string): string {
  return name
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/&/g, "and")
    .replace(/[^a-z0-9]/g, "")
    .slice(0, 63);
}

async function checkDomain(
  domain: string,
  endpoint: string,
): Promise<DomainResult> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 8000);

  try {
    const response = await fetch(`${endpoint}domain/${encodeURIComponent(domain)}`, {
      headers: { Accept: "application/rdap+json" },
      signal: controller.signal,
    });

    if (response.ok) return { domain, availability: "taken" };
    if (response.status === 404) return { domain, availability: "available" };
    return { domain, availability: "unknown" };
  } catch {
    return { domain, availability: "unknown" };
  } finally {
    window.clearTimeout(timeout);
  }
}

export async function checkNameDomains(
  name: string,
  category = "any industry",
): Promise<DomainResult[]> {
  const stem = toDomainStem(name);
  if (!stem) return [];

  const tlds = CATEGORY_TLDS[category] ?? DEFAULT_TLDS;

  return Promise.all(
    tlds.map((tld) => {
      const endpoint = TLD_ENDPOINTS[tld] ?? RDAP_REGISTRIES.identityDigital;
      return checkDomain(`${stem}.${tld}`, endpoint);
    }),
  );
}
