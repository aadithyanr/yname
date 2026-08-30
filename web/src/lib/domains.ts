export type DomainAvailability = "available" | "taken" | "unknown";

export type DomainResult = {
  domain: string;
  availability: DomainAvailability;
};

const RDAP_SERVERS = {
  com: "https://rdap.verisign.com/com/v1/",
  ai: "https://rdap.identitydigital.services/rdap/",
  dev: "https://pubapi.registry.google/rdap/",
} as const;

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

export async function checkNameDomains(name: string): Promise<DomainResult[]> {
  const stem = toDomainStem(name);
  if (!stem) return [];

  return Promise.all(
    Object.entries(RDAP_SERVERS).map(([tld, endpoint]) =>
      checkDomain(`${stem}.${tld}`, endpoint),
    ),
  );
}
