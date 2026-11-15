/**
 * Who may connect to the local bridge.
 *
 * The threat is a website the user did not choose, scanning 127.0.0.1 for
 * this port. A page cannot set or forge its `Origin` header, and a browser
 * sends `Origin: https://example.com` only for a page it loaded from that
 * site over verified TLS. So an exact match on `Origin` is the whole defence:
 * no token, no pairing, no network call.
 *
 * A connection with no `Origin` is a program on this computer (this
 * extension pushing to Blender, a script). It belongs to the same user as
 * the editor and is accepted.
 *
 * No `vscode` import: this file is tested on its own.
 */

/** Web origins accepted without any setting. */
export const BUILT_IN_ORIGINS: readonly string[] = ["https://ginnung.tech"];

/** `local` = no Origin (a program on this computer). `web` = an allowed page. */
export type Role = "local" | "web";

export interface Allowlist {
  readonly origins: ReadonlySet<string>;
  /** User entries that were ignored, each with the reason, for one warning. */
  readonly rejected: readonly string[];
}

const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1"]);

/**
 * Why `entry` cannot be an allowed origin, or null when it can.
 *
 * An entry is a bare origin exactly as a browser sends it: scheme, host and
 * (non-default) port, nothing else. It must be https; http is accepted only
 * on this computer, for a developer's own dev server. A wildcard, a path or
 * a trailing slash would turn the exact match into a guess.
 */
export function entryProblem(entry: unknown): string | null {
  if (typeof entry !== "string" || entry.trim() === "") {
    return "not a text value";
  }
  if (entry !== entry.trim()) {
    return "has spaces around it";
  }
  if (entry.includes("*")) {
    return "wildcards are not supported; list each origin";
  }
  let url: URL;
  try {
    url = new URL(entry);
  } catch {
    return "not a web address";
  }
  if (url.origin !== entry) {
    return `must be written exactly as "${url.origin === "null" ? "https://host" : url.origin}" (no path, no wildcard, no trailing slash)`;
  }
  if (url.protocol === "https:") {
    return null;
  }
  if (url.protocol === "http:" && LOOPBACK_HOSTS.has(url.hostname)) {
    return null;
  }
  return "must be https (http is accepted only on localhost or 127.0.0.1)";
}

/** The built-in origins plus the user's valid entries. */
export function parseAllowlist(userEntries: readonly unknown[] | undefined): Allowlist {
  const origins = new Set<string>(BUILT_IN_ORIGINS);
  const rejected: string[] = [];
  for (const entry of userEntries ?? []) {
    const problem = entryProblem(entry);
    if (problem === null) {
      origins.add(entry as string);
    } else {
      rejected.push(`${JSON.stringify(entry)}: ${problem}`);
    }
  }
  return { origins, rejected };
}

/**
 * The role of a connection asking to be upgraded, or null to refuse it.
 *
 * `Host` must name this computer's loopback and this port: a page that
 * reached us through a DNS name pointed at 127.0.0.1 (DNS rebinding) carries
 * that name here. `Origin` absent is a local program; present, it must equal
 * an allowlist entry.
 */
export function checkUpgrade(
  host: string | undefined,
  origin: string | undefined,
  port: number,
  allowlist: Allowlist
): Role | null {
  const h = (host ?? "").toLowerCase();
  if (h !== `127.0.0.1:${port}` && h !== `localhost:${port}`) {
    return null;
  }
  if (origin === undefined || origin === "") {
    return "local";
  }
  return allowlist.origins.has(origin) ? "web" : null;
}

/**
 * Whether a pushed file is one the bridge carries: a `.py` or `.ifc` name,
 * and a `fileType` (when given) that agrees with it.
 */
export function allowedFile(filename: string, fileType?: string): boolean {
  const lower = filename.toLowerCase();
  const kind = lower.endsWith(".py") ? "python" : lower.endsWith(".ifc") ? "ifc" : null;
  if (kind === null) {
    return false;
  }
  return fileType === undefined || fileType === kind;
}
