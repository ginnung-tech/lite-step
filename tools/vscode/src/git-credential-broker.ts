/**
 * Loopback credential broker and 2-tier `git-credential` resolution logic.
 *
 * Keeps the user's long-lived Ginnung client token inside VS Code SecretStorage
 * while serving short-lived (15-min) single-repo Artifacts tokens to local
 * `git pull/push ginnung` invocations (VS Code Git UI, integrated terminal, or
 * external coding agents).
 */

import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as http from "node:http";
import * as os from "node:os";
import * as path from "node:path";
import {
  CANONICAL_PROJECT_RE,
  DEFAULT_API_BASE_URL,
  type GitTokenRequest,
  type GitTokenResponse,
  isLoopbackHostname,
  mintGitToken,
  readGinnungRemoteLink,
  stripTokenQuery,
  validateApiBaseUrl,
  validateProjectId,
} from "./git-remote";

export interface BrokerFile {
  port: number;
  secret: string;
  pid: number;
  apiBaseUrl: string;
}

export interface StandaloneTokenFile {
  token: string;
  apiBaseUrl?: string;
}

export interface CredentialQuery {
  protocol?: string;
  host?: string;
  path?: string;
  username?: string;
}

export function getGinnungConfigDir(overrideDir?: string): string {
  const candidate = overrideDir || process.env.GINNUNG_CONFIG_DIR;
  if (candidate?.trim()) {
    return path.resolve(candidate.trim());
  }
  return path.join(os.homedir(), ".ginnung");
}

export function getBrokerFilePath(configDir?: string): string {
  return path.join(getGinnungConfigDir(configDir), "git-broker.json");
}

export function getStandaloneTokenFilePath(configDir?: string): string {
  return path.join(getGinnungConfigDir(configDir), "client-token.json");
}

function constantTimeEqual(a: string, b: string): boolean {
  const ab = Buffer.from(a, "utf8");
  const bb = Buffer.from(b, "utf8");
  const max = Math.max(ab.length, bb.length);
  let diff = ab.length ^ bb.length;
  for (let i = 0; i < max; i++) {
    diff |= (ab[i] ?? 0) ^ (bb[i] ?? 0);
  }
  return diff === 0;
}

export const TRUSTED_API_ORIGINS: ReadonlySet<string> = new Set([
  "https://api.ginnung.tech",
  "https://staging-api.grab.technology",
]);

export function isTrustedStandaloneApiBaseUrl(
  candidate: string,
  env: NodeJS.ProcessEnv = process.env,
  standaloneApiBaseUrl?: string
): boolean {
  let normalized: string;
  try {
    normalized = validateApiBaseUrl(candidate);
  } catch {
    return false;
  }
  if (TRUSTED_API_ORIGINS.has(normalized)) {
    return true;
  }
  for (const userConfigured of [env.GINNUNG_API_BASE_URL, standaloneApiBaseUrl]) {
    const raw = userConfigured?.trim();
    if (!raw) continue;
    try {
      if (validateApiBaseUrl(raw) === normalized) {
        return true;
      }
    } catch {
      // Ignore malformed env/config entry
    }
  }
  return false;
}

export class GitCredentialBroker {
  private readonly _getClientToken: () => Promise<string | undefined>;
  private readonly _getApiBaseUrl: () => string;
  private readonly _getAllowedProjects?: () => Promise<ReadonlySet<string>>;
  private readonly _configDir: string;
  private readonly _fetchImpl: typeof fetch;
  private readonly _cache = new Map<string, GitTokenResponse>();
  private _server: http.Server | null = null;
  private _brokerFile: BrokerFile | null = null;

  constructor(opts: {
    getClientToken: () => Promise<string | undefined>;
    getApiBaseUrl: () => string;
    getAllowedProjects?: () => Promise<ReadonlySet<string>>;
    configDir?: string;
    fetchImpl?: typeof fetch;
  }) {
    this._getClientToken = opts.getClientToken;
    this._getApiBaseUrl = opts.getApiBaseUrl;
    this._getAllowedProjects = opts.getAllowedProjects;
    this._configDir = getGinnungConfigDir(opts.configDir);
    this._fetchImpl = opts.fetchImpl ?? fetch;
  }

  get brokerFile(): BrokerFile | null {
    return this._brokerFile;
  }

  clearCache(): void {
    this._cache.clear();
  }

  async start(): Promise<BrokerFile> {
    if (this._server && this._brokerFile) {
      return this._brokerFile;
    }

    const secret = crypto.randomBytes(32).toString("hex");
    const server = http.createServer((req, res) => {
      void this._handleRequest(req, res, secret);
    });

    await new Promise<void>((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", () => {
        server.off("error", reject);
        resolve();
      });
    });

    const address = server.address();
    if (!address || typeof address === "string") {
      server.close();
      throw new Error("Failed to bind loopback Git credential broker");
    }

    let apiBaseUrl = DEFAULT_API_BASE_URL;
    try {
      apiBaseUrl = validateApiBaseUrl(this._getApiBaseUrl() || DEFAULT_API_BASE_URL);
    } catch {
      apiBaseUrl = DEFAULT_API_BASE_URL;
    }

    const meta: BrokerFile = {
      port: address.port,
      secret,
      pid: process.pid,
      apiBaseUrl,
    };

    fs.mkdirSync(this._configDir, { recursive: true, mode: 0o700 });
    const filePath = getBrokerFilePath(this._configDir);
    fs.writeFileSync(filePath, JSON.stringify(meta, null, 2) + "\n", { mode: 0o600 });

    this._server = server;
    this._brokerFile = meta;
    return meta;
  }

  async stop(): Promise<void> {
    const filePath = getBrokerFilePath(this._configDir);
    const meta = this._brokerFile;
    this._brokerFile = null;
    this._cache.clear();

    if (meta) {
      try {
        if (fs.existsSync(filePath)) {
          const current = JSON.parse(fs.readFileSync(filePath, "utf-8")) as Partial<BrokerFile>;
          if (current.pid === meta.pid && current.secret === meta.secret) {
            fs.unlinkSync(filePath);
          }
        }
      } catch {
        // Ignore cleanup failures
      }
    }

    const server = this._server;
    this._server = null;
    if (server) {
      await new Promise<void>((resolve) => {
        server.close(() => resolve());
      });
    }
  }

  private async _readRequestBody(req: http.IncomingMessage): Promise<{ ok: true; body: Record<string, unknown> } | { ok: false; status: number; code: string; message: string }> {
    let raw = "";
    try {
      for await (const chunk of req) {
        raw += typeof chunk === "string" ? chunk : (chunk as Buffer).toString("utf-8");
        if (raw.length > 32 * 1024) {
          return { ok: false, status: 413, code: "payload_too_large", message: "Request body too large" };
        }
      }
    } catch {
      return { ok: false, status: 400, code: "bad_request", message: "Failed reading body" };
    }
    try {
      const body = (raw.trim() ? JSON.parse(raw) : {}) as Record<string, unknown>;
      return { ok: true, body };
    } catch {
      return { ok: false, status: 400, code: "bad_request", message: "Invalid JSON body" };
    }
  }

  private _parseTokenRequest(body: Record<string, unknown>): GitTokenRequest {
    const scope: "read" | "write" = body.scope === "read" ? "read" : "write";
    if (body.target === "service") {
      if (body.repo === "pack") {
        return { target: "service", repo: "pack", scope: "read" };
      }
      if (body.repo === "core") {
        return { target: "service", repo: "core", scope: "read" };
      }
      throw new Error("Invalid service repo");
    }
    const pidRaw = typeof body.projectId === "string" ? body.projectId : "";
    return { target: "project", projectId: validateProjectId(pidRaw), scope };
  }

  private _getCachedToken(cacheKey: string): GitTokenResponse | null {
    const cached = this._cache.get(cacheKey);
    if (!cached) return null;
    const expMs = Date.parse(cached.expiresAt);
    if (Number.isFinite(expMs) && expMs - Date.now() > 60_000) {
      return cached;
    }
    this._cache.delete(cacheKey);
    return null;
  }

  private async _isProjectAllowed(projectId: string): Promise<boolean> {
    if (!this._getAllowedProjects) {
      return true;
    }
    const allowed = await this._getAllowedProjects();
    if (allowed.has(projectId)) {
      return true;
    }
    return projectId === "default" && allowed.has("P0001");
  }

  private _parseVerifiedApiBase(body: Record<string, unknown>): { ok: true; apiBaseUrl: string } | { ok: false; status: number; code: string; message: string } {
    let configuredBase: string;
    try {
      configuredBase = validateApiBaseUrl(this._getApiBaseUrl() || DEFAULT_API_BASE_URL);
    } catch (err) {
      return {
        ok: false,
        status: 500,
        code: "bad_config",
        message: err instanceof Error ? err.message : String(err),
      };
    }

    if (typeof body.apiBaseUrl === "string" && body.apiBaseUrl.trim()) {
      let requestedBase: string;
      try {
        requestedBase = validateApiBaseUrl(body.apiBaseUrl);
      } catch (err) {
        return {
          ok: false,
          status: 400,
          code: "bad_request",
          message: err instanceof Error ? err.message : String(err),
        };
      }
      if (requestedBase !== configuredBase) {
        return {
          ok: false,
          status: 403,
          code: "untrusted_api_base_url",
          message: "Refusing token request for untrusted API base URL",
        };
      }
    }
    return { ok: true, apiBaseUrl: configuredBase };
  }

  private async _handleRequest(
    req: http.IncomingMessage,
    res: http.ServerResponse,
    expectedSecret: string
  ): Promise<void> {
    const send = (status: number, body: Record<string, unknown>) => {
      res.writeHead(status, {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
      });
      res.end(JSON.stringify(body));
    };

    if (req.method !== "POST" || req.url !== "/token") {
      send(404, { code: "not_found", message: "Unknown broker route" });
      return;
    }
    if (req.headers.origin) {
      send(403, { code: "forbidden", message: "Cross-origin broker requests are forbidden" });
      return;
    }

    const gotSecretHeader = req.headers["x-broker-secret"];
    const gotSecret = Array.isArray(gotSecretHeader) ? gotSecretHeader[0] : gotSecretHeader ?? "";
    if (!gotSecret || !constantTimeEqual(gotSecret, expectedSecret)) {
      send(401, { code: "unauthorized", message: "Invalid broker secret" });
      return;
    }

    const parsedBody = await this._readRequestBody(req);
    if (!parsedBody.ok) {
      send(parsedBody.status, { code: parsedBody.code, message: parsedBody.message });
      return;
    }

    const body = parsedBody.body;
    let tokenReq: GitTokenRequest;
    try {
      tokenReq = this._parseTokenRequest(body);
    } catch (err) {
      send(400, { code: "bad_request", message: err instanceof Error ? err.message : String(err) });
      return;
    }

    const apiCheck = this._parseVerifiedApiBase(body);
    if (!apiCheck.ok) {
      send(apiCheck.status, { code: apiCheck.code, message: apiCheck.message });
      return;
    }
    const apiBaseUrl = apiCheck.apiBaseUrl;

    if (tokenReq.target === "project" && !(await this._isProjectAllowed(tokenReq.projectId))) {
      send(403, {
        code: "project_not_linked",
        message: "Project is not linked in any open workspace folder",
      });
      return;
    }

    const clientToken = await this._getClientToken();
    if (!clientToken?.trim()) {
      send(401, {
        code: "not_signed_in",
        message: "Not signed in to Ginnung Git (run 'Ginnung Git: Set Client Token')",
      });
      return;
    }

    const repoKey =
      tokenReq.target === "service" ? `service:${tokenReq.repo}` : `project:${tokenReq.projectId}`;
    const cacheKey = `${apiBaseUrl}|${repoKey}|${tokenReq.scope}`;
    if (body.forceRefresh !== true) {
      const cached = this._getCachedToken(cacheKey);
      if (cached) {
        send(200, cached as unknown as Record<string, unknown>);
        return;
      }
    }

    try {
      const minted = await mintGitToken(apiBaseUrl, clientToken, tokenReq, this._fetchImpl);
      this._cache.set(cacheKey, minted);
      send(200, minted as unknown as Record<string, unknown>);
    } catch (err) {
      send(502, {
        code: "mint_failed",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }
}

export function parseCredentialInput(stdinText: string): CredentialQuery {
  const query: CredentialQuery = {};
  for (const rawLine of stdinText.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) break;
    const eq = line.indexOf("=");
    if (eq <= 0) continue;
    const key = line.slice(0, eq);
    const value = line.slice(eq + 1);
    if (key === "protocol") query.protocol = value;
    else if (key === "host") query.host = value;
    else if (key === "path") query.path = value;
    else if (key === "username") query.username = value;
  }
  return query;
}

/**
 * Derive a Git token request from an unlinked Cloudflare Artifacts URL path
 * (`/git/<ns>/<userId>.<projectId>.git` or `/git/<ns>/service-(core|pack).git`).
 */
export function parseArtifactsPath(rawPath: string): GitTokenRequest | null {
  const cleaned = trimSlashes(rawPath);
  const parts = cleaned.split("/");
  if (parts.length < 3 || parts[0] !== "git") return null;
  let repoName = parts.at(-1) ?? "";
  if (repoName.endsWith(".git")) {
    repoName = repoName.slice(0, -4);
  }
  if (repoName === "service-core") {
    return { target: "service", repo: "core", scope: "read" };
  }
  if (repoName === "service-pack") {
    return { target: "service", repo: "pack", scope: "read" };
  }
  const dot = repoName.lastIndexOf(".");
  if (dot <= 0) return null;
  const projectId = repoName.slice(dot + 1);
  if (!CANONICAL_PROJECT_RE.test(projectId)) return null;
  return { target: "project", projectId, scope: "write" };
}

function trimSlashes(p: string): string {
  let start = 0;
  let end = p.length;
  while (start < end && p.charCodeAt(start) === 47) start += 1;
  while (end > start && p.charCodeAt(end - 1) === 47) end -= 1;
  return p.slice(start, end);
}

function normalizeUrlPath(p: string): string {
  return trimSlashes(p);
}

function matchesRemoteUrl(query: CredentialQuery, remoteUrlStr: string): boolean {
  let remote: URL;
  try {
    remote = new URL(remoteUrlStr);
  } catch {
    return false;
  }
  const expectedProto = remote.protocol.replace(/:$/, "").toLowerCase();
  const gotProto = (query.protocol || "").toLowerCase();
  if (!gotProto || gotProto !== expectedProto) return false;

  const expectedHost = remote.host.toLowerCase();
  const gotHost = (query.host || "").toLowerCase();
  if (!gotHost || gotHost !== expectedHost) return false;

  if (query.path) {
    if (normalizeUrlPath(query.path) !== normalizeUrlPath(remote.pathname)) {
      return false;
    }
  }
  return true;
}

function isAllowedArtifactsHost(hostWithPort: string): boolean {
  const colonIdx = hostWithPort.lastIndexOf(":");
  const bareHost = (
    colonIdx > 0 && !hostWithPort.endsWith("]") ? hostWithPort.slice(0, colonIdx) : hostWithPort
  ).toLowerCase();
  return bareHost.endsWith(".artifacts.cloudflare.net") || isLoopbackHostname(bareHost);
}

function readBrokerFile(configDir?: string): BrokerFile | null {
  const filePath = getBrokerFilePath(configDir);
  if (!fs.existsSync(filePath)) return null;
  try {
    const data = JSON.parse(fs.readFileSync(filePath, "utf-8")) as Partial<BrokerFile>;
    if (
      typeof data.port === "number" &&
      Number.isInteger(data.port) &&
      data.port > 0 &&
      typeof data.secret === "string" &&
      data.secret.length > 0
    ) {
      return {
        port: data.port,
        secret: data.secret,
        pid: typeof data.pid === "number" ? data.pid : 0,
        apiBaseUrl: typeof data.apiBaseUrl === "string" ? data.apiBaseUrl : DEFAULT_API_BASE_URL,
      };
    }
  } catch {
    // Ignore corrupt broker file
  }
  return null;
}

function readStandaloneClientAuth(
  configDir?: string,
  env: NodeJS.ProcessEnv = process.env
): { token: string; apiBaseUrl?: string } | null {
  const envTok = env.GINNUNG_CLIENT_TOKEN?.trim();
  if (envTok) {
    return {
      token: envTok,
      apiBaseUrl: env.GINNUNG_API_BASE_URL?.trim() || undefined,
    };
  }
  const filePath = getStandaloneTokenFilePath(configDir);
  if (!fs.existsSync(filePath)) return null;
  try {
    const parsed = JSON.parse(fs.readFileSync(filePath, "utf-8")) as Partial<StandaloneTokenFile>;
    if (typeof parsed.token === "string" && parsed.token.trim()) {
      return {
        token: parsed.token.trim(),
        apiBaseUrl: typeof parsed.apiBaseUrl === "string" ? parsed.apiBaseUrl : undefined,
      };
    }
  } catch {
    // Ignore corrupt token file
  }
  return null;
}

async function resolveTargetRequest(
  query: CredentialQuery,
  cwd: string
): Promise<{ req: GitTokenRequest; apiBaseUrl: string; linked: boolean }> {
  const link = await readGinnungRemoteLink(cwd);
  if (link) {
    if (!matchesRemoteUrl(query, link.remoteUrl)) {
      throw new Error(
        `Refusing Ginnung credentials for unmatched remote (${query.protocol || ""}://${query.host || ""}/${query.path || ""})`
      );
    }
    return {
      req: { target: "project", projectId: link.projectId, scope: link.scope },
      apiBaseUrl: link.apiBaseUrl,
      linked: true,
    };
  }

  if (!query.host || !isAllowedArtifactsHost(query.host)) {
    throw new Error(`Refusing Ginnung credentials for non-Artifacts host '${query.host || ""}'`);
  }
  const parsed = query.path ? parseArtifactsPath(query.path) : null;
  if (!parsed) {
    throw new Error(`Cannot resolve Ginnung repository from credential path '${query.path || ""}'`);
  }
  return { req: parsed, apiBaseUrl: DEFAULT_API_BASE_URL, linked: false };
}

async function readBrokerErrorMessage(res: Response): Promise<string> {
  try {
    const errJson = (await res.json()) as { message?: string };
    if (errJson.message) return errJson.message;
  } catch {
    // Ignore non-JSON error
  }
  return `Broker HTTP ${res.status}`;
}

async function tryBrokerToken(
  broker: BrokerFile,
  req: GitTokenRequest,
  apiBaseUrl: string,
  hasStandaloneFallback: boolean,
  fetchImpl: typeof fetch
): Promise<{ username: string; password: string } | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 2000);
  try {
    const res = await fetchImpl(`http://127.0.0.1:${broker.port}/token`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Broker-Secret": broker.secret,
      },
      body: JSON.stringify({ ...req, apiBaseUrl }),
      signal: controller.signal,
    });
    if (res.ok) {
      const data = (await res.json()) as Partial<GitTokenResponse>;
      if (typeof data.token === "string" && data.token) {
        return { username: "x", password: stripTokenQuery(data.token) };
      }
      return null;
    }
    if (!hasStandaloneFallback) {
      throw new Error(await readBrokerErrorMessage(res));
    }
    return null;
  } catch (err) {
    if (!hasStandaloneFallback && err instanceof Error && !/fetch failed|ECONNREFUSED|abort/i.test(err.message)) {
      throw err;
    }
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Resolve short-lived Basic credentials (`username = x`, `password = <secret>`)
 * for Git `credential get`, checking the repo's remote identity strictly first.
 */
export async function resolveCredentialForRequest(opts: {
  query: CredentialQuery;
  cwd: string;
  configDir?: string;
  env?: NodeJS.ProcessEnv;
  fetchImpl?: typeof fetch;
}): Promise<{ username: string; password: string }> {
  const fetchImpl = opts.fetchImpl ?? fetch;
  const env = opts.env ?? process.env;
  const { req, apiBaseUrl, linked } = await resolveTargetRequest(opts.query, opts.cwd);
  const standalone = readStandaloneClientAuth(opts.configDir, env);

  // Tier 1: Query running VS Code / Antigravity loopback broker if available.
  const broker = readBrokerFile(opts.configDir);
  if (broker) {
    const brokerCreds = await tryBrokerToken(broker, req, apiBaseUrl, Boolean(standalone), fetchImpl);
    if (brokerCreds) return brokerCreds;
  }

  // Tier 2: Standalone token fallback (terminal agents when IDE is closed).
  if (standalone) {
    const effectiveBase = linked ? apiBaseUrl : standalone.apiBaseUrl || apiBaseUrl;
    if (!isTrustedStandaloneApiBaseUrl(effectiveBase, env, standalone.apiBaseUrl)) {
      throw new Error(
        `Refusing to send Ginnung client token to untrusted API base URL '${effectiveBase}'`
      );
    }
    const minted = await mintGitToken(effectiveBase, standalone.token, req, fetchImpl);
    return { username: "x", password: stripTokenQuery(minted.token) };
  }

  throw new Error(
    "Ginnung Git auth: open VS Code / Antigravity with the Lite-STEP extension signed in, or set GINNUNG_CLIENT_TOKEN."
  );
}
