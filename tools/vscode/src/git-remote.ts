/**
 * Cloudflare Artifacts named Git remote ("ginnung") management & token minting.
 *
 * Pure Node module (zero `vscode` imports) shared by the VS Code extension and
 * the standalone `git-credential-ginnung` helper CLI.
 */

import { execFile } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export const DEFAULT_REMOTE_NAME = "ginnung";
export const DEFAULT_API_BASE_URL = "https://api.ginnung.tech";
export const CANONICAL_PROJECT_RE =
  /^(?:default|P(?:000[1-9]|00[1-9]\d|0[1-9]\d{2}|[1-9]\d{3,5}))$/;
export const REMOTE_NAME_RE = /^[A-Za-z\d][A-Za-z\d._-]{0,63}$/;

const CONTROL_CHARS_RE = /[\r\n\0]/;
const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);

export type GitTokenRequest =
  | { target: "project"; projectId: string; scope: "read" | "write" }
  | { target: "service"; repo: "core" | "pack"; scope: "read" };

export interface GitTokenResponse {
  remote: string;
  repoName: string;
  scope: "read" | "write";
  token: string;
  expiresAt: string;
}

export interface GinnungRemoteLink {
  repoRoot: string;
  remoteName: string;
  remoteUrl: string;
  projectId: string;
  apiBaseUrl: string;
  scope: "read" | "write";
}

export function isLoopbackHostname(hostname: string): boolean {
  return LOOPBACK_HOSTS.has(hostname.toLowerCase());
}

/**
 * Strip Cloudflare Artifacts `?expires=...` suffix for Git HTTP Basic auth
 * (`username = x`, `password = <secret>`) and guard against newline/null injection.
 */
export function stripTokenQuery(token: string): string {
  if (typeof token !== "string" || CONTROL_CHARS_RE.test(token)) {
    throw new Error("Invalid Git token: unexpected control characters");
  }
  const secret = token.trim().split("?")[0]?.trim() ?? "";
  if (!secret) {
    throw new Error("Invalid Git token: empty secret");
  }
  return secret;
}

function trimTrailingSlashes(s: string): string {
  let end = s.length;
  while (end > 0 && s.charCodeAt(end - 1) === 47) {
    end -= 1;
  }
  return s.slice(0, end);
}

export function validateApiBaseUrl(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed || CONTROL_CHARS_RE.test(trimmed)) {
    throw new Error("Invalid API base URL");
  }
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    throw new Error(`Invalid API base URL: ${trimmed}`);
  }
  if (parsed.username || parsed.password) {
    throw new Error("API base URL must not contain embedded credentials");
  }
  const proto = parsed.protocol.toLowerCase();
  if (proto !== "https:" && !(proto === "http:" && isLoopbackHostname(parsed.hostname))) {
    throw new Error(
      `API base URL must use https:// (or http:// on loopback only), got ${parsed.origin}`
    );
  }
  const cleanPath = trimTrailingSlashes(parsed.pathname);
  return cleanPath && cleanPath !== "/" ? `${parsed.origin}${cleanPath}` : parsed.origin;
}

export function validateRemoteUrl(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed || CONTROL_CHARS_RE.test(trimmed)) {
    throw new Error("Invalid Git remote URL");
  }
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    throw new Error(`Invalid Git remote URL: ${trimmed}`);
  }
  if (parsed.username || parsed.password) {
    throw new Error("Git remote URL must not embed credentials");
  }
  const proto = parsed.protocol.toLowerCase();
  if (proto !== "https:" && !(proto === "http:" && isLoopbackHostname(parsed.hostname))) {
    throw new Error(
      `Git remote URL must use https:// (or http:// on loopback only), got ${parsed.protocol}`
    );
  }
  return parsed.toString();
}

export function validateProjectId(raw: string): string {
  const trimmed = raw.trim();
  if (!CANONICAL_PROJECT_RE.test(trimmed)) {
    throw new Error(
      `Invalid Ginnung project ID '${trimmed}' (expected 'default' or P0001..P999999)`
    );
  }
  return trimmed;
}

export function validateRemoteName(raw: string): string {
  const trimmed = raw.trim();
  if (!REMOTE_NAME_RE.test(trimmed)) {
    throw new Error(`Invalid Git remote name '${trimmed}'`);
  }
  return trimmed;
}

function shQuote(value: string): string {
  return "'" + value.replaceAll("'", String.raw`'\''`) + "'";
}

/**
 * Format a Git `!` credential helper command that works cleanly under POSIX
 * `/bin/sh` on Windows (Git for Windows MSYS2), macOS, and Linux.
 */
export function formatCredentialHelperCommand(
  nodeCommand: string,
  helperScriptPath: string,
  extraEnv?: Record<string, string>
): string {
  const trimmedNode = nodeCommand.trim();
  const trimmedScript = helperScriptPath.trim();
  if (!trimmedNode || CONTROL_CHARS_RE.test(trimmedNode)) {
    throw new Error("Invalid Node executable path for Git credential helper");
  }
  if (!trimmedScript || CONTROL_CHARS_RE.test(trimmedScript)) {
    throw new Error("Invalid script path for Git credential helper");
  }

  const normalizedNode = trimmedNode.replaceAll("\\", "/");
  const normalizedScript = trimmedScript.replaceAll("\\", "/");

  const envParts: string[] = [];
  if (extraEnv) {
    for (const [k, v] of Object.entries(extraEnv)) {
      if (!/^[A-Z_][A-Z0-9_]*$/.test(k)) {
        throw new Error(`Invalid environment variable name: ${k}`);
      }
      if (CONTROL_CHARS_RE.test(v)) {
        throw new Error(`Invalid control character in environment variable ${k}`);
      }
      envParts.push(`${k}=${shQuote(v)}`);
    }
  }

  const cmdParts = [...envParts, shQuote(normalizedNode), shQuote(normalizedScript)];
  return `!${cmdParts.join(" ")}`;
}

function formatMintError(status: number, code: string | undefined, msg: string): Error {
  const suffix = msg ? `: ${msg}` : "";
  if (status === 401) {
    return new Error(`Ginnung Git authorization failed (401): client token expired or revoked${suffix}`);
  }
  if (status === 403) {
    if (code === "pack_entitlement_required") {
      return new Error(
        "Ginnung Git authorization denied (403): your plan does not include the proprietary service pack"
      );
    }
    return new Error(`Ginnung Git authorization forbidden (403)${suffix}`);
  }
  if (status === 404) {
    return new Error(`Ginnung project not found (404)${suffix}`);
  }
  if (status === 503) {
    return new Error(`Ginnung Git token service temporarily unavailable (503)${suffix}`);
  }
  return new Error(`POST /git/tokens failed (HTTP ${status})${suffix}`);
}

function parseJsonObjectSafe(rawText: string): Record<string, unknown> {
  if (!rawText.trim()) return {};
  try {
    return JSON.parse(rawText) as Record<string, unknown>;
  } catch {
    return {};
  }
}

/**
 * Mint a short-lived single-repo Cloudflare Artifacts token from Ginnung's
 * edge broker (`POST /git/tokens`).
 */
export async function mintGitToken(
  apiBaseUrl: string,
  clientToken: string,
  req: GitTokenRequest,
  fetchImpl: typeof fetch = fetch
): Promise<GitTokenResponse> {
  const base = validateApiBaseUrl(apiBaseUrl);
  const bearer = clientToken.trim();
  if (!bearer || CONTROL_CHARS_RE.test(bearer)) {
    throw new Error("Missing or malformed Ginnung client token");
  }

  const res = await fetchImpl(`${base}/git/tokens`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${bearer}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(req),
  });

  const text = await res.text();
  const parsed = parseJsonObjectSafe(text);
  if (!res.ok) {
    const code = typeof parsed.code === "string" ? parsed.code : undefined;
    const msg = typeof parsed.message === "string" ? parsed.message : text.trim();
    throw formatMintError(res.status, code, msg);
  }

  const remote = typeof parsed.remote === "string" ? validateRemoteUrl(parsed.remote) : "";
  const token = typeof parsed.token === "string" ? parsed.token.trim() : "";
  const repoName = typeof parsed.repoName === "string" ? parsed.repoName : "";
  const scope = parsed.scope === "read" || parsed.scope === "write" ? parsed.scope : req.scope;
  const expiresAt = typeof parsed.expiresAt === "string" ? parsed.expiresAt : "";

  if (!remote || !token) {
    throw new Error("POST /git/tokens returned an incomplete response (missing remote or token)");
  }
  stripTokenQuery(token);

  return { remote, repoName, scope, token, expiresAt };
}

export async function runGit(
  args: string[],
  cwd: string,
  extraEnv?: Record<string, string>
): Promise<{ stdout: string; stderr: string }> {
  const { stdout, stderr } = await execFileAsync("git", args, {
    cwd,
    env: extraEnv ? { ...process.env, ...extraEnv } : process.env,
    windowsHide: true,
  });
  return { stdout, stderr };
}

async function gitConfigGetLocal(repoRoot: string, key: string): Promise<string | null> {
  try {
    const { stdout } = await runGit(["config", "--local", "--get", key], repoRoot);
    const val = stdout.trim();
    return val || null;
  } catch {
    return null;
  }
}

export async function findGitRepoRoot(dir: string): Promise<string | null> {
  if (!dir || !fs.existsSync(dir)) return null;
  try {
    const { stdout } = await runGit(["rev-parse", "--show-toplevel"], dir);
    const root = stdout.trim();
    return root ? path.resolve(root) : null;
  } catch {
    return null;
  }
}

export async function ensureGitRepo(dir: string): Promise<string> {
  const existing = await findGitRepoRoot(dir);
  if (existing) return existing;
  fs.mkdirSync(dir, { recursive: true });
  await runGit(["init", "-b", "main"], dir);
  return path.resolve(dir);
}

/**
 * Configure a named remote (default `"ginnung"`) and a repo-local credential
 * helper scoped strictly to `remoteUrl` without touching `origin` or global Git config.
 */
export async function configureGinnungRemote(opts: {
  repoRoot: string;
  remoteName?: string;
  remoteUrl: string;
  projectId: string;
  apiBaseUrl: string;
  helperCommand: string;
  scope?: "read" | "write";
}): Promise<GinnungRemoteLink> {
  const repoRoot = path.resolve(opts.repoRoot);
  const remoteName = validateRemoteName(opts.remoteName || DEFAULT_REMOTE_NAME);
  const remoteUrl = validateRemoteUrl(opts.remoteUrl);
  const projectId = validateProjectId(opts.projectId);
  const apiBaseUrl = validateApiBaseUrl(opts.apiBaseUrl);
  const scope: "read" | "write" = opts.scope === "read" ? "read" : "write";
  if (!opts.helperCommand || CONTROL_CHARS_RE.test(opts.helperCommand)) {
    throw new Error("Invalid credential helper command");
  }

  let hasRemote = false;
  try {
    await runGit(["remote", "get-url", remoteName], repoRoot);
    hasRemote = true;
  } catch {
    hasRemote = false;
  }

  if (hasRemote) {
    await runGit(["remote", "set-url", remoteName, remoteUrl], repoRoot);
  } else {
    await runGit(["remote", "add", remoteName, remoteUrl], repoRoot);
  }

  await runGit(["config", "--local", "ginnung.remote", remoteName], repoRoot);
  await runGit(["config", "--local", "ginnung.projectId", projectId], repoRoot);
  await runGit(["config", "--local", "ginnung.apiBaseUrl", apiBaseUrl], repoRoot);
  await runGit(["config", "--local", "ginnung.remoteUrl", remoteUrl], repoRoot);
  await runGit(["config", "--local", "ginnung.scope", scope], repoRoot);

  // Scoped strictly to this remote URL in .git/config (never global); empty string
  // resets any inherited OS credential helper so short-lived tokens aren't cached.
  await runGit(["config", "--local", `credential.${remoteUrl}.useHttpPath`, "true"], repoRoot);
  await runGit(["config", "--local", "--replace-all", `credential.${remoteUrl}.helper`, ""], repoRoot);
  await runGit(["config", "--local", "--add", `credential.${remoteUrl}.helper`, opts.helperCommand], repoRoot);

  return {
    repoRoot,
    remoteName,
    remoteUrl,
    projectId,
    apiBaseUrl,
    scope,
  };
}

export async function readGinnungRemoteLink(repoRoot: string): Promise<GinnungRemoteLink | null> {
  const resolved = await findGitRepoRoot(repoRoot);
  if (!resolved) return null;

  const projectId = await gitConfigGetLocal(resolved, "ginnung.projectId");
  if (!projectId) return null;

  const remoteName = (await gitConfigGetLocal(resolved, "ginnung.remote")) || DEFAULT_REMOTE_NAME;
  const apiBaseUrl = (await gitConfigGetLocal(resolved, "ginnung.apiBaseUrl")) || DEFAULT_API_BASE_URL;
  const rawScope = await gitConfigGetLocal(resolved, "ginnung.scope");
  const scope: "read" | "write" = rawScope === "read" ? "read" : "write";

  const configuredUrl =
    (await gitConfigGetLocal(resolved, `remote.${remoteName}.url`)) ||
    (await gitConfigGetLocal(resolved, "ginnung.remoteUrl"));
  if (!configuredUrl) return null;

  return {
    repoRoot: resolved,
    remoteName,
    remoteUrl: configuredUrl,
    projectId,
    apiBaseUrl,
    scope,
  };
}

async function unsetLocalConfig(repoRoot: string, key: string): Promise<void> {
  try {
    await runGit(["config", "--local", "--unset-all", key], repoRoot);
  } catch {
    // Ignore if not set
  }
}

export async function unlinkGinnungRemote(
  repoRoot: string,
  opts: { removeRemote?: boolean } = {}
): Promise<void> {
  const link = await readGinnungRemoteLink(repoRoot);
  const resolved = link?.repoRoot ?? (await findGitRepoRoot(repoRoot));
  if (!resolved) return;

  if (link?.remoteUrl) {
    await Promise.all([
      unsetLocalConfig(resolved, `credential.${link.remoteUrl}.helper`),
      unsetLocalConfig(resolved, `credential.${link.remoteUrl}.useHttpPath`),
    ]);
  }

  try {
    await runGit(["config", "--local", "--remove-section", "ginnung"], resolved);
  } catch {
    // Ignore if already absent
  }

  if (opts.removeRemote && link?.remoteName) {
    try {
      await runGit(["remote", "remove", link.remoteName], resolved);
    } catch {
      // Ignore if already removed
    }
  }
}

export async function cloneGinnungProject(opts: {
  targetDir: string;
  remoteUrl: string;
  token: string;
  remoteName?: string;
  projectId: string;
  apiBaseUrl: string;
  helperCommand: string;
  scope?: "read" | "write";
}): Promise<GinnungRemoteLink> {
  const targetDir = path.resolve(opts.targetDir);
  const remoteName = validateRemoteName(opts.remoteName || DEFAULT_REMOTE_NAME);
  const remoteUrl = validateRemoteUrl(opts.remoteUrl);
  if (!opts.token || CONTROL_CHARS_RE.test(opts.token)) {
    throw new Error("Missing or invalid clone token");
  }

  const parentDir = path.dirname(targetDir);
  fs.mkdirSync(parentDir, { recursive: true });

  // Never put the short-lived token in the remote URL; pass via -c extraHeader.
  await runGit(
    [
      "-c",
      `http.extraHeader=Authorization: Bearer ${opts.token.trim()}`,
      "clone",
      "-o",
      remoteName,
      remoteUrl,
      targetDir,
    ],
    parentDir
  );

  return configureGinnungRemote({
    repoRoot: targetDir,
    remoteName,
    remoteUrl,
    projectId: opts.projectId,
    apiBaseUrl: opts.apiBaseUrl,
    helperCommand: opts.helperCommand,
    scope: opts.scope,
  });
}

export async function currentBranch(repoRoot: string): Promise<string | null> {
  try {
    const { stdout } = await runGit(["symbolic-ref", "--quiet", "--short", "HEAD"], repoRoot);
    const branch = stdout.trim();
    return branch || null;
  } catch {
    return null;
  }
}

export async function gitPullFromRemote(
  repoRoot: string,
  remoteName: string,
  branch?: string
): Promise<{ stdout: string; stderr: string }> {
  const validRemote = validateRemoteName(remoteName);
  const targetBranch = branch || (await currentBranch(repoRoot)) || "main";
  return runGit(["pull", validRemote, targetBranch], repoRoot, { GIT_TERMINAL_PROMPT: "0" });
}

export async function gitPushToRemote(
  repoRoot: string,
  remoteName: string,
  branch?: string
): Promise<{ stdout: string; stderr: string }> {
  const validRemote = validateRemoteName(remoteName);
  const targetBranch = branch || (await currentBranch(repoRoot)) || "main";
  return runGit(["push", validRemote, targetBranch], repoRoot, { GIT_TERMINAL_PROMPT: "0" });
}
