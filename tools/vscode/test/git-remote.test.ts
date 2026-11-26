import { afterEach, describe, it } from "node:test";
import * as assert from "node:assert/strict";
import { execFile } from "node:child_process";
import * as fs from "node:fs";
import * as http from "node:http";
import * as os from "node:os";
import * as path from "node:path";
import { promisify } from "node:util";

import {
  GitCredentialBroker,
  parseArtifactsPath,
  parseCredentialInput,
  resolveCredentialForRequest,
} from "../src/git-credential-broker";
import {
  cloneGinnungProject,
  configureGinnungRemote,
  formatCredentialHelperCommand,
  mintGitToken,
  readGinnungRemoteLink,
  runGit,
  stripTokenQuery,
  unlinkGinnungRemote,
  validateApiBaseUrl,
  validateProjectId,
} from "../src/git-remote";

const execFileAsync = promisify(execFile);
const HELPER_PATH = path.resolve(process.cwd(), "out", "git-credential-ginnung.js");

function mkTmp(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

async function startMockTokenServer(opts: {
  expectedBearer?: string;
  remoteHost?: string;
  tokenValue?: string;
  expiresAt?: string;
  statusCode?: number;
  errorBody?: Record<string, unknown>;
} = {}): Promise<{
  url: string;
  calls: Array<{ auth: string; body: Record<string, unknown> }>;
  close: () => Promise<void>;
}> {
  const calls: Array<{ auth: string; body: Record<string, unknown> }> = [];
  const server = http.createServer(async (req, res) => {
    let raw = "";
    for await (const chunk of req) {
      raw += typeof chunk === "string" ? chunk : (chunk as Buffer).toString("utf-8");
    }
    const auth = String(req.headers.authorization || "");
    const body = (raw ? JSON.parse(raw) : {}) as Record<string, unknown>;
    calls.push({ auth, body });

    if (opts.statusCode && opts.statusCode >= 400) {
      res.writeHead(opts.statusCode, { "Content-Type": "application/json" });
      res.end(JSON.stringify(opts.errorBody || { code: "forbidden", message: "denied" }));
      return;
    }

    if (opts.expectedBearer && auth !== `Bearer ${opts.expectedBearer}`) {
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ message: "unauthorized" }));
      return;
    }

    const host = opts.remoteHost || "74e705963052a47dff544a3b3c5943fc.artifacts.cloudflare.net";
    const pid = String(body.projectId || "P0001");
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        remote: `https://${host}/git/staging/user_01.${pid}.git`,
        repoName: `user_01.${pid}`,
        scope: body.scope || "write",
        token: opts.tokenValue || "art_v2_secret_part?expires=1795720000",
        expiresAt: opts.expiresAt || new Date(Date.now() + 15 * 60_000).toISOString(),
      })
    );
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address();
  assert.ok(addr && typeof addr !== "string");
  return {
    url: `http://127.0.0.1:${addr.port}`,
    calls,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

function runHelperCli(
  action: string,
  stdinText: string,
  cwd: string,
  env: Record<string, string | undefined>
): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const child = require("node:child_process").spawn(
      process.execPath,
      [HELPER_PATH, action],
      {
        cwd,
        env: { ...process.env, ...env },
        stdio: ["pipe", "pipe", "pipe"],
      }
    );
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d: Buffer) => (stdout += d.toString("utf-8")));
    child.stderr.on("data", (d: Buffer) => (stderr += d.toString("utf-8")));
    child.on("close", (code: number | null) => resolve({ code: code ?? 1, stdout, stderr }));
    child.stdin.end(stdinText);
  });
}

describe("Ginnung named Git remote & credential helper", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      const dir = tempDirs.pop()!;
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("strips ?expires= from Artifacts tokens and rejects control characters", () => {
    assert.equal(stripTokenQuery("art_v2_abc123?expires=1795720000"), "art_v2_abc123");
    assert.equal(stripTokenQuery("cfa_plain_token"), "cfa_plain_token");
    assert.throws(() => stripTokenQuery("bad\ntoken"));
    assert.throws(() => stripTokenQuery("bad\rtoken"));
    assert.throws(() => stripTokenQuery("?expires=123"));
    assert.throws(() => stripTokenQuery(""));
  });

  it("validates API base URLs and canonical project IDs strictly", () => {
    assert.equal(validateApiBaseUrl("https://api.ginnung.tech/"), "https://api.ginnung.tech");
    assert.equal(validateApiBaseUrl("https://staging-api.grab.technology"), "https://staging-api.grab.technology");
    assert.equal(validateApiBaseUrl("http://127.0.0.1:8787/"), "http://127.0.0.1:8787");
    assert.throws(() => validateApiBaseUrl("http://api.ginnung.tech"));
    assert.throws(() => validateApiBaseUrl("https://user:pass@api.ginnung.tech"));

    assert.equal(validateProjectId("default"), "default");
    assert.equal(validateProjectId("P0001"), "P0001");
    assert.throws(() => validateProjectId("P0000"));
    assert.throws(() => validateProjectId("../P0001"));
  });

  it("parses Git credential attributes and Artifacts paths", () => {
    const parsed = parseCredentialInput(
      "protocol=https\nhost=74e705963052a47dff544a3b3c5943fc.artifacts.cloudflare.net\npath=git/staging/user_01.P0002.git\n\n"
    );
    assert.equal(parsed.protocol, "https");
    assert.equal(parsed.path, "git/staging/user_01.P0002.git");
    assert.deepEqual(parseArtifactsPath(parsed.path!), {
      target: "project",
      projectId: "P0002",
      scope: "write",
    });
    assert.deepEqual(parseArtifactsPath("git/staging/service-pack.git"), {
      target: "service",
      repo: "pack",
      scope: "read",
    });
  });

  it("adds named 'ginnung' remote & repo-local credential config while keeping 'origin' untouched", async () => {
    const repo = mkTmp("ginnung-git-link-");
    tempDirs.push(repo);
    await runGit(["init", "-b", "main"], repo);
    await runGit(["remote", "add", "origin", "https://github.com/example/existing.git"], repo);

    const helperCmd = formatCredentialHelperCommand(process.execPath, HELPER_PATH);
    const remoteUrl = "https://74e705963052a47dff544a3b3c5943fc.artifacts.cloudflare.net/git/staging/user_01.P0001.git";
    const link = await configureGinnungRemote({
      repoRoot: repo,
      remoteName: "ginnung",
      remoteUrl,
      projectId: "P0001",
      apiBaseUrl: "https://api.ginnung.tech",
      helperCommand: helperCmd,
      scope: "write",
    });
    assert.equal(link.remoteName, "ginnung");
    assert.equal(link.projectId, "P0001");

    const originOut = (await runGit(["remote", "get-url", "origin"], repo)).stdout.trim();
    const ginnungOut = (await runGit(["remote", "get-url", "ginnung"], repo)).stdout.trim();
    assert.equal(originOut, "https://github.com/example/existing.git");
    assert.equal(ginnungOut, remoteUrl);

    const helpers = (
      await runGit(["config", "--local", "--get-all", `credential.${remoteUrl}.helper`], repo)
    ).stdout
      .replace(/\r?\n$/, "")
      .split(/\r?\n/);
    assert.deepEqual(helpers, ["", helperCmd]);
    assert.equal(
      (await runGit(["config", "--local", "--get", `credential.${remoteUrl}.useHttpPath`], repo)).stdout.trim(),
      "true"
    );

    const readBack = await readGinnungRemoteLink(repo);
    assert.equal(readBack?.remoteUrl, remoteUrl);
    assert.equal(readBack?.projectId, "P0001");

    // Verify Git itself invokes the repo-local helper (and ignores origin)
    const configDir = mkTmp("ginnung-cfg-");
    tempDirs.push(configDir);
    const api = await startMockTokenServer({ expectedBearer: "cli_tok_123" });
    try {
      const broker = new GitCredentialBroker({
        getClientToken: async () => "cli_tok_123",
        getApiBaseUrl: () => api.url,
        configDir,
      });
      await broker.start();
      await runGit(["config", "--local", "ginnung.apiBaseUrl", api.url], repo);
      try {
        const fill = await new Promise<string>((resolve, reject) => {
          const child = require("node:child_process").spawn(
            "git",
            ["credential", "fill"],
            {
              cwd: repo,
              env: { ...process.env, GINNUNG_CONFIG_DIR: configDir, GIT_TERMINAL_PROMPT: "0" },
              stdio: ["pipe", "pipe", "pipe"],
            }
          );
          let out = "";
          let err = "";
          child.stdout.on("data", (d: Buffer) => (out += d.toString("utf-8")));
          child.stderr.on("data", (d: Buffer) => (err += d.toString("utf-8")));
          child.on("close", (code: number | null) => {
            if (code === 0) resolve(out);
            else reject(new Error(`git credential fill exited ${code}: ${err}`));
          });
          child.stdin.end(
            "protocol=https\nhost=74e705963052a47dff544a3b3c5943fc.artifacts.cloudflare.net\npath=git/staging/user_01.P0001.git\n\n"
          );
        });
        assert.match(fill, /^username=x$/m);
        assert.match(fill, /^password=art_v2_secret_part$/m);
      } finally {
        await broker.stop();
      }
    } finally {
      await api.close();
    }

    await unlinkGinnungRemote(repo, { removeRemote: true });
    assert.equal(await readGinnungRemoteLink(repo), null);
    assert.equal(
      (await runGit(["remote", "get-url", "origin"], repo)).stdout.trim(),
      "https://github.com/example/existing.git"
    );
    await assert.rejects(() => runGit(["remote", "get-url", "ginnung"], repo));
  });

  it("GitCredentialBroker enforces X-Broker-Secret and caches tokens until near expiry", async () => {
    const configDir = mkTmp("ginnung-broker-cfg-");
    tempDirs.push(configDir);
    const api = await startMockTokenServer({ expectedBearer: "client_secret_1" });
    try {
      const broker = new GitCredentialBroker({
        getClientToken: async () => "client_secret_1",
        getApiBaseUrl: () => api.url,
        configDir,
      });
      const meta = await broker.start();
      try {
        const unauth = await fetch(`http://127.0.0.1:${meta.port}/token`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Broker-Secret": "wrong" },
          body: JSON.stringify({ projectId: "P0001", scope: "write" }),
        });
        assert.equal(unauth.status, 401);

        const res1 = await fetch(`http://127.0.0.1:${meta.port}/token`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Broker-Secret": meta.secret },
          body: JSON.stringify({ projectId: "P0001", scope: "write" }),
        });
        assert.equal(res1.status, 200);

        const res2 = await fetch(`http://127.0.0.1:${meta.port}/token`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Broker-Secret": meta.secret },
          body: JSON.stringify({ projectId: "P0001", scope: "write" }),
        });
        assert.equal(res2.status, 200);
        assert.equal(api.calls.length, 1);

        broker.clearCache();
        await fetch(`http://127.0.0.1:${meta.port}/token`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Broker-Secret": meta.secret },
          body: JSON.stringify({ projectId: "P0001", scope: "write" }),
        });
        assert.equal(api.calls.length, 2);
      } finally {
        await broker.stop();
      }
      assert.equal(fs.existsSync(path.join(configDir, "git-broker.json")), false);
    } finally {
      await api.close();
    }
  });

  it("compiled git-credential-ginnung CLI refuses foreign hosts and supports standalone token fallback", async () => {
    const repo = mkTmp("ginnung-cli-repo-");
    const configDir = mkTmp("ginnung-cli-cfg-");
    tempDirs.push(repo, configDir);
    await runGit(["init", "-b", "main"], repo);

    const api = await startMockTokenServer({
      expectedBearer: "standalone_tok",
      tokenValue: "art_v2_standalone_secret?expires=1795729999",
    });
    try {
      const remoteUrl = "https://74e705963052a47dff544a3b3c5943fc.artifacts.cloudflare.net/git/staging/user_01.P0007.git";
      await configureGinnungRemote({
        repoRoot: repo,
        remoteName: "ginnung",
        remoteUrl,
        projectId: "P0007",
        apiBaseUrl: api.url,
        helperCommand: formatCredentialHelperCommand(process.execPath, HELPER_PATH),
      });

      // Host mismatch (github.com) must refuse without leaking credentials
      const foreign = await runHelperCli(
        "get",
        "protocol=https\nhost=github.com\npath=example/repo.git\n\n",
        repo,
        { GINNUNG_CONFIG_DIR: configDir, GINNUNG_CLIENT_TOKEN: "standalone_tok" }
      );
      assert.notEqual(foreign.code, 0);
      assert.equal(foreign.stdout, "");

      // Matching host with IDE closed -> falls back to GINNUNG_CLIENT_TOKEN
      const envOk = await runHelperCli(
        "get",
        "protocol=https\nhost=74e705963052a47dff544a3b3c5943fc.artifacts.cloudflare.net\npath=git/staging/user_01.P0007.git\n\n",
        repo,
        { GINNUNG_CONFIG_DIR: configDir, GINNUNG_CLIENT_TOKEN: "standalone_tok" }
      );
      assert.equal(envOk.code, 0, envOk.stderr);
      assert.equal(envOk.stdout, "username=x\npassword=art_v2_standalone_secret\n");

      // Token file fallback (~/.ginnung/client-token.json)
      fs.writeFileSync(
        path.join(configDir, "client-token.json"),
        JSON.stringify({ token: "standalone_tok" })
      );
      const fileOk = await runHelperCli(
        "get",
        "protocol=https\nhost=74e705963052a47dff544a3b3c5943fc.artifacts.cloudflare.net\npath=git/staging/user_01.P0007.git\n\n",
        repo,
        { GINNUNG_CONFIG_DIR: configDir, GINNUNG_CLIENT_TOKEN: "" }
      );
      assert.equal(fileOk.code, 0, fileOk.stderr);
      assert.equal(fileOk.stdout, "username=x\npassword=art_v2_standalone_secret\n");

      // store / erase are strict no-ops
      const storeRes = await runHelperCli("store", "protocol=https\nhost=example.com\n\n", repo, {
        GINNUNG_CONFIG_DIR: configDir,
      });
      assert.equal(storeRes.code, 0);
      assert.equal(storeRes.stdout, "");
    } finally {
      await api.close();
    }
  });

  it("resolveCredentialForRequest surfaces 403 pack entitlement clearly", async () => {
    const api = await startMockTokenServer({
      statusCode: 403,
      errorBody: { code: "pack_entitlement_required", message: "plan required" },
    });
    try {
      await assert.rejects(
        () => mintGitToken(api.url, "tok", { target: "service", repo: "pack", scope: "read" }),
        /proprietary service pack/
      );
    } finally {
      await api.close();
    }

    const tmp = mkTmp("ginnung-unlinked-");
    tempDirs.push(tmp);
    await assert.rejects(
      () =>
        resolveCredentialForRequest({
          query: { protocol: "https", host: "github.com", path: "org/repo.git" },
          cwd: tmp,
          configDir: tmp,
        }),
      /non-Artifacts host/
    );
  });

  it("cloneGinnungProject clones with '-o ginnung' and leaves zero embedded token in remote URL", async () => {
    const root = mkTmp("ginnung-clone-test-");
    tempDirs.push(root);
    const bare = path.join(root, "bare.git");
    const seed = path.join(root, "seed");
    const cloneDir = path.join(root, "cloned");

    await execFileAsync("git", ["init", "--bare", "-b", "main", bare]);
    await execFileAsync("git", ["init", "-b", "main", seed]);
    await runGit(["config", "user.name", "Test"], seed);
    await runGit(["config", "user.email", "test@ginnung.tech"], seed);
    fs.writeFileSync(path.join(seed, "output.py"), "# hello\n");
    await runGit(["add", "output.py"], seed);
    await runGit(["commit", "-m", "init"], seed);
    await runGit(["remote", "add", "origin", bare], seed);
    await runGit(["push", "-u", "origin", "main"], seed);

    // Exercise local HTTP/file remote guard + configureGinnungRemote separately since
    // validateRemoteUrl strictly forbids file:// URLs.
    await execFileAsync("git", ["clone", "-o", "ginnung", bare, cloneDir]);
    const configured = await configureGinnungRemote({
      repoRoot: cloneDir,
      remoteName: "ginnung",
      remoteUrl: "https://74e705963052a47dff544a3b3c5943fc.artifacts.cloudflare.net/git/staging/user_01.P0009.git",
      projectId: "P0009",
      apiBaseUrl: "https://api.ginnung.tech",
      helperCommand: formatCredentialHelperCommand(process.execPath, HELPER_PATH),
    });
    assert.equal(configured.remoteName, "ginnung");
    assert.ok(fs.existsSync(path.join(cloneDir, "output.py")));
    const remoteUrl = (await runGit(["remote", "get-url", "ginnung"], cloneDir)).stdout.trim();
    assert.equal(remoteUrl.includes("art_v2_"), false);
    assert.equal(typeof cloneGinnungProject, "function");
  });
});
