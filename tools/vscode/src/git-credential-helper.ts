/**
 * Standalone `git-credential` helper CLI bundled to `out/git-credential-ginnung.js`.
 *
 * Configured repo-locally via `git config --local credential.<remoteUrl>.helper`.
 * Reads `get` attributes from stdin, requests a 15-minute Cloudflare Artifacts
 * token from the running extension broker (or standalone GINNUNG_CLIENT_TOKEN),
 * and prints `username=x` / `password=<secret>` without touching global Git state.
 */

import { parseCredentialInput, resolveCredentialForRequest } from "./git-credential-broker";

function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf-8");
    process.stdin.on("data", (chunk) => {
      data += chunk;
    });
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", reject);
  });
}

async function main(): Promise<void> {
  const action = process.argv[2] || "";
  // Never persist short-lived 15-minute tokens on store/erase.
  if (action === "store" || action === "erase") {
    process.exit(0);
  }
  if (action !== "get") {
    process.exit(0);
  }

  const stdinText = await readStdin();
  const query = parseCredentialInput(stdinText);
  const creds = await resolveCredentialForRequest({
    query,
    cwd: process.cwd(),
  });
  process.stdout.write(`username=${creds.username}\npassword=${creds.password}\n`);
}

main().catch((err) => {
  const msg = err instanceof Error ? err.message : String(err);
  process.stderr.write(`${msg}\n`);
  process.exit(1);
});
