/**
 * The extension's real bridge server, run without VS Code.
 *
 * For tests that drive the whole chain from a browser (static-apps e2e,
 * `local-bridge-vscode.spec.ts`): the same `LiteStepServer`, origin rules,
 * handshake and batches as the extension, with VS Code's status bar,
 * settings and editor replaced by lines on stdin and stdout.
 *
 *   node out/bridge-cli.js --port 9877 --allow https://app.example.com
 *
 * stdout, one JSON object per line:
 *   {"event":"listening","port":9877}
 *   {"event":"status","text":"...","tooltip":"..."}
 *   {"event":"warn"|"error","message":"..."}
 *   {"event":"received","filename":"output.py","sizeBytes":5,"content":"..."}
 *   {"event":"pushed","ok":true}
 *
 * stdin, one JSON object per line:
 *   {"cmd":"push","py":"...","ifc":"..."}   Push (the /delete:all + re-send flow)
 *   {"cmd":"stop"}
 *
 * Received files are reported, never written to disk.
 */
import * as readline from "node:readline";
import { LiteStepServer, type ServerHost } from "./server";

function emit(event: Record<string, unknown>): void {
  process.stdout.write(JSON.stringify(event) + "\n");
}

function parseArgs(argv: string[]): { port: number; allow: string[] } {
  let port = 9877;
  const allow: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--port") {
      port = Number(argv[++i]);
    } else if (argv[i] === "--allow") {
      allow.push(argv[++i]);
    } else {
      throw new Error(`unknown argument ${JSON.stringify(argv[i])}`);
    }
  }
  if (!Number.isInteger(port) || port <= 0) {
    throw new Error("--port needs a number");
  }
  return { port, allow };
}

function main(): void {
  const { port, allow } = parseArgs(process.argv.slice(2));
  // Server logs go to stderr so stdout stays one event per line.
  console.log = (...args: unknown[]) => process.stderr.write(args.map(String).join(" ") + "\n");
  console.warn = console.log;

  const host: ServerHost = {
    allowedOrigins: () => allow,
    warn: (message) => emit({ event: "warn", message }),
    error: (message) => emit({ event: "error", message }),
    showStatus: (text, tooltip) => {
      emit({ event: "status", text, tooltip });
      if (text.includes("Listening") && !announced) {
        announced = true;
        emit({ event: "listening", port });
      }
    },
  };
  let announced = false;

  const server = new LiteStepServer(host, port);
  server.onFileReceived = (file) => {
    emit({
      event: "received",
      id: file.id,
      filename: file.filename,
      sizeBytes: file.sizeBytes,
      content: file.data.toString("utf-8"),
    });
  };
  server.start();

  const lines = readline.createInterface({ input: process.stdin });
  lines.on("line", (line) => {
    let cmd: { cmd?: string; py?: string; ifc?: string };
    try {
      cmd = JSON.parse(line);
    } catch {
      emit({ event: "error", message: `not JSON: ${line.slice(0, 80)}` });
      return;
    }
    if (cmd.cmd === "push") {
      emit({ event: "pushed", ok: server.pushModelWithWipe(cmd.py ?? "", cmd.ifc ?? "") });
    } else if (cmd.cmd === "stop") {
      server.stop();
      process.exit(0);
    } else {
      emit({ event: "error", message: `unknown command ${JSON.stringify(cmd.cmd)}` });
    }
  });
  lines.on("close", () => {
    server.stop();
    process.exit(0);
  });
}

main();
