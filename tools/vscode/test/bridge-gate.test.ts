// The door on a real socket: the same verifyClient and ClientGate the
// extension's server uses, on a free port.
import { afterEach, beforeEach, describe,  test } from "node:test";
import assert from "node:assert/strict";
import { WebSocket, WebSocketServer } from "ws";
import type { AddressInfo } from "node:net";
import { ClientGate, makeVerifyClient, roleOf } from "../src/bridge-gate";
import { parseAllowlist } from "../src/origin-policy";

const TIMEOUT_MS = 150;
let wss: WebSocketServer;
let port: number;
let gate: ClientGate;
let handled: string[];

beforeEach(async () => {
  handled = [];
  gate = new ClientGate(TIMEOUT_MS);
  const allowlist = parseAllowlist([]);
  // Port 0 is only known after listen, so the check reads it lazily.
  wss = new WebSocketServer({
    host: "127.0.0.1",
    port: 0,
    verifyClient: (info, done) => makeVerifyClient(allowlist, port)(info, done),
  });
  await new Promise<void>((r) => wss.once("listening", () => r()));
  port = (wss.address() as AddressInfo).port;
  wss.on("connection", (ws, req) => {
    const role = roleOf(req, port, allowlist)!;
    gate.admit(ws, role, (req.headers.origin as string) ?? null);
    ws.send(JSON.stringify({ type: "hello" }));
    ws.on("message", (data) => {
      const msg = JSON.parse(data.toString());
      if (gate.accept(ws, msg.type, msg.status)) handled.push(msg.type);
    });
    ws.on("close", () => gate.release(ws));
  });
});

afterEach(async () => {
  for (const c of wss.clients) c.terminate();
  await new Promise<void>((r) => wss.close(() => r()));
});

/** Resolves with what happened to a connection attempt. */
function attempt(headers: Record<string, string>): Promise<{ ws: WebSocket | null; status: number | null; hello: boolean }> {
  return new Promise((resolve) => {
    const ws = new WebSocket(`ws://127.0.0.1:${port}`, { headers });
    ws.on("unexpected-response", (_req, res) => resolve({ ws: null, status: res.statusCode ?? null, hello: false }));
    ws.on("error", () => resolve({ ws: null, status: null, hello: false }));
    ws.on("message", () => resolve({ ws, status: 101, hello: true }));
  });
}

const closed = (ws: WebSocket) => new Promise<number>((r) => ws.on("close", (code) => r(code)));
const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe("upgrade", { timeout: 5000 }, () => {
  test("a rogue or look-alike origin gets a bare 403 and no hello", async () => {
    for (const origin of ["https://evil.example", "https://ginnung.tech.evil.example", "http://localhost:5173", "null"]) {
      const got = await attempt({ Origin: origin });
      assert.equal(got.status, 403, origin);
      assert.equal(got.hello, false, origin);
    }
  });

  test("a forged Host is refused even with the right Origin", async () => {
    const got = await attempt({ Origin: "https://ginnung.tech", Host: `evil.example:${port}` });
    assert.equal(got.status, 403);
  });

  test("the allowed origin and a local program both get hello", async () => {
    const web = await attempt({ Origin: "https://ginnung.tech" });
    assert.equal(web.hello, true);
    const local = await attempt({});
    assert.equal(local.hello, true);
    web.ws!.close();
    local.ws!.close();
  });
});

describe("handshake", { timeout: 5000 }, () => {
  test("a web client that stays silent is closed after the timeout", async () => {
    const { ws } = await attempt({ Origin: "https://ginnung.tech" });
    assert.equal(await closed(ws!), 1008);
  });

  test("a web client that sends anything before status: connected is closed, and the frame is not handled", async () => {
    const { ws } = await attempt({ Origin: "https://ginnung.tech" });
    ws!.send(JSON.stringify({ type: "file.push", filename: "x.py", content: "x" }));
    assert.equal(await closed(ws!), 1008);
    assert.deepEqual(handled, []);
  });

  test("after status: connected a web client stays open and its frames are handled", async () => {
    const { ws } = await attempt({ Origin: "https://ginnung.tech" });
    assert.deepEqual(gate.webOrigins(), []);
    ws!.send(JSON.stringify({ type: "status", status: "connected" }));
    ws!.send(JSON.stringify({ type: "file.push" }));
    await wait(TIMEOUT_MS * 2);
    assert.equal(ws!.readyState, WebSocket.OPEN);
    assert.deepEqual(handled, ["status", "file.push"]);
    assert.deepEqual(gate.webOrigins(), ["https://ginnung.tech"]);
    ws!.close();
  });

  test("a local program needs no handshake and is never timed out", async () => {
    const { ws } = await attempt({});
    ws!.send(JSON.stringify({ type: "ifc.push" }));
    await wait(TIMEOUT_MS * 2);
    assert.equal(ws!.readyState, WebSocket.OPEN);
    assert.deepEqual(handled, ["ifc.push"]);
    ws!.close();
  });
});
