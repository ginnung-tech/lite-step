// The headless runner the cross-service e2e test starts: the real server,
// driven over stdin/stdout.
import { after, before, describe, test } from "node:test";
import assert from "node:assert/strict";
import { spawn, type ChildProcess } from "node:child_process";
import * as path from "node:path";
import * as readline from "node:readline";
import { WebSocket } from "ws";

const CLI = path.resolve(__dirname, "..", "out-cli", "bridge-cli.js");
const PORT = 19000 + Math.floor(Math.random() * 900);
let child: ChildProcess;
const events: Array<Record<string, unknown>> = [];

function next(match: (e: Record<string, unknown>) => boolean, ms = 4000): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const tick = () => {
      const i = events.findIndex(match);
      if (i >= 0) return resolve(events.splice(i, 1)[0]);
      if (Date.now() - started > ms) return reject(new Error(`no matching event; saw ${JSON.stringify(events)}`));
      setTimeout(tick, 20);
    };
    tick();
  });
}

function connect(origin?: string): Promise<{ ws: WebSocket; frames: Array<Record<string, unknown>> }> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`ws://127.0.0.1:${PORT}`, origin ? { headers: { Origin: origin } } : {});
    const frames: Array<Record<string, unknown>> = [];
    ws.on("message", (d) => {
      frames.push(JSON.parse(d.toString()));
      if (frames.length === 1) resolve({ ws, frames });
    });
    ws.on("unexpected-response", (_q, res) => reject(new Error(`status ${res.statusCode}`)));
    ws.on("error", reject);
  });
}

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe("bridge-cli", { timeout: 15000 }, () => {
  before(async () => {
    child = spawn(process.execPath, [CLI, "--port", String(PORT), "--allow", "https://app.example.com"], { stdio: ["pipe", "pipe", "ignore"] });
    readline.createInterface({ input: child.stdout! }).on("line", (l) => events.push(JSON.parse(l)));
    await next((e) => e.event === "listening");
  });

  after(() => {
    child.stdin!.end();
  });

  test("a rogue origin is refused, an allowed one says hello", async () => {
    await assert.rejects(connect("https://evil.example"), /status 403/);
    const { ws, frames } = await connect("https://app.example.com");
    assert.equal(frames[0].type, "hello");
    ws.close();
  });

  test("Push with no page connected fails and sends nothing", async () => {
    child.stdin!.write(JSON.stringify({ cmd: "push", py: "x = 1", ifc: "ISO" }) + "\n");
    assert.deepEqual(await next((e) => e.event === "pushed"), { event: "pushed", ok: false });
  });

  test("Push sends /delete:all, and the reloaded page gets the pair once by batch id", async () => {
    const page = await connect("https://ginnung.tech");
    page.ws.send(JSON.stringify({ type: "status", id: "s", status: "connected", wantsModel: false }));
    await wait(100);
    child.stdin!.write(JSON.stringify({ cmd: "push", py: "x = 1", ifc: "ISO-10303-21;" }) + "\n");
    assert.deepEqual(await next((e) => e.event === "pushed"), { event: "pushed", ok: true });
    await wait(100);
    const wipe = page.frames.find((f) => f.command === "/delete:all");
    assert.ok(wipe, JSON.stringify(page.frames));
    // What the Ginnung page requires of any file.push (contract local-bridge,
    // `content` required): without it the page drops the frame, wipe and all.
    assert.equal(typeof wipe!.content, "string");
    assert.equal(wipe!.fileType, "python");
    assert.equal(typeof wipe!.batchId, "string");
    page.ws.close();

    const reloaded = await connect("https://ginnung.tech");
    reloaded.ws.send(JSON.stringify({ type: "status", id: "s2", status: "connected", wantsModel: true, awaitingBatchId: wipe!.batchId }));
    await wait(200);
    const files = reloaded.frames.filter((f) => f.type === "file.push").map((f) => [f.filename, f.content]);
    assert.deepEqual(files, [["output.py", "x = 1"], ["output.ifc", "ISO-10303-21;"]]);
    assert.ok(reloaded.frames.every((f) => f.command === undefined));
    reloaded.ws.close();

    const again = await connect("https://ginnung.tech");
    again.ws.send(JSON.stringify({ type: "status", id: "s3", status: "connected", wantsModel: true, awaitingBatchId: wipe!.batchId }));
    await wait(200);
    assert.equal(again.frames.filter((f) => f.type === "file.push").length, 0);
    again.ws.close();
  });

  test("a file pushed from the page is reported, not written", async () => {
    const page = await connect("https://ginnung.tech");
    page.ws.send(JSON.stringify({ type: "status", id: "s", status: "connected" }));
    page.ws.send(JSON.stringify({ type: "file.push", id: "f1", version: "1.3", direction: "to-vscode", filename: "output.py", fileType: "python", content: "y = 2", sizeBytes: 5 }));
    const got = await next((e) => e.event === "received");
    assert.equal(got.filename, "output.py");
    assert.equal(got.content, "y = 2");
    page.ws.close();
  });
});
