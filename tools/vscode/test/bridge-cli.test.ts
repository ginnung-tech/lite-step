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

  test("Push sends the pair, no command: the page replaces its current project's model", async () => {
    const page = await connect("https://ginnung.tech");
    page.ws.send(JSON.stringify({ type: "status", id: "s", status: "connected", capabilities: ["file.push", "command", "new:project"] }));
    await wait(100);
    child.stdin!.write(JSON.stringify({ cmd: "push", py: "x = 1", ifc: "ISO-10303-21;" }) + "\n");
    assert.deepEqual(await next((e) => e.event === "pushed"), { event: "pushed", ok: true });
    await wait(150);
    const files = page.frames.filter((f) => f.type === "file.push");
    assert.deepEqual(files.map((f) => [f.filename, f.fileType, f.content]), [["output.py", "python", "x = 1"], ["output.ifc", "ifc", "ISO-10303-21;"]]);
    assert.ok(files.every((f) => f.command === undefined), "a Push never carries a command");
    assert.equal(files[0].batchId, files[1].batchId);
    page.ws.close();
  });

  test("Push as New Project: one frame, then the pair once to the new project's page by batch id", async () => {
    const page = await connect("https://ginnung.tech");
    page.ws.send(JSON.stringify({ type: "status", id: "s", status: "connected", capabilities: ["file.push", "command", "new:project"] }));
    await wait(100);
    child.stdin!.write(JSON.stringify({ cmd: "push-new", py: "x = 2", ifc: "ISO-2", name: "Carport study" }) + "\n");
    assert.deepEqual(await next((e) => e.event === "pushed-new"), { event: "pushed-new", result: "sent" });
    await wait(150);
    const sent = page.frames.filter((f) => f.type === "file.push");
    assert.equal(sent.length, 1, "only the command frame: the .ifc must not reach the current project");
    const cmd = sent[0];
    assert.equal(cmd.command, "/new:project");
    assert.equal(cmd.projectName, "Carport study");
    assert.equal(typeof cmd.content, "string");
    assert.equal(typeof cmd.batchId, "string");
    page.ws.close();

    // The new project's page asks for that batch by id: the pair, once, no command.
    const moved = await connect("https://ginnung.tech");
    moved.ws.send(JSON.stringify({ type: "status", id: "s2", status: "connected", wantsModel: true, awaitingBatchId: cmd.batchId }));
    await wait(200);
    const files = moved.frames.filter((f) => f.type === "file.push");
    assert.deepEqual(files.map((f) => [f.filename, f.content]), [["output.py", "x = 2"], ["output.ifc", "ISO-2"]]);
    assert.ok(files.every((f) => f.command === undefined));
    moved.ws.close();

    const again = await connect("https://ginnung.tech");
    again.ws.send(JSON.stringify({ type: "status", id: "s3", status: "connected", wantsModel: true, awaitingBatchId: cmd.batchId }));
    await wait(200);
    assert.equal(again.frames.filter((f) => f.type === "file.push").length, 0);
    again.ws.close();
  });

  test("Push as New Project is not sent to a page that cannot open projects", async () => {
    const page = await connect("https://ginnung.tech");
    page.ws.send(JSON.stringify({ type: "status", id: "s", status: "connected", capabilities: ["file.push", "command"] }));
    await wait(100);
    child.stdin!.write(JSON.stringify({ cmd: "push-new", py: "x", ifc: "I", name: "n" }) + "\n");
    assert.deepEqual(await next((e) => e.event === "pushed-new"), { event: "pushed-new", result: "unsupported" });
    await wait(100);
    assert.equal(page.frames.filter((f) => f.type === "file.push").length, 0);
    page.ws.close();
  });

  test("a refusal from the page reaches the user", async () => {
    const page = await connect("https://ginnung.tech");
    page.ws.send(JSON.stringify({ type: "status", id: "s", status: "connected" }));
    page.ws.send(JSON.stringify({ type: "error", version: "1.4", id: "newproj-1", error: "Projects are not available on this Ginnung page." }));
    const warn = await next((e) => e.event === "warn" && String(e.message).includes("Projects are not available"));
    assert.ok(warn);
    page.ws.close();
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
