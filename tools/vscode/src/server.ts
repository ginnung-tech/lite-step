/**
 * WebSocket server for Lite-STEP sync.
 *
 * Runs on 127.0.0.1 (configurable port, default 9877).
 * Implements the file push/pull protocol:
 *   hello → file.push / ifc.push / status / error
 *
 * Who may connect, and the strict handshake for web pages, live in
 * `origin-policy.ts` and `bridge-gate.ts`.
 */


import type { IncomingMessage } from "node:http";
import { WebSocketServer, WebSocket, RawData } from "ws";
import * as vscode from "vscode";
import { ClientGate, makeVerifyClient, roleOf } from "./bridge-gate";
import { allowedFile, parseAllowlist, type Allowlist } from "./origin-policy";
import { OutstandingBatches, type ModelPair } from "./batches";
import {
  buildHello,
  buildStatus,
  buildError,
  buildPush,
  parseMessage,
  type ProtocolMessage,
} from "./protocol";

export interface IncomingFile {
  id: string;
  filename: string;
  sizeBytes: number;
  data: Buffer;
}

export type OnFileReceived = (file: IncomingFile) => void;

export class LiteStepServer {
  private _wss: WebSocketServer | null = null;
  private readonly _clients: Set<WebSocket> = new Set();
  private _port: number;
  private _onFileReceived: OnFileReceived | null = null;
  private readonly _gate = new ClientGate();
  private _allowlist: Allowlist = parseAllowlist([]);
  /** `/delete:all` batches still owed to the pages that ran them (`batches.ts`). */
  private readonly _outstanding = new OutstandingBatches();

  /** Per-client state for incoming binary frames */
  private readonly _incomingState: Map<
    WebSocket,
    { id: string; filename: string; sizeBytes: number }
  > = new Map();

  private readonly _statusBarItem: vscode.StatusBarItem;
  private readonly _heartbeatIntervals: Map<WebSocket, ReturnType<typeof setInterval>> =
    new Map();
  private readonly _onDidChangeStatus = new vscode.EventEmitter<void>();
  public readonly onDidChangeStatus = this._onDidChangeStatus.event;

  constructor(statusBarItem: vscode.StatusBarItem, port: number = 9877) {
    this._statusBarItem = statusBarItem;
    this._port = port;
    this._updateStatusBar();
  }

  get port(): number {
    return this._port;
  }

  get clientCount(): number {
    return this._clients.size;
  }

  get isRunning(): boolean {
    return this._wss !== null;
  }

  set onFileReceived(cb: OnFileReceived | null) {
    this._onFileReceived = cb;
  }

  start(port?: number): void {
    if (this._wss) {
      return;
    }

    if (port !== undefined) {
      this._port = port;
    }

    const config = vscode.workspace.getConfiguration("litestep");
    this._allowlist = parseAllowlist(config.get<unknown[]>("allowedOrigins", []));
    const ignored = this._allowlist.rejected;
    if (ignored.length > 0) {
      vscode.window.showWarningMessage(
        `Lite-STEP: ignored ${ignored.length} entr${ignored.length === 1 ? "y" : "ies"} in litestep.allowedOrigins. ${ignored.join("; ")}`
      );
    }

    this._wss = new WebSocketServer({
      host: "127.0.0.1",
      port: this._port,
      maxPayload: 100 * 1024 * 1024,
      verifyClient: makeVerifyClient(this._allowlist, this._port),
    });

    this._wss.on("connection", (ws, req) => this._handleConnection(ws, req));

    this._wss.on("error", (err) => {
      vscode.window.showErrorMessage(
        `Lite-STEP server error: ${err.message}`
      );
    });

    this._wss.on("listening", () => {
      console.log(`[Lite-STEP] Server listening on ws://127.0.0.1:${this._port}`);
      this._updateStatusBar();
    });
  }

  stop(): void {
    if (!this._wss) {
      return;
    }

    for (const interval of this._heartbeatIntervals.values()) {
      clearInterval(interval);
    }
    this._heartbeatIntervals.clear();

    for (const ws of this._clients) {
      try {
        ws.close(1000, "Server shutting down");
      } catch {
        // ignore
      }
    }
    this._clients.clear();
    this._incomingState.clear();
    this._gate.clear();

    this._wss.close();
    this._wss = null;

    console.log("[Lite-STEP] Server stopped");
    this._updateStatusBar();
  }

  /** Current active model pair (output.py and output.ifc) */

  /**
   * Push the Python model and its compiled IFC: send /delete:all, then
   * re-send the pair when the reloaded page asks for this batch by id.
   *
   * Returns false, and keeps nothing, when no client is connected: a push is
   * a user action that either reaches a page now or fails (protocol 1.2).
   */
  pushModelWithWipe(
    pyContent: string,
    ifcContent: string,
    pyFilename: string = "output.py",
    ifcFilename: string = "output.ifc"
  ): boolean {
    const batchId = `vsc-batch-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const recipients = this._readyClients();
    if (recipients.length === 0) {
      console.log("[Lite-STEP] No clients connected; nothing pushed.");
      return false;
    }
    for (const id of this._outstanding.dropExpired()) {
      console.warn(`[Lite-STEP] Batch ${id} was not asked for within two minutes; dropped.`);
    }
    this._outstanding.add(batchId, { pyContent, pyFilename, ifcContent, ifcFilename }, recipients.length);

    console.log(`[Lite-STEP] Stored current model pair in server memory (batch: ${batchId}).`);

    const wipeMsg = JSON.stringify({
      type: "file.push",
      version: "1.1",
      id: `wipe-${Date.now()}`,
      direction: "to-client",
      filename: pyFilename,
      fileType: "python",
      command: "/delete:all",
      batchId,
    });

    for (const ws of recipients) {
      try {
        ws.send(wipeMsg);
        console.log(`[Lite-STEP] Sent /delete:all wipe command to client.`);
      } catch (e) {
        console.error(`[Lite-STEP] Failed to send wipe command: ${e}`);
      }
    }
    return true;
  }

  /** Send a batch's model pair to a client */
  private _sendModelPair(ws: WebSocket, pair: ModelPair): void {
    const { pyContent, pyFilename, ifcContent, ifcFilename } = pair;
    const batchId = `vsc-pair-${Date.now()}`;

    try {
      const pyPush = JSON.stringify({
        type: "file.push",
        version: "1.1",
        id: `pair-py-${Date.now()}`,
        direction: "to-client",
        filename: pyFilename,
        fileType: "python",
        content: pyContent,
        batchId,
      });
      ws.send(pyPush);
      console.log(`[Lite-STEP] Sent Python model '${pyFilename}' to client.`);

      const ifcPush = JSON.stringify({
        type: "file.push",
        version: "1.1",
        id: `pair-ifc-${Date.now()}`,
        direction: "to-client",
        filename: ifcFilename,
        fileType: "ifc",
        content: ifcContent,
        batchId,
      });
      ws.send(ifcPush);
      console.log(`[Lite-STEP] Sent IFC model '${ifcFilename}' to client.`);
    } catch (e) {
      console.error(`[Lite-STEP] Failed sending model pair to client: ${e}`);
    }
  }

  dispose(): void {
    this.stop();
  }

  /** Send raw text message to all connected clients */
  private _broadcast(text: string): void {
    for (const ws of this._readyClients()) {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(text);
        } catch (e) {
          console.error(`[Lite-STEP] Error sending to client: ${e}`);
        }
      }
    }
  }

  /** Push an IFC file to all connected clients */
  pushIfc(ifcBytes: Buffer, filename: string = "model.ifc"): void {
    const id = `vsc-ifc-${Date.now()}`;
    const pushMsg = buildPush(id, "to-client", filename, ifcBytes.length);

    for (const ws of this._readyClients()) {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(pushMsg);
          ws.send(ifcBytes);
          console.log(`[Lite-STEP] Pushed IFC '${filename}' (${ifcBytes.length} bytes)`);
        } catch (e) {
          console.error(`[Lite-STEP] Error pushing IFC: ${e}`);
        }
      }
    }
  }

  /** Push a Python or IFC file inline to all connected clients */
  pushFile(
    filename: string,
    content: string,
    options?: {
      fileType?: "python" | "ifc";
      batchId?: string;
      clearExisting?: boolean;
      overwrite?: boolean;
      command?: string;
    }
  ): void {
    const id = `vsc-file-${Date.now()}`;
    const pushMsg = buildPush(
      id,
      "to-client",
      filename,
      content.length,
      content,
      options
    );

    this._broadcast(pushMsg);
    console.log(
      `[Lite-STEP] Pushed inline file '${filename}' (${content.length} chars)`
    );
  }

  /** Clients that completed the handshake: the only ones a push may reach. */
  private _readyClients(): WebSocket[] {
    return [...this._clients].filter((ws) => this._gate.isReady(ws));
  }

  private _handleConnection(ws: WebSocket, req: IncomingMessage): void {
    // verifyClient already refused anything without a role; a null here would
    // mean the two checks disagree, and the safe reading is "not allowed".
    const role = roleOf(req, this._port, this._allowlist);
    if (role === null) {
      ws.close(1008, "not allowed");
      return;
    }
    const origin = typeof req.headers.origin === "string" ? req.headers.origin : null;
    this._gate.admit(ws, role, origin);
    this._clients.add(ws);
    console.log(
      `[Lite-STEP] Client connected. Total clients: ${this._clients.size}`
    );

    // Send hello handshake
    try {
      ws.send(buildHello());
    } catch (e) {
      console.error(`[Lite-STEP] Error sending hello: ${e}`);
    }

    // Ping/pong heartbeat
    const pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.ping();
      }
    }, 30_000);
    this._heartbeatIntervals.set(ws, pingInterval);

    ws.on("message", (data: RawData, isBinary: boolean) => {
      if (isBinary) {
        this._handleBinary(ws, data as Buffer);
      } else {
        this._handleText(ws, data.toString());
      }
    });

    ws.on("close", () => {
      this._clients.delete(ws);
      this._incomingState.delete(ws);
      this._gate.release(ws);
      const interval = this._heartbeatIntervals.get(ws);
      if (interval) {
        clearInterval(interval);
        this._heartbeatIntervals.delete(ws);
      }
      console.log(
        `[Lite-STEP] Client disconnected. Total clients: ${this._clients.size}`
      );
      this._updateStatusBar();
    });

    ws.on("error", (err) => {
      console.error(`[Lite-STEP] Client error: ${err.message}`);
    });

    this._updateStatusBar();
  }

  private _handlePushMessage(ws: WebSocket, msg: ProtocolMessage & { type: "file.push" | "ifc.push" }): void {
    if (msg.direction !== "to-vscode" && msg.direction !== "to-blender") {
      return;
    }
    const filename = msg.filename || "pushed_file.py";
    if (!allowedFile(filename, msg.fileType)) {
      console.warn(`[Lite-STEP] Refused pushed file '${filename}': only .py and .ifc are carried`);
      ws.send(buildError(msg.id, "Only .py and .ifc files are accepted"));
      return;
    }
    if (msg.content) {
      this._handleInlineFile(ws, msg.id, filename, msg.content, msg.sizeBytes);
    } else {
      this._incomingState.set(ws, {
        id: msg.id,
        filename,
        sizeBytes: msg.sizeBytes,
      });
      console.log(`[Lite-STEP] Expecting file '${filename}' (${msg.sizeBytes} bytes)`);
    }
  }

  private _handleInlineFile(ws: WebSocket, id: string, filename: string, content: string, sizeBytes?: number): void {
    try {
      if (this._onFileReceived) {
        this._onFileReceived({
          id,
          filename,
          sizeBytes: sizeBytes || content.length,
          data: Buffer.from(content, "utf-8"),
        });
      }
      ws.send(buildStatus(id, "loaded"));
    } catch (e) {
      console.error(`[Lite-STEP] Error handling inline file: ${e}`);
      ws.send(buildError(id, `Load failed: ${e}`));
    }
  }

  private _handleStatusMessage(ws: WebSocket, msg: ProtocolMessage & { type: "status" }): void {
    console.log(`[Lite-STEP] Status for ${msg.id}: ${msg.status}`);
    if (msg.status === "connected") {
      console.log(
        `[Lite-STEP] Web client connected handshake (capabilities: ${JSON.stringify(msg.capabilities || [])})`
      );
      this._updateStatusBar();
      // Protocol 1.2: a connection is not a request for the model. The one
      // re-send is the batch whose /delete:all this client ran, asked for by
      // id, once, within two minutes.
      if (msg.wantsModel !== true) {
        return;
      }
      const pair = this._outstanding.claim(msg.awaitingBatchId);
      if (pair) {
        this._sendModelPair(ws, pair);
      } else {
        console.warn(
          `[Lite-STEP] Client asked for batch ${JSON.stringify(msg.awaitingBatchId)}, which is not in flight; nothing sent`
        );
      }
    }
  }

  private _handleText(ws: WebSocket, raw: string): void {
    try {
      const msg: ProtocolMessage = parseMessage(raw);
      const status = msg.type === "status" ? msg.status : undefined;
      if (!this._gate.accept(ws, msg.type, status)) {
        return;
      }
      switch (msg.type) {
        case "file.push":
        case "ifc.push":
          this._handlePushMessage(ws, msg);
          break;
        case "status":
          this._handleStatusMessage(ws, msg);
          break;
        case "hello":
          console.log("[Lite-STEP] Received hello-ack from client");
          break;
      }
    } catch (e) {
      console.error(`[Lite-STEP] Error parsing message: ${e}`);
      ws.send(buildError(null, `Parse error: ${e}`));
    }
  }

  private _handleBinary(ws: WebSocket, data: Buffer): void {
    if (!this._gate.accept(ws, null)) {
      return;
    }
    const state = this._incomingState.get(ws);
    if (!state) {
      console.warn(
        "[Lite-STEP] Unexpected binary frame without prior control message"
      );
      return;
    }
    this._incomingState.delete(ws);

    console.log(
      `[Lite-STEP] Received file '${state.filename}', ${data.length} bytes`
    );

    if (this._onFileReceived) {
      try {
        this._onFileReceived({
          id: state.id,
          filename: state.filename,
          sizeBytes: state.sizeBytes,
          data,
        });
        ws.send(buildStatus(state.id, "loaded"));
      } catch (e) {
        console.error(`[Lite-STEP] Error processing file: ${e}`);
        ws.send(buildError(state.id, `Load failed: ${e}`));
      }
    }
  }

  private _updateStatusBar(): void {
    if (this._wss) {
      const count = this._clients.size;
      if (count > 0) {
        this._statusBarItem.text = `$(check) Lite-STEP: ${count} client${count > 1 ? "s" : ""}`;
        const origins = this._gate.webOrigins();
        this._statusBarItem.tooltip =
          `Lite-STEP server running on port ${this._port} — ${count} connected` +
          (origins.length > 0 ? ` (${origins.join(", ")})` : "");
      } else {
        this._statusBarItem.text = `$(radio-tower) Lite-STEP: Listening`;
        this._statusBarItem.tooltip = `Lite-STEP server listening on port ${this._port}`;
      }
    } else {
      this._statusBarItem.text = `$(circle-slash) Lite-STEP: Off`;
      this._statusBarItem.tooltip = "Lite-STEP server is not running";
    }
    this._statusBarItem.show();
    this._onDidChangeStatus.fire();
  }
}
