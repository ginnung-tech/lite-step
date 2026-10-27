/**
 * WebSocket server for Lite-STEP sync.
 *
 * Runs on 127.0.0.1 (configurable port, default 9877).
 * Implements the file push/pull protocol:
 *   hello → file.push / ifc.push / ifc.pull / status / error / document.sync
 */

import { WebSocketServer, WebSocket, RawData } from "ws";
import * as vscode from "vscode";
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

export type OnFileReceived = (file: IncomingFile) => void | Promise<void>;

export class LiteStepServer {
  private _wss: WebSocketServer | null = null;
  private readonly _clients: Set<WebSocket> = new Set();
  private _port: number;
  private _onFileReceived: OnFileReceived | null = null;

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
    const allowedOrigins = config.get<string[]>("allowedOrigins", [
      "http://localhost:5173",
      "http://127.0.0.1:5173",
      "http://localhost:5180",
      "http://127.0.0.1:5180",
      "vscode-webview://",
    ]);

    this._wss = new WebSocketServer({
      host: "127.0.0.1",
      port: this._port,
      maxPayload: 100 * 1024 * 1024,
      verifyClient: (info: { origin: string }) => {
        const origin = info.origin;
        if (!origin) return true;
        return allowedOrigins.some((allowed) => origin === allowed || origin.startsWith(allowed));
      },
    });

    this._wss.on("connection", (ws) => this._handleConnection(ws));

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

    this._wss.close();
    this._wss = null;

    console.log("[Lite-STEP] Server stopped");
    this._updateStatusBar();
  }

  /** Current active model pair (output.py and output.ifc) */
  private _currentModel: {
    pyContent: string;
    pyFilename: string;
    ifcContent: string;
    ifcFilename: string;
  } | null = null;

  /** Push both Python model and compiled IFC model by issuing /delete:all and waiting for reconnect */
  pushModelWithWipe(
    pyContent: string,
    ifcContent: string,
    pyFilename: string = "output.py",
    ifcFilename: string = "output.ifc"
  ): void {
    const batchId = `vsc-batch-${Date.now()}`;
    this._currentModel = {
      pyContent,
      pyFilename,
      ifcContent,
      ifcFilename,
    };

    console.log(`[Lite-STEP] Stored current model pair in server memory (batch: ${batchId}).`);

    if (this._clients.size === 0) {
      console.log("[Lite-STEP] No clients connected to push wipe command to.");
      return;
    }

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

    for (const ws of this._clients) {
      try {
        ws.send(wipeMsg);
        console.log(`[Lite-STEP] Sent /delete:all wipe command to client.`);
      } catch (e) {
        console.error(`[Lite-STEP] Failed to send wipe command: ${e}`);
      }
    }
  }

  /** Send current stored model pair to a client */
  sendModelPairToClient(ws: WebSocket): void {
    if (!this._currentModel) {
      console.log("[Lite-STEP] No stored model to send to client.");
      return;
    }

    const { pyContent, pyFilename, ifcContent, ifcFilename } = this._currentModel;
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

  /** Send document.sync message to all connected clients */
  broadcastDocumentSync(
    documentId: "lite_step" | "ifc_step_file",
    content: string,
    contentType: "text/x-python" | "application/x-step",
    fileName: string
  ): void {
    const id = `vsc-sync-${Date.now()}`;
    const msg = JSON.stringify({
      type: "document.sync",
      version: "1.1",
      id,
      payload: {
        documentId,
        content,
        contentType,
        metadata: {
          lastModified: Date.now(),
          source: "vscode",
          fileName,
        },
      },
    });

    this._broadcast(msg);
  }

  /** Send a document.sync text message to all clients */
  sendTextSync(jsonMessage: string): void {
    this._broadcast(jsonMessage);
  }

  dispose(): void {
    this.stop();
  }

  /** Send raw text message to all connected clients */
  private _broadcast(text: string): void {
    for (const ws of this._clients) {
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

    for (const ws of this._clients) {
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

  private _handleConnection(ws: WebSocket): void {
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
      this.sendModelPairToClient(ws);
    }
  }

  private _handleText(ws: WebSocket, raw: string): void {
    try {
      const msg: ProtocolMessage = parseMessage(raw);
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
        this._statusBarItem.tooltip = `Lite-STEP server running on port ${this._port} — ${count} connected`;
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
