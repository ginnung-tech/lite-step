/**
 * WebSocket server for Ginnung sync.
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

export type OnFileReceived = (file: IncomingFile) => void;

export class GinnungServer {
  private _wss: WebSocketServer | null = null;
  private _clients: Set<WebSocket> = new Set();
  private _port: number;
  private _onFileReceived: OnFileReceived | null = null;

  /** Per-client state for incoming binary frames */
  private _incomingState: Map<
    WebSocket,
    { id: string; filename: string; sizeBytes: number }
  > = new Map();

  private _statusBarItem: vscode.StatusBarItem;
  private _heartbeatIntervals: Map<WebSocket, ReturnType<typeof setInterval>> =
    new Map();
  private _onDidChangeStatus = new vscode.EventEmitter<void>();
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

    this._wss = new WebSocketServer({
      host: "127.0.0.1",
      port: this._port,
      maxPayload: 100 * 1024 * 1024,
    });

    this._wss.on("connection", (ws) => this._handleConnection(ws));

    this._wss.on("error", (err) => {
      vscode.window.showErrorMessage(
        `Ginnung server error: ${err.message}`
      );
    });

    this._wss.on("listening", () => {
      console.log(`[Ginnung] Server listening on ws://127.0.0.1:${this._port}`);
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

    console.log("[Ginnung] Server stopped");
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
    pyFilename: string = "output.py",
    ifcContent: string,
    ifcFilename: string = "output.ifc"
  ): void {
    const batchId = `vsc-batch-${Date.now()}`;
    this._currentModel = {
      pyContent,
      pyFilename,
      ifcContent,
      ifcFilename,
    };

    if (!this._wss || this._clients.size === 0) {
      console.log("[Ginnung] No clients connected; model pair cached for next connection.");
      return;
    }

    // Send initial /delete:all command to trigger account wipe & page reload
    const wipeMsg = buildPush(
      `vsc-wipe-${Date.now()}`,
      "to-ginnung",
      pyFilename,
      Buffer.byteLength(pyContent, "utf-8"),
      pyContent,
      {
        fileType: "python",
        batchId,
        command: "/delete:all",
      }
    );
    this._broadcast(wipeMsg);

    console.log(
      `[Ginnung] Sent /delete:all command to Web App (batchId: ${batchId}). Waiting for reconnect to send both output.py and output.ifc.`
    );
  }

  /** Send both output.py and output.ifc to a client with NO command */
  sendModelPairToClient(ws: WebSocket): void {
    if (!this._currentModel) {
      return;
    }

    const { pyContent, pyFilename, ifcContent, ifcFilename } =
      this._currentModel;
    const batchId = `vsc-batch-${Date.now()}`;

    console.log(
      `[Ginnung] Sending both '${pyFilename}' and '${ifcFilename}' to Web App (batchId: ${batchId}, no command)`
    );

    // 1. Send Python file (output.py)
    const pyMsg = buildPush(
      `vsc-py-${Date.now()}`,
      "to-ginnung",
      pyFilename,
      Buffer.byteLength(pyContent, "utf-8"),
      pyContent,
      {
        fileType: "python",
        batchId,
      }
    );

    // 2. Send IFC file (output.ifc)
    const ifcMsg = buildPush(
      `vsc-ifc-${Date.now()}`,
      "to-ginnung",
      ifcFilename,
      Buffer.byteLength(ifcContent, "utf-8"),
      ifcContent,
      {
        fileType: "ifc",
        batchId,
      }
    );

    try {
      ws.send(pyMsg);
      ws.send(ifcMsg);
    } catch (e) {
      console.error(`[Ginnung] Error sending model pair to client: ${e}`);
    }
  }

  private _broadcast(textMessage: string): void {
    for (const ws of this._clients) {
      try {
        ws.send(textMessage);
      } catch (e) {
        console.error(`[Ginnung] Error broadcasting to client: ${e}`);
      }
    }
  }

  /** Send a file (.py or .ifc) to all connected clients with overwrite/clear command */
  sendFileToClients(
    fileBytes: Buffer,
    filename: string,
    options: { overwrite?: boolean; clearExisting?: boolean; command?: string } = {
      overwrite: true,
      clearExisting: true,
      command: "/delete:all",
    }
  ): void {
    if (!this._wss || this._clients.size === 0) {
      return;
    }

    const id = `vsc-push-${Date.now()}`;
    const textContent = fileBytes.toString("utf-8");
    const pushMsg = buildPush(
      id,
      "to-ginnung",
      filename,
      fileBytes.length,
      textContent,
      options
    );

    for (const ws of this._clients) {
      try {
        ws.send(pushMsg);
        // Send binary payload as well for clients expecting binary frame
        ws.send(fileBytes);
        console.log(`[Ginnung] Sent file '${filename}' to client (overwrite=true, command=/delete:all)`);
      } catch (e) {
        console.error(`[Ginnung] Error sending to client: ${e}`);
      }
    }
  }

  /** Send a document.sync text message to all clients */
  sendTextSync(jsonMessage: string): void {
    for (const ws of this._clients) {
      try {
        ws.send(jsonMessage);
      } catch (e) {
        console.error(`[Ginnung] Error sending sync: ${e}`);
      }
    }
  }

  dispose(): void {
    this.stop();
  }

  // ── Private ──────────────────────────────────────────────

  private _handleConnection(ws: WebSocket): void {
    this._clients.add(ws);
    console.log("[Ginnung] Client connected");

    ws.send(buildHello());

    const hb = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.ping();
      }
    }, 15_000);
    this._heartbeatIntervals.set(ws, hb);

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
      console.log("[Ginnung] Client disconnected");
      this._updateStatusBar();
    });

    ws.on("error", (err) => {
      console.error(`[Ginnung] Client error: ${err.message}`);
    });

    this._updateStatusBar();
  }

  private _handleText(ws: WebSocket, raw: string): void {
    try {
      const msg: ProtocolMessage = parseMessage(raw);

      if (msg.type === "file.push" || msg.type === "ifc.push") {
        if (msg.direction === "to-vscode" || msg.direction === "to-blender") {
          const filename = msg.filename || "pushed_file.py";
          if (msg.content) {
            // Inline content present
            try {
              if (this._onFileReceived) {
                this._onFileReceived({
                  id: msg.id,
                  filename,
                  sizeBytes: msg.sizeBytes || msg.content.length,
                  data: Buffer.from(msg.content, "utf-8"),
                });
              }
              ws.send(buildStatus(msg.id, "loaded"));
            } catch (e) {
              console.error(`[Ginnung] Error handling inline file: ${e}`);
              ws.send(buildError(msg.id, `Load failed: ${e}`));
            }
          } else {
            // Expect binary payload frame next
            this._incomingState.set(ws, {
              id: msg.id,
              filename,
              sizeBytes: msg.sizeBytes,
            });
            console.log(
              `[Ginnung] Expecting file '${filename}' (${msg.sizeBytes} bytes)`
            );
          }
        }
      } else if (msg.type === "status") {
        console.log(`[Ginnung] Status for ${msg.id}: ${msg.status}`);
        if (msg.status === "connected") {
          console.log(
            `[Ginnung] Web client connected handshake (capabilities: ${JSON.stringify(msg.capabilities || [])})`
          );
          this.sendModelPairToClient(ws);
        }
      } else if (msg.type === "hello") {
        console.log("[Ginnung] Received hello-ack from client");
      }
    } catch (e) {
      console.error(`[Ginnung] Error parsing message: ${e}`);
      ws.send(buildError(null, `Parse error: ${e}`));
    }
  }

  private _handleBinary(ws: WebSocket, data: Buffer): void {
    const state = this._incomingState.get(ws);
    if (!state) {
      console.warn(
        "[Ginnung] Unexpected binary frame without prior control message"
      );
      return;
    }
    this._incomingState.delete(ws);

    console.log(
      `[Ginnung] Received file '${state.filename}', ${data.length} bytes`
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
        console.error(`[Ginnung] Error processing file: ${e}`);
        ws.send(buildError(state.id, `Load failed: ${e}`));
      }
    }
  }

  private _updateStatusBar(): void {
    if (this._wss) {
      const count = this._clients.size;
      if (count > 0) {
        this._statusBarItem.text = `$(check) Ginnung: ${count} client${count > 1 ? "s" : ""}`;
        this._statusBarItem.tooltip = `Ginnung server running on port ${this._port} — ${count} connected`;
      } else {
        this._statusBarItem.text = `$(radio-tower) Ginnung: Listening`;
        this._statusBarItem.tooltip = `Ginnung server listening on port ${this._port}`;
      }
    } else {
      this._statusBarItem.text = `$(circle-slash) Ginnung: Off`;
      this._statusBarItem.tooltip = "Ginnung server is not running";
    }
    this._statusBarItem.show();
    this._onDidChangeStatus.fire();
  }
}
