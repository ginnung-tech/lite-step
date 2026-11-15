/**
 * Wire protocol types and builders for the Lite-STEP WebSocket bridge.
 *
 * Two-way file push. A push is a user action (protocol 1.2): nothing is sent
 * because a file was saved or a page connected.
 * - VS Code → Client: Pushes raw .py or .ifc file.
 * - Client → VS Code: Pushes raw .py or .ifc file (opened as new unsaved document in VS Code).
 */

export interface HelloMessage {
  type: "hello";
  version: string;
  capabilities: string[];
  source?: string;
}

export interface PushMessage {
  type: "ifc.push" | "file.push";
  version: string;
  id: string;
  direction: "to-blender" | "to-client" | "to-vscode";
  filename: string;
  fileType?: "python" | "ifc";
  content?: string;
  sizeBytes: number;
  batchId?: string;
  clearExisting?: boolean;
  overwrite?: boolean;
  command?: string;
}

export interface StatusMessage {
  type: "status";
  version: string;
  id: string;
  status: string;
  direction?: string;
  capabilities?: string[];
  /** 1.2: true only when the client is owed the batch named below. */
  wantsModel?: boolean;
  /** 1.2: the `batchId` of the push whose `/delete:all` the client ran. */
  awaitingBatchId?: string;
}

export interface ErrorMessage {
  type: "error";
  version: string;
  id: string | null;
  /** What the Ginnung page sends (local-bridge 1.4). */
  error?: string;
  /** What older tools sent. */
  message?: string;
}

export type ProtocolMessage =
  | HelloMessage
  | PushMessage
  | StatusMessage
  | ErrorMessage;

const PROTOCOL_VERSION = "1.4";

let _counter = 0;
function nextId(): string {
  return `vsc-${Date.now()}-${++_counter}`;
}

export function buildHello(): string {
  const msg: HelloMessage = {
    type: "hello",
    version: PROTOCOL_VERSION,
    capabilities: ["file.push", "ifc.push"],
    source: "vscode",
  };
  return JSON.stringify(msg);
}

export function buildPush(
  id: string,
  direction: PushMessage["direction"],
  filename: string,
  sizeBytes: number,
  content?: string,
  options?: {
    fileType?: "python" | "ifc";
    batchId?: string;
    clearExisting?: boolean;
    overwrite?: boolean;
    command?: string;
  }
): string {
  const fileType =
    options?.fileType || (filename.endsWith(".py") ? "python" : "ifc");
  const msg: PushMessage = {
    type: "file.push",
    version: PROTOCOL_VERSION,
    id,
    direction,
    filename,
    fileType,
    content,
    sizeBytes,
    batchId: options?.batchId,
    clearExisting: options?.clearExisting,
    overwrite: options?.overwrite,
    command: options?.command,
  };
  return JSON.stringify(msg);
}

export function buildStatus(id: string, status: string): string {
  const msg: StatusMessage = {
    type: "status",
    version: PROTOCOL_VERSION,
    id,
    status,
  };
  return JSON.stringify(msg);
}

export function buildError(id: string | null, message: string): string {
  const msg: ErrorMessage = {
    type: "error",
    version: PROTOCOL_VERSION,
    id,
    message,
  };
  return JSON.stringify(msg);
}

export function parseMessage(raw: string): ProtocolMessage {
  const data = JSON.parse(raw);
  if (!data.type) {
    throw new Error("Missing 'type' field in message");
  }
  return data as ProtocolMessage;
}
