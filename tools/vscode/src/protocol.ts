/**
 * Wire protocol types and builders for the Lite-STEP WebSocket bridge.
 *
 * Exposes two-way file push/pull and document sync protocol:
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
}

export interface ErrorMessage {
  type: "error";
  version: string;
  id: string | null;
  message: string;
}

export interface DocumentSyncMessage {
  type: "document.sync";
  version: string;
  id: string;
  payload: {
    documentId: "lite_step" | "ifc_step_file";
    content: string;
    contentType: "text/x-python" | "application/x-step";
    metadata: {
      lastModified: number;
      source: string;
      fileName: string;
    };
  };
}

export type ProtocolMessage =
  | HelloMessage
  | PushMessage
  | StatusMessage
  | ErrorMessage
  | DocumentSyncMessage;

const PROTOCOL_VERSION = "1.1";

let _counter = 0;
function nextId(): string {
  return `vsc-${Date.now()}-${++_counter}`;
}

export function buildHello(): string {
  const msg: HelloMessage = {
    type: "hello",
    version: PROTOCOL_VERSION,
    capabilities: ["file.push", "ifc.push", "document.sync"],
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

export function buildDocumentSync(
  fileName: string,
  content: string,
  isIfc: boolean
): string {
  const id = nextId();
  const msg: DocumentSyncMessage = {
    type: "document.sync",
    version: PROTOCOL_VERSION,
    id,
    payload: {
      documentId: isIfc ? "ifc_step_file" : "lite_step",
      content,
      contentType: isIfc ? "application/x-step" : "text/x-python",
      metadata: {
        lastModified: Date.now(),
        source: "vscode",
        fileName,
      },
    },
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
