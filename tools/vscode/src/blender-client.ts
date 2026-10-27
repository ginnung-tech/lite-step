/**
 * Client for pushing IFC models directly from VS Code to Blender's WebSocket server.
 */

import { WebSocket } from "ws";

export interface BlenderPushResult {
  success: boolean;
  message: string;
}

/**
 * Send an IFC file (raw buffer) to the running Blender Lite-STEP server.
 *
 * @param ifcBytes - Buffer of the IFC model
 * @param filename - Name of the IFC file (e.g. model.ifc)
 * @param port - WebSocket port of the Blender server (default: 9876)
 * @param timeoutMs - Timeout in milliseconds (default: 10000)
 */
export async function pushIfcToBlender(
  ifcBytes: Buffer,
  filename: string,
  port: number = 9876,
  timeoutMs: number = 10000,
  blendPath?: string
): Promise<BlenderPushResult> {
  return new Promise((resolve) => {
    let resolved = false;
    let ws: WebSocket;

    const timeout = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        try {
          if (ws) ws.close();
        } catch {
          // ignore
        }
        resolve({
          success: false,
          message: `Timed out connecting to Blender on ws://127.0.0.1:${port}. Make sure Blender is running and the Lite-STEP server is started.`,
        });
      }
    }, timeoutMs);

    try {
      ws = new WebSocket(`ws://127.0.0.1:${port}`);
    } catch (e: any) {
      clearTimeout(timeout);
      return resolve({
        success: false,
        message: `Failed to initialize connection to Blender: ${e.message}`,
      });
    }

    ws.on("open", () => {
      const id = `vsc-blender-${Date.now()}`;
      const controlMsg = JSON.stringify({
        type: "ifc.push",
        version: "1.1",
        id,
        direction: "to-blender",
        filename,
        blendPath,
        sizeBytes: ifcBytes.length,
      });

      try {
        ws.send(controlMsg);
        ws.send(ifcBytes);
      } catch (e: any) {
        if (!resolved) {
          resolved = true;
          clearTimeout(timeout);
          try { ws.close(); } catch {}
          resolve({
            success: false,
            message: `Failed sending data to Blender: ${e.message}`,
          });
        }
      }
    });

    ws.on("message", (data) => {
      try {
        const msg = JSON.parse(data.toString());
        if (msg.type === "status" && msg.status === "loaded") {
          if (!resolved) {
            resolved = true;
            clearTimeout(timeout);
            try { ws.close(); } catch {}
            resolve({
              success: true,
              message: `Model '${filename}' successfully loaded in Blender.`,
            });
          }
        } else if (msg.type === "error") {
          if (!resolved) {
            resolved = true;
            clearTimeout(timeout);
            try { ws.close(); } catch {}
            resolve({
              success: false,
              message: `Blender error: ${msg.error || msg.message}`,
            });
          }
        }
      } catch {
        // ignore non-JSON messages (e.g. binary echoes)
      }
    });

    ws.on("error", (err) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        resolve({
          success: false,
          message: `Could not connect to Blender (ws://127.0.0.1:${port}): ${err.message}. Please make sure Blender is open and the Lite-STEP server is started.`,
        });
      }
    });

    ws.on("close", () => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        resolve({
          success: true,
          message: `Sent '${filename}' to Blender.`,
        });
      }
    });
  });
}

/**
 * Probes whether Blender's Lite-STEP WebSocket server is reachable and online.
 *
 * @param port - Port to check (default: 9876)
 * @param timeoutMs - Probe timeout (default: 1200ms)
 * @returns true if Blender server is responding, false otherwise
 */
export async function probeBlenderConnection(
  port: number = 9876,
  timeoutMs: number = 1200
): Promise<boolean> {
  return new Promise((resolve) => {
    let resolved = false;
    let ws: WebSocket;

    const timer = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        try { if (ws) ws.close(); } catch {}
        resolve(false);
      }
    }, timeoutMs);

    try {
      ws = new WebSocket(`ws://127.0.0.1:${port}`);
    } catch {
      clearTimeout(timer);
      return resolve(false);
    }

    ws.on("open", () => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timer);
        try { ws.close(); } catch {}
        resolve(true);
      }
    });

    ws.on("error", () => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timer);
        resolve(false);
      }
    });
  });
}
