/**
 * The bridge's door: the upgrade check and the strict handshake.
 *
 * Kept apart from `server.ts` (which needs the `vscode` module) so both can
 * be run against a real socket in a test.
 */

import type { IncomingMessage } from "node:http";
import type { WebSocket } from "ws";
import { checkUpgrade, type Allowlist, type Role } from "./origin-policy";

/** How long a web client has to answer `hello` with `status: "connected"`. */
export const HANDSHAKE_TIMEOUT_MS = 5000;

/** WebSocket close code for "you broke the rules" (RFC 6455 policy violation). */
const POLICY_VIOLATION = 1008;

type VerifyCallback = (ok: boolean, code?: number, message?: string) => void;

/**
 * `verifyClient` for `ws`: refuse with a bare 403 before the socket exists.
 * A refused page learns nothing about what listens here.
 */
export function makeVerifyClient(allowlist: Allowlist, port: number) {
  return (info: { origin: string; req: IncomingMessage }, done: VerifyCallback): void => {
    const role = roleOf(info.req, port, allowlist);
    if (role === null) {
      done(false, 403, "");
    } else {
      done(true);
    }
  };
}

/** The role of an upgrade request, or null to refuse it. */
export function roleOf(req: IncomingMessage, port: number, allowlist: Allowlist): Role | null {
  const origin = req.headers.origin;
  return checkUpgrade(req.headers.host, typeof origin === "string" ? origin : undefined, port, allowlist);
}

interface ClientState {
  role: Role;
  origin: string | null;
  ready: boolean;
  timer: ReturnType<typeof setTimeout> | null;
}

/**
 * Per-connection handshake state.
 *
 * A `web` client must send `status: "connected"` within the timeout, and
 * nothing else before it; until then it receives no push. A `local` client
 * is ready at once: local programs are not the threat, and older ones push
 * on open without a handshake.
 */
export class ClientGate {
  private readonly _clients = new Map<WebSocket, ClientState>();

  constructor(private readonly _timeoutMs: number = HANDSHAKE_TIMEOUT_MS) {}

  /** Register a connection that passed the upgrade check. */
  admit(ws: WebSocket, role: Role, origin: string | null): void {
    const state: ClientState = { role, origin, ready: role === "local", timer: null };
    if (!state.ready) {
      state.timer = setTimeout(() => {
        if (!state.ready) {
          ws.close(POLICY_VIOLATION, "handshake timeout");
        }
      }, this._timeoutMs);
    }
    this._clients.set(ws, state);
  }

  /**
   * Whether a frame of this message type may be handled now. A frame that
   * may not closes the connection. `status: "connected"` completes the
   * handshake.
   */
  accept(ws: WebSocket, type: string | null, status?: string): boolean {
    const state = this._clients.get(ws);
    if (!state) {
      return false;
    }
    if (state.ready) {
      return true;
    }
    if (type === "status" && status === "connected") {
      state.ready = true;
      if (state.timer) {
        clearTimeout(state.timer);
        state.timer = null;
      }
      return true;
    }
    ws.close(POLICY_VIOLATION, "handshake first");
    return false;
  }

  isReady(ws: WebSocket): boolean {
    return this._clients.get(ws)?.ready === true;
  }

  roleFor(ws: WebSocket): Role | null {
    return this._clients.get(ws)?.role ?? null;
  }

  /** Origins of the connected web clients that completed the handshake. */
  webOrigins(): string[] {
    const out: string[] = [];
    for (const state of this._clients.values()) {
      if (state.ready && state.role === "web" && state.origin) {
        out.push(state.origin);
      }
    }
    return out;
  }

  release(ws: WebSocket): void {
    const state = this._clients.get(ws);
    if (state?.timer) {
      clearTimeout(state.timer);
    }
    this._clients.delete(ws);
  }

  clear(): void {
    for (const state of this._clients.values()) {
      if (state.timer) {
        clearTimeout(state.timer);
      }
    }
    this._clients.clear();
  }
}
