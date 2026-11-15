/**
 * `/delete:all` batches still owed to the pages that ran them (protocol 1.2).
 *
 * A wipe reloads the page, which destroys the push that carried it; the page
 * then asks for the batch by id. Each batch keeps its own model pair, so a
 * later push does not replace it, and the number of pages it was sent to: it
 * is given once per page that asks, and dropped when all have asked or after
 * the time limit. An unknown, used-up or expired id gets nothing.
 *
 * No `vscode` import: this file is tested on its own.
 */

export interface ModelPair {
  pyContent: string;
  pyFilename: string;
  ifcContent: string;
  ifcFilename: string;
}

/** How long a batch stays owed to the pages that ran its wipe. */
export const INFLIGHT_BATCH_MS = 120_000;

interface Outstanding {
  pair: ModelPair;
  at: number;
  /** Pages that received the wipe and have not asked for the pair yet. */
  remaining: number;
}

export class OutstandingBatches {
  private readonly _batches = new Map<string, Outstanding>();

  constructor(
    private readonly _ttlMs: number = INFLIGHT_BATCH_MS,
    private readonly _now: () => number = Date.now
  ) {}

  /** Record a batch sent to `recipients` pages. */
  add(id: string, pair: ModelPair, recipients: number): void {
    this.dropExpired();
    if (recipients > 0) {
      this._batches.set(id, { pair, at: this._now(), remaining: recipients });
    }
  }

  /** The pair of an outstanding batch, once per recipient, or null. */
  claim(id: unknown): ModelPair | null {
    this.dropExpired();
    if (typeof id !== "string") {
      return null;
    }
    const batch = this._batches.get(id);
    if (!batch) {
      return null;
    }
    batch.remaining -= 1;
    if (batch.remaining <= 0) {
      this._batches.delete(id);
    }
    return batch.pair;
  }

  /** Forget batches no page asked for in time; returns their ids. */
  dropExpired(): string[] {
    const now = this._now();
    const dropped: string[] = [];
    for (const [id, batch] of this._batches) {
      if (now - batch.at > this._ttlMs) {
        this._batches.delete(id);
        dropped.push(id);
      }
    }
    return dropped;
  }

  get size(): number {
    return this._batches.size;
  }
}
