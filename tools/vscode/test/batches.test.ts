import { describe, test } from "node:test";
import assert from "node:assert/strict";
import { OutstandingBatches, type ModelPair } from "../src/batches";

const pair = (tag: string): ModelPair => ({
  pyContent: `# ${tag}`, pyFilename: "output.py", ifcContent: `ISO ${tag}`, ifcFilename: "output.ifc",
});

describe("outstanding /delete:all batches", () => {
  test("each page that received the wipe gets the pair once, then the batch is gone", () => {
    const batches = new OutstandingBatches();
    batches.add("X", pair("x"), 2);
    assert.deepEqual(batches.claim("X"), pair("x"));
    assert.deepEqual(batches.claim("X"), pair("x"));
    assert.equal(batches.claim("X"), null);
    assert.equal(batches.size, 0);
  });

  test("a later push keeps the earlier batch and its own pair", () => {
    const batches = new OutstandingBatches();
    batches.add("X", pair("x"), 1);
    batches.add("Y", pair("y"), 1);
    assert.deepEqual(batches.claim("X"), pair("x"));
    assert.deepEqual(batches.claim("Y"), pair("y"));
  });

  test("an unknown or missing id gets nothing and uses nothing up", () => {
    const batches = new OutstandingBatches();
    batches.add("X", pair("x"), 1);
    for (const id of ["Y", "", undefined, null, 7]) {
      assert.equal(batches.claim(id), null);
    }
    assert.deepEqual(batches.claim("X"), pair("x"));
  });

  test("a batch nobody asks for in time is dropped", () => {
    let now = 1_000;
    const batches = new OutstandingBatches(100, () => now);
    batches.add("X", pair("x"), 1);
    now += 101;
    assert.equal(batches.claim("X"), null);
    assert.equal(batches.size, 0);
  });

  test("a batch sent to no page is not kept", () => {
    const batches = new OutstandingBatches();
    batches.add("X", pair("x"), 0);
    assert.equal(batches.size, 0);
  });
});
