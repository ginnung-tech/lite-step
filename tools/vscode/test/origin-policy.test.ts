import { describe,  test } from "node:test";
import assert from "node:assert/strict";
import { allowedFile, checkUpgrade, entryProblem, parseAllowlist } from "../src/origin-policy";

const PORT = 9877;
const builtIn = parseAllowlist([]);

describe("checkUpgrade", () => {
  test("the built-in origin is a web client, by exact match", () => {
    assert.equal(checkUpgrade(`127.0.0.1:${PORT}`, "https://ginnung.tech", PORT, builtIn), "web");
    assert.equal(checkUpgrade(`localhost:${PORT}`, "https://ginnung.tech", PORT, builtIn), "web");
  });

  test("no Origin is a program on this computer", () => {
    assert.equal(checkUpgrade(`127.0.0.1:${PORT}`, undefined, PORT, builtIn), "local");
  });

  test("look-alike and unlisted origins are refused", () => {
    for (const origin of [
      "https://ginnung.tech.evil.example",
      "https://evilginnung.tech",
      "https://www.ginnung.tech",
      "http://ginnung.tech",
      "https://ginnung.tech:8443",
      "https://GINNUNG.tech",
      "https://ginnung.tech/",
      "http://localhost:5173",
      "http://localhost:5173.evil.example",
      "vscode-webview://abc",
      "null",
      "https://example.com",
    ]) {
      assert.equal(checkUpgrade(`127.0.0.1:${PORT}`, origin, PORT, builtIn), null, origin);
    }
  });

  test("a Host that is not this loopback port is refused, whatever the Origin", () => {
    for (const host of [undefined, "", "evil.example", `evil.example:${PORT}`, "127.0.0.1:9999", "127.0.0.1", `192.168.1.5:${PORT}`]) {
      assert.equal(checkUpgrade(host, "https://ginnung.tech", PORT, builtIn), null, String(host));
      assert.equal(checkUpgrade(host, undefined, PORT, builtIn), null, String(host));
    }
  });

  test("a user entry is allowed, and only that exact origin", () => {
    const list = parseAllowlist(["https://app.example.com", "http://localhost:5173"]);
    assert.deepEqual(list.rejected, []);
    assert.equal(checkUpgrade(`127.0.0.1:${PORT}`, "https://app.example.com", PORT, list), "web");
    assert.equal(checkUpgrade(`127.0.0.1:${PORT}`, "http://localhost:5173", PORT, list), "web");
    assert.equal(checkUpgrade(`127.0.0.1:${PORT}`, "http://localhost:5174", PORT, list), null);
    assert.equal(checkUpgrade(`127.0.0.1:${PORT}`, "https://app.example.com.evil.example", PORT, list), null);
  });
});

describe("allowlist entries", () => {
  test("https origins and loopback http origins are accepted", () => {
    for (const ok of ["https://app.example.com", "https://app.example.com:8443", "http://localhost:5173", "http://127.0.0.1:5180"]) {
      assert.equal(entryProblem(ok), null, ok);
    }
  });

  test("everything else is ignored, with a reason, and never widens the list", () => {
    const bad = [
      "http://app.example.com",
      "https://*.example.com",
      "https://app.example.com/",
      "https://app.example.com/path",
      "https://APP.example.com",
      " https://app.example.com",
      "app.example.com",
      "vscode-webview://",
      "*",
      "",
      42,
      null,
    ];
    const list = parseAllowlist(bad);
    assert.equal(list.rejected.length, bad.length);
    assert.deepEqual([...list.origins], ["https://ginnung.tech"]);
  });
});

describe("allowedFile", () => {
  test("only .py and .ifc, and fileType must agree", () => {
    assert.equal(allowedFile("output.py"), true);
    assert.equal(allowedFile("Model.IFC", "ifc"), true);
    assert.equal(allowedFile("output.py", "python"), true);
    assert.equal(allowedFile("output.py", "ifc"), false);
    for (const name of ["run.bat", "settings.json", "output.py.exe", "model.blend", "", ".bashrc"]) {
      assert.equal(allowedFile(name), false, name);
    }
  });
});
