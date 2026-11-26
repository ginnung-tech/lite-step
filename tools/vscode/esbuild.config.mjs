import * as esbuild from "esbuild";
import * as fs from "node:fs";
import * as path from "node:path";

const isWatch = process.argv.includes("--watch");

function copyVendorAssets() {
  const srcVendor = path.resolve("src", "webview", "vendor");
  const outVendor = path.resolve("out", "webview", "vendor");
  if (fs.existsSync(srcVendor)) {
    fs.mkdirSync(outVendor, { recursive: true });
    fs.cpSync(srcVendor, outVendor, { recursive: true });
    console.log("[esbuild] copied vendor assets to out/webview/vendor");
  }
}

/** @type {esbuild.BuildOptions} */
const extensionConfig = {
  entryPoints: ["src/extension.ts"],
  bundle: true,
  outfile: "out/extension.js",
  external: ["vscode"],
  format: "cjs",
  platform: "node",
  target: "node20",
  sourcemap: true,
  minify: !isWatch,
};

/** @type {esbuild.BuildOptions} */
const webviewConfig = {
  entryPoints: ["src/webview/render-app.ts"],
  bundle: true,
  outfile: "out/webview/render-app.js",
  format: "esm",
  platform: "browser",
  target: "es2022",
  sourcemap: true,
  minify: !isWatch,
};

/** @type {esbuild.BuildOptions} */
const credentialHelperConfig = {
  entryPoints: ["src/git-credential-helper.ts"],
  bundle: true,
  outfile: "out/git-credential-ginnung.js",
  format: "cjs",
  platform: "node",
  target: "node20",
  sourcemap: true,
  minify: !isWatch,
  banner: {
    js: "#!/usr/bin/env node",
  },
};

copyVendorAssets();
if (isWatch) {
  const extCtx = await esbuild.context(extensionConfig);
  const webCtx = await esbuild.context(webviewConfig);
  const helperCtx = await esbuild.context(credentialHelperConfig);
  await extCtx.watch();
  await webCtx.watch();
  await helperCtx.watch();
  console.log("[esbuild] watching...");
} else {
  await esbuild.build(extensionConfig);
  await esbuild.build(webviewConfig);
  await esbuild.build(credentialHelperConfig);
  if (process.platform !== "win32") {
    try {
      fs.chmodSync(path.resolve("out", "git-credential-ginnung.js"), 0o755);
    } catch {
      // ignore chmod failures
    }
  }
  console.log("[esbuild] build complete.");
}
