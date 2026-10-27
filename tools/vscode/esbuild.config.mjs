import * as esbuild from "esbuild";
import * as fs from "fs";
import * as path from "path";

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

async function main() {
  copyVendorAssets();
  if (isWatch) {
    const extCtx = await esbuild.context(extensionConfig);
    const webCtx = await esbuild.context(webviewConfig);
    await extCtx.watch();
    await webCtx.watch();
    console.log("[esbuild] watching...");
  } else {
    await esbuild.build(extensionConfig);
    await esbuild.build(webviewConfig);
    console.log("[esbuild] build complete.");
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
