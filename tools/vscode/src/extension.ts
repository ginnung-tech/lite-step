/**
 * Lite-STEP VS Code Extension — main entry point.
 *
 * Provides:
 * 1. WebSocket server for push/pull sync (compatible with Blender add-on protocol)
 * 2. 3D Preview panel for rendering IFC files
 * 3. Lite-STEP compilation via the lite-step CLI
 * 4. Editor integration (CodeLens, Hover, File Decorations)
 */

import * as vscode from "vscode";
import * as fs from "node:fs";
import * as path from "node:path";

import { LiteStepServer } from "./server";
import { compileLiteStep, readIfcFile, findLiteStepRepo } from "./compiler";
import { isLiteStepFile, clearCache } from "./detection";
import { buildDocumentSync } from "./protocol";
import { PreviewPanel } from "./preview-panel";
import { pushIfcToBlender } from "./blender-client";
import { LiteStepCodeLensProvider } from "./providers/codelens";
import { LiteStepHoverProvider } from "./providers/hover";
import { LiteStepDecorationProvider } from "./providers/decoration";
import { LiteStepSidebarProvider } from "./providers/sidebar-tree";

let server: LiteStepServer;
let statusBarItem: vscode.StatusBarItem;
let codeLensProvider: LiteStepCodeLensProvider;
let decorationProvider: LiteStepDecorationProvider;
let sidebarProvider: LiteStepSidebarProvider;

// ── Activation ─────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext) {
  console.log("[Lite-STEP] Extension activating…");

  // Status bar
  statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
    100
  );
  statusBarItem.command = "litestep.startServer";

  // Server
  const config = vscode.workspace.getConfiguration("litestep");
  const port = config.get<number>("port", 9877);
  server = new LiteStepServer(statusBarItem, port);

  // Handle incoming files pushed from web app → save in workspace root under litestep-<timestamp>
  server.onFileReceived = (file) => {
    void (async () => {
      try {
        const rawFilename = file.filename || "output.py";
        const filename = path.basename(rawFilename.replaceAll("\\", "/"));
        const workspaceRoot = getWorkspaceRoot();
        const folderName = getTimestampFolder();
        const targetDir = path.join(workspaceRoot, folderName);

        if (!fs.existsSync(targetDir)) {
          fs.mkdirSync(targetDir, { recursive: true });
        }

        const targetPath = path.join(targetDir, filename);
        const resolvedTarget = path.resolve(targetPath);
        const resolvedDir = path.resolve(targetDir);
        if (!resolvedTarget.startsWith(resolvedDir + path.sep) && resolvedTarget !== resolvedDir) {
          throw new Error(`Rejected unsafe filename '${file.filename}'`);
        }
        fs.writeFileSync(targetPath, file.data);

        const doc = await vscode.workspace.openTextDocument(targetPath);
        await vscode.window.showTextDocument(doc, { preview: false });

        vscode.window.showInformationMessage(
          `Lite-STEP: Saved and opened '${filename}' in '${folderName}'.`
        );
      } catch (e) {
        vscode.window.showErrorMessage(`Failed to save pushed file in VS Code: ${e}`);
      }
    })();
  };

  // Auto-start by default
  if (config.get<boolean>("autoStartServer", true)) {
    server.start(port);
  }

  // ── Providers ──────────────────────────────────────────

  sidebarProvider = new LiteStepSidebarProvider(server);
  codeLensProvider = new LiteStepCodeLensProvider();
  const hoverProvider = new LiteStepHoverProvider();
  decorationProvider = new LiteStepDecorationProvider();

  // ── Commands & Subscriptions ───────────────────────────

  context.subscriptions.push(
    statusBarItem,
    sidebarProvider,
    vscode.commands.registerCommand("litestep.startServer", () => {
      if (server.isRunning) {
        vscode.window.showInformationMessage(
          `Lite-STEP server already running on port ${server.port}`
        );
        return;
      }
      const cfg = vscode.workspace.getConfiguration("litestep");
      server.start(cfg.get<number>("port", 9877));
      vscode.window.showInformationMessage(
        `Lite-STEP server started on port ${server.port}`
      );
    }),
    vscode.commands.registerCommand("litestep.stopServer", () => {
      server.stop();
      vscode.window.showInformationMessage("Lite-STEP server stopped");
    }),
    vscode.commands.registerCommand(
      "litestep.previewFile",
      async (uri?: vscode.Uri) => {
        const filePath = resolveTargetFile(uri);
        if (!filePath) {
          vscode.window.showWarningMessage(
            "No .py or .ifc file selected (or output.py / output.ifc found)"
          );
          return;
        }
        await previewFile(filePath, context.extensionUri);
      }
    ),
    vscode.commands.registerCommand(
      "litestep.pushToWeb",
      async (uri?: vscode.Uri) => {
        if (!server.isRunning) {
          vscode.window.showWarningMessage("Start the Lite-STEP server first");
          return;
        }
        if (server.clientCount === 0) {
          vscode.window.showWarningMessage("No clients connected");
          return;
        }
        const filePath = resolveTargetFile(uri);
        if (!filePath) {
          vscode.window.showWarningMessage(
            "No .py or .ifc file selected (or output.py / output.ifc found)"
          );
          return;
        }
        await pushFileToWeb(filePath);
      }
    ),
    vscode.commands.registerCommand(
      "litestep.pushToBlender",
      async (uri?: vscode.Uri) => {
        const filePath = resolveTargetFile(uri);
        if (!filePath) {
          vscode.window.showWarningMessage(
            "No .py or .ifc file selected (or output.py / output.ifc found)"
          );
          return;
        }
        await pushFileToBlender(filePath);
      }
    ),
    vscode.commands.registerCommand("litestep.refresh", () => {
      clearCache();
      codeLensProvider.refresh();
      decorationProvider.refresh();
      sidebarProvider.refresh();
      void sidebarProvider.probeBlender().catch(() => {});
      vscode.window.showInformationMessage(
        "Lite-STEP: Detection cache and connection status refreshed"
      );
    }),
    vscode.commands.registerCommand("litestep.refreshStatus", async () => {
      sidebarProvider.refresh();
      await sidebarProvider.probeBlender();
    }),
    vscode.window.registerTreeDataProvider("litestep.statusView", sidebarProvider),
    vscode.languages.registerCodeLensProvider(
      [
        { language: "python" },
        { pattern: "**/*.ifc" },
      ],
      codeLensProvider
    ),
    vscode.languages.registerHoverProvider({ language: "python" }, hoverProvider),
    vscode.window.registerFileDecorationProvider(decorationProvider),
    vscode.workspace.onDidSaveTextDocument((doc) => {
      if (!server.isRunning || server.clientCount === 0) {
        return;
      }
      const fsPath = doc.uri.fsPath.toLowerCase();
      if (fsPath.endsWith(".ifc")) {
        const syncMsg = buildDocumentSync(
          path.basename(doc.uri.fsPath),
          doc.getText(),
          true
        );
        server.sendTextSync(syncMsg);
      } else if (
        fsPath.endsWith(".py") &&
        isLiteStepFile(doc.getText(), doc.uri.fsPath)
      ) {
        const syncMsg = buildDocumentSync(
          path.basename(doc.uri.fsPath),
          doc.getText(),
          false
        );
        server.sendTextSync(syncMsg);
      }
    }),
    vscode.workspace.onDidChangeTextDocument((e) => {
      if (e.document.languageId === "python") {
        clearCache();
      }
    })
  );

  console.log("[Lite-STEP] Extension activated");
}

// ── Deactivation ───────────────────────────────────────────

export function deactivate() {
  if (server) {
    server.dispose();
  }
  console.log("[Lite-STEP] Extension deactivated");
}

// ── Helpers ────────────────────────────────────────────────

function getWorkspaceRoot(): string {
  if (vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length > 0) {
    return vscode.workspace.workspaceFolders[0].uri.fsPath;
  }
  return process.cwd();
}

function getTimestampFolder(): string {
  const now = new Date();
  const pad = (n: number) => n.toString().padStart(2, "0");
  const year = now.getFullYear();
  const month = pad(now.getMonth() + 1);
  const day = pad(now.getDate());
  const hours = pad(now.getHours());
  const minutes = pad(now.getMinutes());
  const seconds = pad(now.getSeconds());
  return `litestep-${year}${month}${day}-${hours}${minutes}${seconds}`;
}

function resolveTargetFile(uri?: vscode.Uri): string | undefined {
  if (uri) {
    return uri.fsPath;
  }

  const editor = vscode.window.activeTextEditor;
  if (editor) {
    const fsPath = editor.document.uri.fsPath.toLowerCase();
    if (fsPath.endsWith(".py") || fsPath.endsWith(".ifc")) {
      return editor.document.uri.fsPath;
    }
  }

  // Fallback to checking workspace root for output.py or output.ifc
  const root = getWorkspaceRoot();
  const outputPy = path.join(root, "output.py");
  if (fs.existsSync(outputPy)) {
    return outputPy;
  }
  const outputIfc = path.join(root, "output.ifc");
  if (fs.existsSync(outputIfc)) {
    return outputIfc;
  }

  return undefined;
}

async function previewFile(
  filePath: string,
  extensionUri: vscode.Uri
): Promise<void> {
  const ext = path.extname(filePath).toLowerCase();
  const baseName = path.basename(filePath);

  if (ext === ".ifc") {
    const ifcData = await readIfcFile(filePath);
    PreviewPanel.show(extensionUri, ifcData, baseName);
    return;
  }

  if (ext === ".py") {
    // Compile Lite-STEP first for local 3D preview
    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `Compiling ${baseName}…`,
        cancellable: false,
      },
      async () => {
        const result = await compileLiteStep(filePath);
        if (result.success) {
          const ifcData = await readIfcFile(result.ifcPath);
          PreviewPanel.show(extensionUri, ifcData, baseName);
        } else {
          vscode.window.showErrorMessage(
            `Lite-STEP compilation failed:\n${result.stderr}`
          );
        }
      }
    );
  }
}

async function pushFileToWeb(filePath: string): Promise<void> {
  const ext = path.extname(filePath).toLowerCase();
  const baseName = path.basename(filePath);

  if (ext === ".py") {
    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `Compiling ${baseName} and pushing to Web App…`,
        cancellable: false,
      },
      async () => {
        const compResult = await compileLiteStep(filePath);
        if (!compResult.success) {
          vscode.window.showErrorMessage(
            `Lite-STEP compilation failed before pushing to Web App:\n${compResult.stderr}`
          );
          return;
        }

        const pyContent = fs.readFileSync(filePath, "utf-8");
        const ifcContent = fs.readFileSync(compResult.ifcPath, "utf-8");

        server.pushModelWithWipe(pyContent, ifcContent, "output.py", "output.ifc");

        vscode.window.showInformationMessage(
          `Lite-STEP: Sent /delete:all to Web App. Pushing 'output.py' & 'output.ifc' upon reconnect.`
        );
      }
    );
  }

  if (ext === ".ifc") {
    const ifcContent = fs.readFileSync(filePath, "utf-8");
    const pyCandidate = path.join(
      path.dirname(filePath),
      `${path.basename(filePath, ".ifc")}.py`
    );
    const workspacePy = path.join(getWorkspaceRoot(), "output.py");

    let pyContent = "";
    if (fs.existsSync(pyCandidate)) {
      pyContent = fs.readFileSync(pyCandidate, "utf-8");
    } else if (fs.existsSync(workspacePy)) {
      pyContent = fs.readFileSync(workspacePy, "utf-8");
    }

    server.pushModelWithWipe(pyContent, ifcContent, "output.py", "output.ifc");

    vscode.window.showInformationMessage(
      `Lite-STEP: Sent /delete:all to Web App. Pushing 'output.py' & 'output.ifc' upon reconnect.`
    );
  }
}

function findBlendPath(filePath: string): string | undefined {
  const repoRoot = findLiteStepRepo(filePath);
  const candidates = [
    path.join(path.dirname(filePath), "model.blend"),
    path.join(path.dirname(filePath), "..", "model.blend"),
    path.join(repoRoot, "lite_step", "assets", "model.blend"),
    path.join(repoRoot, "assets", "model.blend"),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) {
      return c;
    }
  }
  return undefined;
}

async function pushFileToBlender(filePath: string): Promise<void> {
  const config = vscode.workspace.getConfiguration("litestep");
  const port = config.get<number>("blenderPort", 9876);
  const ext = path.extname(filePath).toLowerCase();
  const baseName = path.basename(filePath);
  const blendPath = findBlendPath(filePath);

  if (ext === ".ifc") {
    const buffer = fs.readFileSync(filePath);
    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `Pushing ${baseName} to Blender…`,
        cancellable: false,
      },
      async () => {
        const result = await pushIfcToBlender(buffer, baseName, port, 10000, blendPath);
        if (result.success) {
          vscode.window.showInformationMessage(`Lite-STEP: ${result.message}`);
        } else {
          vscode.window.showErrorMessage(`Lite-STEP: ${result.message}`);
        }
      }
    );
    return;
  }

  if (ext === ".py") {
    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `Compiling ${baseName} and pushing to Blender…`,
        cancellable: false,
      },
      async () => {
        const compResult = await compileLiteStep(filePath);
        if (!compResult.success) {
          vscode.window.showErrorMessage(
            `Lite-STEP compilation failed before pushing to Blender:\n${compResult.stderr}`
          );
          return;
        }

        const ifcBuffer = fs.readFileSync(compResult.ifcPath);
        const ifcName = `${path.basename(filePath, ".py")}.ifc`;
        const result = await pushIfcToBlender(ifcBuffer, ifcName, port, 10000, blendPath);
        if (result.success) {
          vscode.window.showInformationMessage(`Lite-STEP: ${result.message}`);
        } else {
          vscode.window.showErrorMessage(`Lite-STEP: ${result.message}`);
        }
      }
    );
  }
}
