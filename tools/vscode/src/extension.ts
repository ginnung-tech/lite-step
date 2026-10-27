/**
 * Ginnung VS Code Extension — main entry point.
 *
 * Provides:
 * 1. WebSocket server for push/pull sync (compatible with Blender add-on protocol)
 * 2. 3D Preview panel for rendering IFC files
 * 3. Lite-STEP compilation via the lite-step CLI
 * 4. Editor integration (CodeLens, Hover, File Decorations)
 */

import * as vscode from "vscode";
import * as fs from "fs";
import * as path from "path";

import { GinnungServer } from "./server";
import { compileLiteStep, readIfcFile, findLiteStepRepo } from "./compiler";
import { isLiteStepFile, clearCache } from "./detection";
import { buildDocumentSync } from "./protocol";
import { PreviewPanel } from "./preview-panel";
import { pushIfcToBlender } from "./blender-client";
import { GinnungCodeLensProvider } from "./providers/codelens";
import { GinnungHoverProvider } from "./providers/hover";
import { GinnungDecorationProvider } from "./providers/decoration";
import { GinnungSidebarProvider } from "./providers/sidebar-tree";

let server: GinnungServer;
let statusBarItem: vscode.StatusBarItem;
let codeLensProvider: GinnungCodeLensProvider;
let decorationProvider: GinnungDecorationProvider;
let sidebarProvider: GinnungSidebarProvider;

// ── Activation ─────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext) {
  console.log("[Ginnung] Extension activating…");

  // Status bar
  statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
    100
  );
  statusBarItem.command = "ginnung.startServer";
  context.subscriptions.push(statusBarItem);

  // Server
  const config = vscode.workspace.getConfiguration("ginnung");
  const port = config.get<number>("port", 9877);
  server = new GinnungServer(statusBarItem, port);

  // Handle incoming files pushed from Ginnung web app → save in workspace root under ginnung-<timestamp>
  server.onFileReceived = async (file) => {
    try {
      const filename = file.filename || "output.py";
      const workspaceRoot = getWorkspaceRoot();
      const folderName = getTimestampFolder();
      const targetDir = path.join(workspaceRoot, folderName);

      if (!fs.existsSync(targetDir)) {
        fs.mkdirSync(targetDir, { recursive: true });
      }

      const targetPath = path.join(targetDir, filename);
      fs.writeFileSync(targetPath, file.data);

      const doc = await vscode.workspace.openTextDocument(targetPath);
      await vscode.window.showTextDocument(doc, { preview: false });

      vscode.window.showInformationMessage(
        `Ginnung: Saved and opened '${filename}' in '${folderName}'.`
      );
    } catch (e) {
      vscode.window.showErrorMessage(`Failed to save pushed file in VS Code: ${e}`);
    }
  };

  // Auto-start by default
  if (config.get<boolean>("autoStartServer", true)) {
    server.start(port);
  }

  // ── Commands ───────────────────────────────────────────

  context.subscriptions.push(
    vscode.commands.registerCommand("ginnung.startServer", () => {
      if (server.isRunning) {
        vscode.window.showInformationMessage(
          `Ginnung server already running on port ${server.port}`
        );
        return;
      }
      const cfg = vscode.workspace.getConfiguration("ginnung");
      server.start(cfg.get<number>("port", 9877));
      vscode.window.showInformationMessage(
        `Ginnung server started on port ${server.port}`
      );
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ginnung.stopServer", () => {
      server.stop();
      vscode.window.showInformationMessage("Ginnung server stopped");
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "ginnung.previewFile",
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
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "ginnung.pushToWeb",
      async (uri?: vscode.Uri) => {
        if (!server.isRunning) {
          vscode.window.showWarningMessage(
            "Start the Ginnung server first"
          );
          return;
        }
        if (server.clientCount === 0) {
          vscode.window.showWarningMessage(
            "No Ginnung clients connected"
          );
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
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "ginnung.pushToBlender",
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
    )
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ginnung.refresh", () => {
      clearCache();
      codeLensProvider.refresh();
      decorationProvider.refresh();
      sidebarProvider.refresh();
      sidebarProvider.probeBlender();
      vscode.window.showInformationMessage(
        "Ginnung: Detection cache and connection status refreshed"
      );
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ginnung.refreshStatus", async () => {
      sidebarProvider.refresh();
      await sidebarProvider.probeBlender();
    })
  );

  // ── Providers ──────────────────────────────────────────

  sidebarProvider = new GinnungSidebarProvider(server);
  context.subscriptions.push(sidebarProvider);
  context.subscriptions.push(
    vscode.window.registerTreeDataProvider(
      "ginnung.statusView",
      sidebarProvider
    )
  );

  codeLensProvider = new GinnungCodeLensProvider();
  context.subscriptions.push(
    vscode.languages.registerCodeLensProvider(
      [
        { language: "python" },
        { pattern: "**/*.ifc" },
      ],
      codeLensProvider
    )
  );

  const hoverProvider = new GinnungHoverProvider();
  context.subscriptions.push(
    vscode.languages.registerHoverProvider(
      { language: "python" },
      hoverProvider
    )
  );

  decorationProvider = new GinnungDecorationProvider();
  context.subscriptions.push(
    vscode.window.registerFileDecorationProvider(decorationProvider)
  );

  // ── Auto-sync on save ──────────────────────────────────

  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument(async (doc) => {
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
    })
  );

  // ── Invalidate cache on file change ────────────────────

  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument((e) => {
      if (e.document.languageId === "python") {
        clearCache();
      }
    })
  );

  console.log("[Ginnung] Extension activated");
}

// ── Deactivation ───────────────────────────────────────────

export function deactivate() {
  if (server) {
    server.dispose();
  }
  console.log("[Ginnung] Extension deactivated");
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
  return `ginnung-${year}${month}${day}-${hours}${minutes}${seconds}`;
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
    const ifcData = readIfcFile(filePath);
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
          const ifcData = readIfcFile(result.ifcPath);
          PreviewPanel.show(extensionUri, ifcData, baseName);
        } else {
          vscode.window.showErrorMessage(
            `Lite-STEP compilation failed:\n${result.stderr}`
          );
        }
      }
    );
    return;
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

        server.pushModelWithWipe(pyContent, "output.py", ifcContent, "output.ifc");

        vscode.window.showInformationMessage(
          `Ginnung: Sent /delete:all to Web App. Pushing 'output.py' & 'output.ifc' upon reconnect.`
        );
      }
    );
    return;
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

    server.pushModelWithWipe(pyContent, "output.py", ifcContent, "output.ifc");

    vscode.window.showInformationMessage(
      `Ginnung: Sent /delete:all to Web App. Pushing 'output.py' & 'output.ifc' upon reconnect.`
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
  const config = vscode.workspace.getConfiguration("ginnung");
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
          vscode.window.showInformationMessage(`Ginnung: ${result.message}`);
        } else {
          vscode.window.showErrorMessage(`Ginnung: ${result.message}`);
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
          vscode.window.showInformationMessage(`Ginnung: ${result.message}`);
        } else {
          vscode.window.showErrorMessage(`Ginnung: ${result.message}`);
        }
      }
    );
  }
}
