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
import { vscodeHost } from "./vscode-host";
import { compileLiteStep, readIfcFile, findLiteStepRepo } from "./compiler";
import { isLiteStepFile, clearCache } from "./detection";
import { PreviewPanel } from "./preview-panel";
import { pushIfcToBlender } from "./blender-client";
import { GitCredentialBroker } from "./git-credential-broker";
import {
  DEFAULT_API_BASE_URL,
  DEFAULT_REMOTE_NAME,
  type GinnungRemoteLink,
  cloneGinnungProject,
  configureGinnungRemote,
  ensureGitRepo,
  findGitRepoRoot,
  formatCredentialHelperCommand,
  gitPullFromRemote,
  gitPushToRemote,
  mintGitToken,
  readGinnungRemoteLink,
  unlinkGinnungRemote,
  validateApiBaseUrl,
  validateProjectId,
  validateRemoteName,
} from "./git-remote";
import { LiteStepCodeLensProvider } from "./providers/codelens";
import { LiteStepHoverProvider } from "./providers/hover";
import { LiteStepDecorationProvider } from "./providers/decoration";
import { LiteStepSidebarProvider } from "./providers/sidebar-tree";

const GIT_TOKEN_SECRET_KEY = "litestep.ginnungClientToken";

let server: LiteStepServer;
let gitBroker: GitCredentialBroker | undefined;
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
  server = new LiteStepServer(vscodeHost(statusBarItem), port);

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

  // Loopback credential broker for repo-local `git pull/push ginnung`
  gitBroker = new GitCredentialBroker({
    getClientToken: async () => context.secrets.get(GIT_TOKEN_SECRET_KEY),
    getApiBaseUrl: () => getConfiguredApiBaseUrl(),
    getAllowedProjects: () => getAllowedWorkspaceProjectIds(),
  });
  void gitBroker.start().catch((err) => {
    console.warn(`[Lite-STEP] Failed to start Ginnung Git credential broker: ${err}`);
  });

  // ── Providers ──────────────────────────────────────────

  sidebarProvider = new LiteStepSidebarProvider(server, async () => {
    const token = await context.secrets.get(GIT_TOKEN_SECRET_KEY);
    const link = await getWorkspaceGitLink();
    return { signedIn: Boolean(token?.trim()), link };
  });
  codeLensProvider = new LiteStepCodeLensProvider();
  const hoverProvider = new LiteStepHoverProvider();
  decorationProvider = new LiteStepDecorationProvider();

  // ── Commands & Subscriptions ───────────────────────────

  context.subscriptions.push(
    statusBarItem,
    sidebarProvider,
    {
      dispose: () => {
        void gitBroker?.stop();
      },
    },
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
      "litestep.pushToWebNewProject",
      async (uri?: vscode.Uri) => {
        if (!server.isRunning) {
          vscode.window.showWarningMessage("Start the Lite-STEP server first");
          return;
        }
        const filePath = resolveTargetFile(uri);
        if (!filePath) {
          vscode.window.showWarningMessage(
            "No .py or .ifc file selected (or output.py / output.ifc found)"
          );
          return;
        }
        const name = await vscode.window.showInputBox({
          title: "Push as New Project",
          prompt: "Name of the new Ginnung project",
          value: defaultProjectName(filePath),
          validateInput: (v) => {
            const t = v.trim();
            if (t.length === 0) return "A project needs a name";
            if (t.length > 80) return "At most 80 characters";
            return null;
          },
        });
        if (name === undefined) {
          return;
        }
        await pushFileToWeb(filePath, name.trim());
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
    vscode.commands.registerCommand("litestep.gitSignIn", async () => {
      const entered = await vscode.window.showInputBox({
        title: "Ginnung Git: Set Client Token",
        prompt: "Paste your Ginnung client token (with git:read / git:write scope)",
        password: true,
        ignoreFocusOut: true,
        validateInput: (val) => (val.trim().length > 0 ? null : "Token cannot be empty"),
      });
      if (entered === undefined) return;
      await context.secrets.store(GIT_TOKEN_SECRET_KEY, entered.trim());
      gitBroker?.clearCache();
      sidebarProvider.refresh();
      vscode.window.showInformationMessage("Ginnung Git: client token saved to IDE SecretStorage.");
    }),
    vscode.commands.registerCommand("litestep.gitSignOut", async () => {
      await context.secrets.delete(GIT_TOKEN_SECRET_KEY);
      gitBroker?.clearCache();
      sidebarProvider.refresh();
      vscode.window.showInformationMessage("Ginnung Git: client token cleared.");
    }),
    vscode.commands.registerCommand("litestep.gitLinkRemote", async () => {
      await linkWorkspaceToGinnung(context);
    }),
    vscode.commands.registerCommand("litestep.gitCloneProject", async () => {
      await cloneProjectFromGinnung(context);
    }),
    vscode.commands.registerCommand("litestep.gitPull", async () => {
      await syncWorkspaceWithGinnung("pull");
    }),
    vscode.commands.registerCommand("litestep.gitPush", async () => {
      await syncWorkspaceWithGinnung("push");
    }),
    vscode.commands.registerCommand("litestep.gitUnlinkRemote", async () => {
      await unlinkWorkspaceFromGinnung();
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
  if (gitBroker) {
    void gitBroker.stop();
    gitBroker = undefined;
  }
  console.log("[Lite-STEP] Extension deactivated");
}

// ── Helpers ────────────────────────────────────────────────

/** Shown when Push finds no page connected: nothing is queued (protocol 1.2). */
const NOT_CONNECTED =
  "Lite-STEP: no web app is connected, so nothing was pushed. Open the web app, choose VS Code in its connection menu, then push again.";

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

/** The folder's name, or the script's when it is generic (output.py). */
function defaultProjectName(filePath: string): string {
  const stem = path.basename(filePath, path.extname(filePath));
  return stem === "output" ? path.basename(path.dirname(filePath)) : stem;
}

/**
 * Send the model to the connected Ginnung page(s): replacing the current
 * project's model, or, with `newProjectName`, into a new project.
 */
function sendToWeb(pyContent: string, ifcContent: string, newProjectName?: string): void {
  if (newProjectName === undefined) {
    if (!server.pushModel(pyContent, ifcContent, "output.py", "output.ifc")) {
      vscode.window.showWarningMessage(NOT_CONNECTED);
      return;
    }
    vscode.window.showInformationMessage("Lite-STEP: sent 'output.py' and 'output.ifc' to Ginnung.");
    return;
  }
  const result = server.pushAsNewProject(pyContent, ifcContent, newProjectName, "output.py", "output.ifc");
  if (result === "not-connected") {
    vscode.window.showWarningMessage(NOT_CONNECTED);
  } else if (result === "unsupported") {
    vscode.window.showWarningMessage(
      "Lite-STEP: the connected Ginnung page cannot open projects. Update the page (reload it), or use Push Model to Web to replace the current project's model."
    );
  } else {
    vscode.window.showInformationMessage(
      `Lite-STEP: asked Ginnung to open project '${newProjectName}'; the model follows when it has loaded.`
    );
  }
}

async function pushFileToWeb(filePath: string, newProjectName?: string): Promise<void> {
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
        sendToWeb(pyContent, ifcContent, newProjectName);
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

    sendToWeb(pyContent, ifcContent, newProjectName);
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

// ── Ginnung Git remote helpers ─────────────────────────────

function getConfiguredApiBaseUrl(): string {
  const raw = vscode.workspace.getConfiguration("litestep").get<string>("apiBaseUrl", DEFAULT_API_BASE_URL);
  try {
    return validateApiBaseUrl(raw || DEFAULT_API_BASE_URL);
  } catch {
    return DEFAULT_API_BASE_URL;
  }
}

function getConfiguredRemoteName(): string {
  const raw = vscode.workspace.getConfiguration("litestep").get<string>("gitRemoteName", DEFAULT_REMOTE_NAME);
  try {
    return validateRemoteName(raw || DEFAULT_REMOTE_NAME);
  } catch {
    return DEFAULT_REMOTE_NAME;
  }
}

function resolveHelperCommand(context: vscode.ExtensionContext): string {
  const helperScriptPath = context.asAbsolutePath(path.join("out", "git-credential-ginnung.js"));
  const configuredNode = vscode.workspace.getConfiguration("litestep").get<string>("nodePath", "").trim();
  if (configuredNode) {
    return formatCredentialHelperCommand(configuredNode, helperScriptPath);
  }
  const fixedNodePaths =
    process.platform === "win32"
      ? [
          path.join(process.env.ProgramFiles || "C:\\Program Files", "nodejs", "node.exe"),
          path.join(process.env["ProgramFiles(x86)"] || "C:\\Program Files (x86)", "nodejs", "node.exe"),
        ]
      : ["/opt/homebrew/bin/node", "/usr/local/bin/node", "/usr/bin/node"];
  for (const candidate of fixedNodePaths) {
    if (fs.existsSync(candidate)) {
      return formatCredentialHelperCommand(candidate, helperScriptPath);
    }
  }
  return formatCredentialHelperCommand(process.execPath, helperScriptPath, {
    ELECTRON_RUN_AS_NODE: "1",
  });
}

async function ensureClientToken(context: vscode.ExtensionContext): Promise<string | undefined> {
  const existing = (await context.secrets.get(GIT_TOKEN_SECRET_KEY))?.trim();
  if (existing) return existing;

  const entered = await vscode.window.showInputBox({
    title: "Ginnung Git Sign-In",
    prompt: "Paste your Ginnung client token (with git:read / git:write scope) to link or clone projects",
    password: true,
    ignoreFocusOut: true,
    validateInput: (val) => (val.trim().length > 0 ? null : "Token cannot be empty"),
  });
  if (entered === undefined) return undefined;
  const cleaned = entered.trim();
  await context.secrets.store(GIT_TOKEN_SECRET_KEY, cleaned);
  gitBroker?.clearCache();
  sidebarProvider.refresh();
  return cleaned;
}

async function getWorkspaceGitLink(): Promise<GinnungRemoteLink | null> {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) return null;
  return readGinnungRemoteLink(folders[0].uri.fsPath);
}

async function getAllowedWorkspaceProjectIds(): Promise<ReadonlySet<string>> {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    return new Set();
  }
  const links = await Promise.all(
    folders.map((folder) => readGinnungRemoteLink(folder.uri.fsPath).catch(() => null))
  );
  const allowed = new Set<string>();
  for (const link of links) {
    if (link?.projectId) {
      allowed.add(link.projectId);
    }
  }
  return allowed;
}

async function promptProjectId(initialValue = "P0001"): Promise<string | undefined> {
  const entered = await vscode.window.showInputBox({
    title: "Ginnung Project ID",
    prompt: "Enter canonical project ID (e.g. P0001, P0002, or default)",
    value: initialValue,
    ignoreFocusOut: true,
    validateInput: (val) => {
      try {
        validateProjectId(val);
        return null;
      } catch (err) {
        return err instanceof Error ? err.message : "Invalid project ID";
      }
    },
  });
  return entered === undefined ? undefined : validateProjectId(entered);
}

async function linkWorkspaceToGinnung(context: vscode.ExtensionContext): Promise<void> {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    vscode.window.showWarningMessage("Open a workspace folder first before linking a Ginnung Git remote.");
    return;
  }
  const clientToken = await ensureClientToken(context);
  if (!clientToken) return;

  const existingLink = await getWorkspaceGitLink();
  const projectId = await promptProjectId(existingLink?.projectId || "P0001");
  if (!projectId) return;

  const remoteName = await vscode.window.showInputBox({
    title: "Git Remote Name",
    prompt: "Name of the Ginnung Git remote (leaves 'origin' untouched)",
    value: existingLink?.remoteName || getConfiguredRemoteName(),
    ignoreFocusOut: true,
    validateInput: (val) => {
      try {
        validateRemoteName(val);
        return null;
      } catch (err) {
        return err instanceof Error ? err.message : "Invalid remote name";
      }
    },
  });
  if (remoteName === undefined) return;

  const apiBaseUrl = getConfiguredApiBaseUrl();
  const helperCommand = resolveHelperCommand(context);

  try {
    const link = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `Linking remote '${remoteName}' to ${projectId}…`,
        cancellable: false,
      },
      async () => {
        const repoRoot = await ensureGitRepo(folders[0].uri.fsPath);
        const minted = await mintGitToken(apiBaseUrl, clientToken, {
          target: "project",
          projectId,
          scope: "write",
        });
        return configureGinnungRemote({
          repoRoot,
          remoteName,
          remoteUrl: minted.remote,
          projectId,
          apiBaseUrl,
          helperCommand,
          scope: "write",
        });
      }
    );
    sidebarProvider.refresh();
    vscode.window.showInformationMessage(
      `Ginnung Git: linked remote '${link.remoteName}' (${link.projectId}). Use 'git push ${link.remoteName} main' or 'Pull from Ginnung'.`
    );
  } catch (err) {
    vscode.window.showErrorMessage(
      `Ginnung Git link failed: ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

async function cloneProjectFromGinnung(context: vscode.ExtensionContext): Promise<void> {
  const clientToken = await ensureClientToken(context);
  if (!clientToken) return;

  const projectId = await promptProjectId("P0001");
  if (!projectId) return;

  const picked = await vscode.window.showOpenDialog({
    canSelectFolders: true,
    canSelectFiles: false,
    canSelectMany: false,
    openLabel: "Select parent folder for clone",
  });
  if (!picked || picked.length === 0) return;

  const targetDir = path.join(picked[0].fsPath, `ginnung-${projectId}`);
  const remoteName = getConfiguredRemoteName();
  const apiBaseUrl = getConfiguredApiBaseUrl();
  const helperCommand = resolveHelperCommand(context);

  try {
    const link = await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: `Cloning Ginnung ${projectId}…`,
        cancellable: false,
      },
      async () => {
        const minted = await mintGitToken(apiBaseUrl, clientToken, {
          target: "project",
          projectId,
          scope: "write",
        });
        return cloneGinnungProject({
          targetDir,
          remoteUrl: minted.remote,
          token: minted.token,
          remoteName,
          projectId,
          apiBaseUrl,
          helperCommand,
          scope: "write",
        });
      }
    );
    const openNow = await vscode.window.showInformationMessage(
      `Ginnung Git: cloned ${link.projectId} with remote '${link.remoteName}'. Open folder now?`,
      "Open Folder",
      "Stay Here"
    );
    if (openNow === "Open Folder") {
      await vscode.commands.executeCommand("vscode.openFolder", vscode.Uri.file(targetDir));
    }
  } catch (err) {
    vscode.window.showErrorMessage(
      `Ginnung Git clone failed: ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

async function syncWorkspaceWithGinnung(direction: "pull" | "push"): Promise<void> {
  const link = await getWorkspaceGitLink();
  if (!link) {
    vscode.window.showWarningMessage(
      "This workspace is not linked to a Ginnung Git remote. Run 'Ginnung Git: Link Remote to Workspace' first."
    );
    return;
  }

  try {
    await vscode.window.withProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title:
          direction === "pull"
            ? `Pulling from '${link.remoteName}'…`
            : `Pushing to '${link.remoteName}'…`,
        cancellable: false,
      },
      async () => {
        if (direction === "pull") {
          await gitPullFromRemote(link.repoRoot, link.remoteName);
        } else {
          await gitPushToRemote(link.repoRoot, link.remoteName);
        }
      }
    );
    vscode.window.showInformationMessage(
      direction === "pull"
        ? `Ginnung Git: pulled latest from '${link.remoteName}'.`
        : `Ginnung Git: pushed commits to '${link.remoteName}'.`
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (direction === "push" && /non-fast-forward|fetch first|rejected/i.test(msg)) {
      vscode.window.showErrorMessage(
        `Ginnung Git push rejected (remote '${link.remoteName}' has newer commits). Pull first, resolve any conflict, then push again.`
      );
      return;
    }
    vscode.window.showErrorMessage(`Ginnung Git ${direction} failed: ${msg}`);
  }
}

async function unlinkWorkspaceFromGinnung(): Promise<void> {
  const link = await getWorkspaceGitLink();
  const root = link?.repoRoot ?? (await findGitRepoRoot(getWorkspaceRoot()));
  if (!link || !root) {
    vscode.window.showInformationMessage("Workspace is not linked to a Ginnung Git remote.");
    return;
  }

  const choice = await vscode.window.showQuickPick(
    [
      { label: `Remove helper config and remote '${link.remoteName}'`, removeRemote: true },
      { label: "Remove helper config only (keep Git remote URL)", removeRemote: false },
    ],
    { title: "Unlink Ginnung Git Remote" }
  );
  if (!choice) return;

  await unlinkGinnungRemote(root, { removeRemote: choice.removeRemote });
  sidebarProvider.refresh();
  vscode.window.showInformationMessage(
    `Ginnung Git: unlinked '${link.remoteName}' (${choice.removeRemote ? "remote removed" : "remote URL kept"}).`
  );
}
