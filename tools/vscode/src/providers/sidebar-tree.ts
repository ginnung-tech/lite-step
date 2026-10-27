/**
 * TreeDataProvider for the Lite-STEP Activity Bar view container.
 *
 * Provides live server administration and real-time status indication for:
 * 1. VS Code Local WebSocket Server (ws://127.0.0.1:9877)
 * 2. Connected Web clients
 * 3. Blender connection status probe (ws://127.0.0.1:9876)
 * 4. Quick push/preview actions
 */

import * as vscode from "vscode";
import { LiteStepServer } from "../server";
import { probeBlenderConnection } from "../blender-client";

export type TreeItemType =
  | "section"
  | "server-status"
  | "web-app-status"
  | "blender-status"
  | "action-push-web"
  | "action-push-blender"
  | "action-preview"
  | "action-start-server"
  | "action-stop-server";

export class LiteStepTreeItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly itemType: TreeItemType,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState = vscode.TreeItemCollapsibleState.None,
    options?: {
      description?: string;
      tooltip?: string;
      iconPath?: vscode.ThemeIcon;
      contextValue?: string;
      command?: vscode.Command;
    }
  ) {
    super(label, collapsibleState);
    if (options?.description) this.description = options.description;
    if (options?.tooltip) this.tooltip = options.tooltip;
    if (options?.iconPath) this.iconPath = options.iconPath;
    if (options?.contextValue) this.contextValue = options.contextValue;
    if (options?.command) this.command = options.command;
  }
}

type TreeDataChangeEvent = LiteStepTreeItem | undefined | null | void;

export class LiteStepSidebarProvider
  implements vscode.TreeDataProvider<LiteStepTreeItem>, vscode.Disposable
{
  private readonly _onDidChangeTreeData: vscode.EventEmitter<TreeDataChangeEvent> =
    new vscode.EventEmitter<TreeDataChangeEvent>();
  public readonly onDidChangeTreeData: vscode.Event<TreeDataChangeEvent> =
    this._onDidChangeTreeData.event;

  private _blenderOnline: boolean = false;
  private _blenderChecking: boolean = false;
  private _probeTimer: ReturnType<typeof setInterval> | null = null;
  private _disposables: vscode.Disposable[] = [];

  constructor(private readonly server: LiteStepServer) {
    // Listen to server status changes
    this._disposables.push(
      server.onDidChangeStatus(() => {
        this.refresh();
      })
    );

    // Periodic probe of Blender status (every 8 seconds)
    this._probeTimer = setInterval(() => {
      void this.probeBlender().catch(() => {});
    }, 8000);
    setTimeout(() => {
      void this.probeBlender().catch(() => {});
    }, 0);
  }

  public refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  public async probeBlender(): Promise<void> {
    if (this._blenderChecking) return;
    this._blenderChecking = true;

    try {
      const config = vscode.workspace.getConfiguration("litestep");
      const port = config.get<number>("blenderPort", 9876);
      const isOnline = await probeBlenderConnection(port, 1000);
      if (this._blenderOnline !== isOnline) {
        this._blenderOnline = isOnline;
        this.refresh();
      }
    } catch {
      this._blenderOnline = false;
    } finally {
      this._blenderChecking = false;
    }
  }

  getTreeItem(element: LiteStepTreeItem): vscode.TreeItem {
    return element;
  }

  getChildren(element?: LiteStepTreeItem): LiteStepTreeItem[] {
    if (element) {
      return [];
    }

    const items: LiteStepTreeItem[] = [];
    const config = vscode.workspace.getConfiguration("litestep");
    const vscPort = this.server.port;
    const blenderPort = config.get<number>("blenderPort", 9876);

    // ── 1. VS Code Server Status ─────────────────────────────
    if (this.server.isRunning) {
      items.push(
        new LiteStepTreeItem(
          "VS Code Server: Running",
          "server-status",
          vscode.TreeItemCollapsibleState.None,
          {
            description: `ws://127.0.0.1:${vscPort}`,
            tooltip: `VS Code Lite-STEP WebSocket Server is ACTIVE on port ${vscPort}\nConnected Web Clients: ${this.server.clientCount}`,
            iconPath: new vscode.ThemeIcon(
              "check",
              new vscode.ThemeColor("testing.iconPassed")
            ),
            contextValue: "serverRunning",
          }
        )
      );
    } else {
      items.push(
        new LiteStepTreeItem(
          "VS Code Server: Stopped",
          "server-status",
          vscode.TreeItemCollapsibleState.None,
          {
            description: `Port ${vscPort}`,
            tooltip: `VS Code Lite-STEP WebSocket Server is STOPPED. Click to start.`,
            iconPath: new vscode.ThemeIcon(
              "circle-slash",
              new vscode.ThemeColor("testing.iconFailed")
            ),
            contextValue: "serverStopped",
            command: {
              command: "litestep.startServer",
              title: "Start Server",
            },
          }
        )
      );
    }

    // ── 2. Web App Connection Indicator ───────────────────────
    const webClients = this.server.clientCount;
    if (this.server.isRunning && webClients > 0) {
      items.push(
        new LiteStepTreeItem(
          "Web App: Connected",
          "web-app-status",
          vscode.TreeItemCollapsibleState.None,
          {
            description: `${webClients} client${webClients > 1 ? "s" : ""} active`,
            tooltip: `Web Client is connected and ready for pushes!`,
            iconPath: new vscode.ThemeIcon(
              "globe",
              new vscode.ThemeColor("charts.green")
            ),
            contextValue: "webAppConnected",
          }
        )
      );
    } else {
      items.push(
        new LiteStepTreeItem(
          "Web App: Disconnected",
          "web-app-status",
          vscode.TreeItemCollapsibleState.None,
          {
            description: this.server.isRunning
              ? "Listening on ws://127.0.0.1:" + vscPort
              : "Server off",
            tooltip:
              "No web client currently connected. Open a client and connect.",
            iconPath: new vscode.ThemeIcon(
              "plug",
              new vscode.ThemeColor("disabledForeground")
            ),
            contextValue: "webAppDisconnected",
          }
        )
      );
    }

    // ── 3. Blender Connection Indicator ────────────────────────
    if (this._blenderOnline) {
      items.push(
        new LiteStepTreeItem(
          "Blender: Connected",
          "blender-status",
          vscode.TreeItemCollapsibleState.None,
          {
            description: `ws://127.0.0.1:${blenderPort}`,
            tooltip: `Blender Lite-STEP server is ONLINE and ready to receive models!`,
            iconPath: new vscode.ThemeIcon(
              "symbol-misc",
              new vscode.ThemeColor("charts.green")
            ),
            contextValue: "blenderOnline",
          }
        )
      );
    } else {
      items.push(
        new LiteStepTreeItem(
          "Blender: Disconnected",
          "blender-status",
          vscode.TreeItemCollapsibleState.None,
          {
            description: `Port ${blenderPort} unreachable`,
            tooltip: `Blender Lite-STEP server is OFFLINE. Make sure Blender is running and the Lite-STEP server is started.`,
            iconPath: new vscode.ThemeIcon(
              "symbol-misc",
              new vscode.ThemeColor("disabledForeground")
            ),
            contextValue: "blenderOffline",
          }
        )
      );
    }

    // ── 4. Quick Actions ─────────────────────────────────────
    items.push(
      new LiteStepTreeItem(
        "Push Model to Web App",
        "action-push-web",
        vscode.TreeItemCollapsibleState.None,
        {
          description: "Overwrites web scene (/delete:all)",
          tooltip: "Push active .py or .ifc model (or output.py / output.ifc) to connected Web App",
          iconPath: new vscode.ThemeIcon("cloud-upload"),
          command: {
            command: "litestep.pushToWeb",
            title: "Push Model to Web App",
          },
        }
      ),
      new LiteStepTreeItem(
        "Push Model to Blender",
        "action-push-blender",
        vscode.TreeItemCollapsibleState.None,
        {
          description: `To ws://127.0.0.1:${blenderPort}`,
          tooltip: "Push active .py or .ifc model directly to Blender",
          iconPath: new vscode.ThemeIcon("run-all"),
          command: {
            command: "litestep.pushToBlender",
            title: "Push Model to Blender",
          },
        }
      ),
      new LiteStepTreeItem(
        "Preview Model in 3D",
        "action-preview",
        vscode.TreeItemCollapsibleState.None,
        {
          description: "3D ThatOpen Viewer",
          tooltip: "Preview active .py or .ifc model in local 3D viewer",
          iconPath: new vscode.ThemeIcon("eye"),
          command: {
            command: "litestep.previewFile",
            title: "Preview Model",
          },
        }
      )
    );

    return items;
  }

  dispose(): void {
    if (this._probeTimer) {
      clearInterval(this._probeTimer);
      this._probeTimer = null;
    }
    for (const d of this._disposables) {
      d.dispose();
    }
    this._disposables = [];
  }
}
