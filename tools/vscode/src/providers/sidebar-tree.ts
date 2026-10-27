/**
 * TreeDataProvider for the Ginnung Activity Bar view container.
 *
 * Provides live server administration and real-time status indication for:
 * 1. VS Code Local WebSocket Server (ws://127.0.0.1:9877)
 * 2. Connected Web App clients
 * 3. Blender connection status probe (ws://127.0.0.1:9876)
 * 4. Quick push/preview actions
 */

import * as vscode from "vscode";
import { GinnungServer } from "../server";
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

export class GinnungTreeItem extends vscode.TreeItem {
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

export class GinnungSidebarProvider
  implements vscode.TreeDataProvider<GinnungTreeItem>, vscode.Disposable
{
  private _onDidChangeTreeData: vscode.EventEmitter<
    GinnungTreeItem | undefined | null | void
  > = new vscode.EventEmitter<GinnungTreeItem | undefined | null | void>();
  public readonly onDidChangeTreeData: vscode.Event<
    GinnungTreeItem | undefined | null | void
  > = this._onDidChangeTreeData.event;

  private _blenderOnline: boolean = false;
  private _blenderChecking: boolean = false;
  private _probeTimer: ReturnType<typeof setInterval> | null = null;
  private _disposables: vscode.Disposable[] = [];

  constructor(private readonly server: GinnungServer) {
    // Listen to server status changes
    this._disposables.push(
      server.onDidChangeStatus(() => {
        this.refresh();
      })
    );

    // Initial probe & background periodic probe of Blender status (every 8 seconds)
    this.probeBlender();
    this._probeTimer = setInterval(() => {
      this.probeBlender();
    }, 8000);
  }

  public refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  public async probeBlender(): Promise<void> {
    if (this._blenderChecking) return;
    this._blenderChecking = true;

    try {
      const config = vscode.workspace.getConfiguration("ginnung");
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

  getTreeItem(element: GinnungTreeItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: GinnungTreeItem): Promise<GinnungTreeItem[]> {
    if (element) {
      return [];
    }

    const items: GinnungTreeItem[] = [];
    const config = vscode.workspace.getConfiguration("ginnung");
    const vscPort = this.server.port;
    const blenderPort = config.get<number>("blenderPort", 9876);

    // ── 1. VS Code Server Status ─────────────────────────────
    if (this.server.isRunning) {
      items.push(
        new GinnungTreeItem(
          "VS Code Server: Running",
          "server-status",
          vscode.TreeItemCollapsibleState.None,
          {
            description: `ws://127.0.0.1:${vscPort}`,
            tooltip: `VS Code Ginnung WebSocket Server is ACTIVE on port ${vscPort}\nConnected Web Clients: ${this.server.clientCount}`,
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
        new GinnungTreeItem(
          "VS Code Server: Stopped",
          "server-status",
          vscode.TreeItemCollapsibleState.None,
          {
            description: `Port ${vscPort}`,
            tooltip: `VS Code Ginnung WebSocket Server is STOPPED. Click to start.`,
            iconPath: new vscode.ThemeIcon(
              "circle-slash",
              new vscode.ThemeColor("testing.iconFailed")
            ),
            contextValue: "serverStopped",
            command: {
              command: "ginnung.startServer",
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
        new GinnungTreeItem(
          "Web App: Connected",
          "web-app-status",
          vscode.TreeItemCollapsibleState.None,
          {
            description: `${webClients} client${webClients > 1 ? "s" : ""} active`,
            tooltip: `Ginnung Web App is connected and ready for pushes!`,
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
        new GinnungTreeItem(
          "Web App: Disconnected",
          "web-app-status",
          vscode.TreeItemCollapsibleState.None,
          {
            description: this.server.isRunning
              ? "Listening on ws://127.0.0.1:" + vscPort
              : "Server off",
            tooltip:
              "No Ginnung web client currently connected. Open the web app and connect.",
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
        new GinnungTreeItem(
          "Blender: Connected",
          "blender-status",
          vscode.TreeItemCollapsibleState.None,
          {
            description: `ws://127.0.0.1:${blenderPort}`,
            tooltip: `Blender Ginnung server is ONLINE and ready to receive models!`,
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
        new GinnungTreeItem(
          "Blender: Disconnected",
          "blender-status",
          vscode.TreeItemCollapsibleState.None,
          {
            description: `Port ${blenderPort} unreachable`,
            tooltip: `Blender Ginnung server is OFFLINE. Make sure Blender is running and the Ginnung server is started.`,
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
      new GinnungTreeItem(
        "Push Model to Web App",
        "action-push-web",
        vscode.TreeItemCollapsibleState.None,
        {
          description: "Overwrites web scene (/delete:all)",
          tooltip: "Push active .py or .ifc model (or output.py / output.ifc) to connected Web App",
          iconPath: new vscode.ThemeIcon("cloud-upload"),
          command: {
            command: "ginnung.pushToWeb",
            title: "Push Model to Web App",
          },
        }
      )
    );

    items.push(
      new GinnungTreeItem(
        "Push Model to Blender",
        "action-push-blender",
        vscode.TreeItemCollapsibleState.None,
        {
          description: `To ws://127.0.0.1:${blenderPort}`,
          tooltip: "Push active .py or .ifc model directly to Blender",
          iconPath: new vscode.ThemeIcon("run-all"),
          command: {
            command: "ginnung.pushToBlender",
            title: "Push Model to Blender",
          },
        }
      )
    );

    items.push(
      new GinnungTreeItem(
        "Preview Model in 3D",
        "action-preview",
        vscode.TreeItemCollapsibleState.None,
        {
          description: "3D ThatOpen Viewer",
          tooltip: "Preview active .py or .ifc model in local 3D viewer",
          iconPath: new vscode.ThemeIcon("eye"),
          command: {
            command: "ginnung.previewFile",
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
