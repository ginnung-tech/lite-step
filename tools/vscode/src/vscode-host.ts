/** The server's host inside VS Code: settings, messages and the status bar. */
import * as vscode from "vscode";
import type { ServerHost } from "./server";

export function vscodeHost(statusBarItem: vscode.StatusBarItem): ServerHost {
  return {
    allowedOrigins: () =>
      vscode.workspace.getConfiguration("litestep").get<unknown[]>("allowedOrigins", []),
    warn: (message) => {
      void vscode.window.showWarningMessage(message);
    },
    error: (message) => {
      void vscode.window.showErrorMessage(message);
    },
    showStatus: (text, tooltip) => {
      statusBarItem.text = text;
      statusBarItem.tooltip = tooltip;
      statusBarItem.show();
    },
  };
}
