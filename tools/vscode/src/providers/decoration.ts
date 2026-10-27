/**
 * File decoration provider for Lite-STEP files.
 *
 * Adds a green "3D" badge to Python files detected as Lite-STEP models
 * in the VS Code explorer sidebar.
 */

import * as vscode from "vscode";
import * as fs from "fs";
import { isLiteStepFile, clearCache } from "../detection";

export class GinnungDecorationProvider
  implements vscode.FileDecorationProvider
{
  private _onDidChangeFileDecorations =
    new vscode.EventEmitter<vscode.Uri | vscode.Uri[] | undefined>();
  public readonly onDidChangeFileDecorations =
    this._onDidChangeFileDecorations.event;

  public refresh(): void {
    clearCache();
    this._onDidChangeFileDecorations.fire(undefined);
  }

  provideFileDecoration(
    uri: vscode.Uri
  ): vscode.FileDecoration | undefined {
    if (!uri.fsPath.endsWith(".py")) {
      return undefined;
    }

    try {
      const content = fs.readFileSync(uri.fsPath, "utf-8");
      if (isLiteStepFile(content, uri.fsPath)) {
        return new vscode.FileDecoration(
          "3D",
          "Lite-STEP Model",
          new vscode.ThemeColor("charts.green")
        );
      }
    } catch {
      // File may not be readable
    }

    return undefined;
  }
}
