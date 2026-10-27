/**
 * CodeLens provider for Lite-STEP files.
 *
 * Adds "▶ Preview Model" and "▶ Push to Web" lenses at the top of
 * detected Lite-STEP Python files and IFC files.
 */

import * as vscode from "vscode";
import { isLiteStepFile } from "../detection";

export class GinnungCodeLensProvider implements vscode.CodeLensProvider {
  private readonly _onDidChangeCodeLenses = new vscode.EventEmitter<void>();
  public readonly onDidChangeCodeLenses = this._onDidChangeCodeLenses.event;

  public refresh(): void {
    this._onDidChangeCodeLenses.fire();
  }

  provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
    const lenses: vscode.CodeLens[] = [];
    const range = new vscode.Range(0, 0, 0, 0);

    if (document.languageId === "python") {
      if (!isLiteStepFile(document.getText(), document.uri.fsPath)) {
        return [];
      }
    }

    // Only for .py (LiteSTEP) and .ifc files
    const ext = document.uri.fsPath.toLowerCase();
    if (!ext.endsWith(".py") && !ext.endsWith(".ifc")) {
      return [];
    }

    lenses.push(
      new vscode.CodeLens(range, {
        title: "$(eye) Preview Model",
        command: "ginnung.previewFile",
        tooltip: "Compile and preview this model in 3D",
      }),
      new vscode.CodeLens(range, {
        title: "$(cloud-upload) Push to Web",
        command: "ginnung.pushToWeb",
        tooltip: "Push this model to connected Ginnung web clients",
      }),
      new vscode.CodeLens(range, {
        title: "$(symbol-misc) Push to Blender",
        command: "ginnung.pushToBlender",
        tooltip: "Push this model to Blender (ws://127.0.0.1:9876)",
      })
    );

    return lenses;
  }
}
