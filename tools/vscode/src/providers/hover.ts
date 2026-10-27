/**
 * Hover provider for Lite-STEP files.
 *
 * Shows a tooltip with a clickable "Preview Model" link when hovering
 * over detected Lite-STEP Python files.
 */

import * as vscode from "vscode";
import { isLiteStepFile } from "../detection";

export class LiteStepHoverProvider implements vscode.HoverProvider {
  provideHover(
    document: vscode.TextDocument,
    _position: vscode.Position
  ): vscode.Hover | null {
    if (document.languageId !== "python") {
      return null;
    }

    if (!isLiteStepFile(document.getText(), document.uri.fsPath)) {
      return null;
    }

    const md = new vscode.MarkdownString(
      `**Lite-STEP Model** detected\n\n` +
      `[$(eye) Preview Model](command:litestep.previewFile "Compile and preview in 3D") · ` +
      `[$(cloud-upload) Push to Web](command:litestep.pushToWeb "Push to connected clients")`
    );
    md.isTrusted = true;
    md.supportThemeIcons = true;

    return new vscode.Hover(md);
  }
}
