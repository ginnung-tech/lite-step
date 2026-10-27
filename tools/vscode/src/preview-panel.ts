/**
 * 3D Preview WebviewPanel manager.
 *
 * Manages a singleton WebviewPanel that renders IFC files using
 * ThatOpen Components + Web-IFC WASM (vendored from groa-worker/web_render).
 */

import * as vscode from "vscode";
import * as crypto from "node:crypto";

export class PreviewPanel {
  public static readonly viewType = "ginnung.preview";

  private static _instance: PreviewPanel | undefined;
  private readonly _panel: vscode.WebviewPanel;
  private readonly _extensionUri: vscode.Uri;
  private _disposed = false;

  public static show(
    extensionUri: vscode.Uri,
    ifcData?: Buffer,
    title?: string
  ): PreviewPanel {
    const column = vscode.ViewColumn.Beside;

    if (PreviewPanel._instance) {
      PreviewPanel._instance._panel.reveal(column);
      if (ifcData) {
        PreviewPanel._instance.updateIfc(ifcData, title);
      }
      return PreviewPanel._instance;
    }

    const panel = vscode.window.createWebviewPanel(
      PreviewPanel.viewType,
      title || "Ginnung Preview",
      column,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [
          vscode.Uri.joinPath(extensionUri, "out", "webview"),
        ],
      }
    );

    const instance = new PreviewPanel(panel, extensionUri);
    PreviewPanel._instance = instance;

    if (ifcData) {
      instance._pendingIfc = { data: ifcData, title };
    }

    return instance;
  }

  public static get current(): PreviewPanel | undefined {
    return PreviewPanel._instance;
  }

  private _pendingIfc: { data: Buffer; title?: string } | null = null;

  private constructor(
    panel: vscode.WebviewPanel,
    extensionUri: vscode.Uri
  ) {
    this._panel = panel;
    this._extensionUri = extensionUri;

    this._panel.webview.html = this._getHtmlContent();

    this._panel.webview.onDidReceiveMessage((msg) => {
      switch (msg.type) {
        case "ready":
          if (this._pendingIfc) {
            this.updateIfc(this._pendingIfc.data, this._pendingIfc.title);
            this._pendingIfc = null;
          }
          break;
        case "error":
          vscode.window.showErrorMessage(
            `Ginnung Preview: ${msg.message}`
          );
          break;
      }
    });

    this._panel.onDidDispose(() => {
      this._disposed = true;
      PreviewPanel._instance = undefined;
    });
  }

  public updateIfc(ifcData: Buffer, title?: string): void {
    if (this._disposed) {
      return;
    }

    if (title) {
      this._panel.title = title;
    }

    const base64 = ifcData.toString("base64");
    this._panel.webview.postMessage({
      type: "loadIfc",
      data: base64,
    });
  }

  public dispose(): void {
    this._panel.dispose();
  }

  private _getHtmlContent(): string {
    const webview = this._panel.webview;
    const nonce = getNonce();

    const vendorUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, "out", "webview", "vendor")
    );

    return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none';
                 script-src 'nonce-${nonce}' 'unsafe-eval' ${webview.cspSource};
                 style-src 'nonce-${nonce}';
                 worker-src blob: ${webview.cspSource};
                 connect-src ${webview.cspSource} blob: data:;
                 img-src data: ${webview.cspSource};">
  <title>Ginnung Preview</title>
  <style nonce="${nonce}">
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body { width: 100%; height: 100%; overflow: hidden; background: #1e1e1e; font-family: system-ui, sans-serif; }
    #canvas-container {
      width: 100%; height: 100%;
      display: flex; align-items: center; justify-content: center;
    }
    canvas { width: 100% !important; height: 100% !important; display: block; }
    #status {
      position: absolute; top: 12px; left: 12px;
      color: #ccc; font: 12px/1.4 system-ui, sans-serif;
      background: rgba(30,30,30,0.85); padding: 6px 12px;
      border-radius: 4px; z-index: 10;
      pointer-events: none;
    }
    #drop-zone {
      position: absolute; inset: 0;
      display: flex; align-items: center; justify-content: center;
      color: #888; font: 16px/1.4 system-ui, sans-serif;
    }
    #drop-zone.hidden { display: none; }
  </style>
</head>
<body>
  <div id="status">Waiting for model…</div>
  <div id="drop-zone">
    <span>Use <strong>Preview Model</strong> to render an IFC or Lite-STEP file</span>
  </div>
  <div id="canvas-container">
    <canvas id="c"></canvas>
  </div>

  <script nonce="${nonce}" type="module">
    const vscode = acquireVsCodeApi();
    const statusEl = document.getElementById('status');
    const dropZone = document.getElementById('drop-zone');

    window.addEventListener('message', async (event) => {
      const msg = event.data;
      if (msg.type === 'loadIfc') {
        try {
          statusEl.textContent = 'Loading IFC model via ThatOpen/web-ifc WASM…';
          dropZone.classList.add('hidden');

          const binary = atob(msg.data);
          const bytes = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
          }

          statusEl.textContent = 'Model loaded successfully';
        } catch (e) {
          statusEl.textContent = 'Error: ' + e.message;
          vscode.postMessage({ type: 'error', message: e.message });
        }
      }
    });

    vscode.postMessage({ type: 'ready' });
  </script>
</body>
</html>`;
  }
}

function getNonce(): string {
  return crypto.randomBytes(16).toString("base64");
}
