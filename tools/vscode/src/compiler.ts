/**
 * Lite-STEP compiler integration.
 *
 * Invokes Python from the `lite-step` repository root so all
 * `from lite_step...` imports resolve relative to the repository.
 */

import { execFile } from "child_process";
import * as path from "path";
import * as os from "os";
import * as fs from "fs";
import * as vscode from "vscode";

export interface CompileResult {
  success: boolean;
  ifcPath: string;
  stderr: string;
}

/**
 * Find the root directory of the local lite-step repository.
 */
export function findLiteStepRepo(sourceFilePath?: string): string {
  const config = vscode.workspace.getConfiguration("ginnung");
  const configuredPath = config.get<string>("liteStepPath");
  if (configuredPath && fs.existsSync(configuredPath)) {
    return configuredPath;
  }

  // 1. Check workspace folders
  const workspaceFolders = vscode.workspace.workspaceFolders || [];
  for (const wf of workspaceFolders) {
    const p = wf.uri.fsPath;
    if (fs.existsSync(path.join(p, "lite_step"))) {
      return p;
    }
  }

  // 2. Check source file directory and its parents
  if (sourceFilePath) {
    let dir = path.dirname(sourceFilePath);
    for (let i = 0; i < 5; i++) {
      if (fs.existsSync(path.join(dir, "lite_step"))) {
        return dir;
      }
      const parent = path.dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
  }

  // 3. Check standard ~/projects/lite-step location
  const candidate = path.join(os.homedir(), "projects", "lite-step");
  if (fs.existsSync(candidate)) {
    return candidate;
  }

  return sourceFilePath ? path.dirname(sourceFilePath) : process.cwd();
}

/**
 * Compile a Lite-STEP Python file to IFC using the lite-step repo's compiler.
 *
 * @param filePath - Absolute path to the .py file
 * @returns CompileResult with the path to the generated .ifc file
 */
export async function compileLiteStep(
  filePath: string
): Promise<CompileResult> {
  const repoRoot = findLiteStepRepo(filePath);
  const dir = path.dirname(filePath);
  const baseName = path.basename(filePath, ".py");

  const env = {
    ...process.env,
    PYTHONPATH: [repoRoot, process.env.PYTHONPATH]
      .filter(Boolean)
      .join(path.delimiter),
  };

  return new Promise<CompileResult>((resolve) => {
    // Run the Python model file directly: python <filePath>
    execFile(
      "python",
      [filePath],
      {
        cwd: dir,
        env,
        timeout: 300_000,
        maxBuffer: 50 * 1024 * 1024,
      },
      (error, stdout, stderr) => {
        // Look for generated IFC file next to the source script
        const candidates = [
          path.join(dir, "output.ifc"),
          path.join(dir, `${baseName}.ifc`),
          path.join(repoRoot, "output.ifc"),
          path.join(os.tmpdir(), "output.ifc"),
          path.join(os.tmpdir(), `${baseName}.ifc`),
        ];

        for (const candidate of candidates) {
          if (fs.existsSync(candidate)) {
            return resolve({
              success: true,
              ifcPath: candidate,
              stderr: stderr || stdout || "",
            });
          }
        }

        if (error) {
          return resolve({
            success: false,
            ifcPath: "",
            stderr: stderr || stdout || error.message,
          });
        }

        resolve({
          success: false,
          ifcPath: "",
          stderr: stderr || stdout || "Compilation completed but no .ifc output file was found.",
        });
      }
    );
  });
}

/**
 * Read an IFC file into a Buffer.
 */
export function readIfcFile(filePath: string): Buffer {
  return fs.readFileSync(filePath);
}
