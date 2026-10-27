/**
 * Lite-STEP compiler integration.
 *
 * Invokes Python from the `lite-step` repository root so all
 * `from lite_step...` imports resolve relative to the repository.
 */

import { execFile } from "node:child_process";
import * as path from "node:path";
import * as os from "node:os";
import * as fs from "node:fs";
import * as vscode from "vscode";

export interface CompileResult {
  success: boolean;
  ifcPath: string;
  stderr: string;
}

function checkWorkspaceFolders(): string | null {
  const workspaceFolders = vscode.workspace.workspaceFolders || [];
  for (const wf of workspaceFolders) {
    const p = wf.uri.fsPath;
    if (fs.existsSync(path.join(p, "lite_step"))) {
      return p;
    }
  }
  return null;
}

function checkParentDirs(sourceFilePath: string): string | null {
  let dir = path.dirname(sourceFilePath);
  for (let i = 0; i < 5; i++) {
    if (fs.existsSync(path.join(dir, "lite_step"))) {
      return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return null;
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

  const fromWorkspace = checkWorkspaceFolders();
  if (fromWorkspace) return fromWorkspace;

  if (sourceFilePath) {
    const fromParents = checkParentDirs(sourceFilePath);
    if (fromParents) return fromParents;
  }

  const homeCandidate = path.join(os.homedir(), "projects", "lite-step");
  if (fs.existsSync(homeCandidate)) {
    return homeCandidate;
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
  const startedAt = Date.now();

  const config = vscode.workspace.getConfiguration("ginnung");
  const pythonExecutable = config.get<string>("pythonPath") || "python";

  const env = {
    ...process.env,
    PYTHONPATH: [repoRoot, process.env.PYTHONPATH]
      .filter(Boolean)
      .join(path.delimiter),
  };

  return new Promise<CompileResult>((resolve) => {
    execFile(
      pythonExecutable,
      [filePath],
      {
        cwd: dir,
        env,
        timeout: 300_000,
        maxBuffer: 50 * 1024 * 1024,
      },
      (error, stdout, stderr) => {
        if (error) {
          return resolve({
            success: false,
            ifcPath: "",
            stderr: stderr || stdout || error.message,
          });
        }

        // Look for generated IFC file next to the source script first
        const localCandidates = [
          path.join(dir, "output.ifc"),
          path.join(dir, `${baseName}.ifc`),
        ];

        for (const candidate of localCandidates) {
          if (fs.existsSync(candidate)) {
            return resolve({
              success: true,
              ifcPath: candidate,
              stderr: stderr || stdout || "",
            });
          }
        }

        // External fallbacks must have been updated during this run
        const externalCandidates = [
          path.join(repoRoot, "output.ifc"),
          path.join(os.tmpdir(), "output.ifc"),
          path.join(os.tmpdir(), `${baseName}.ifc`),
        ];

        for (const candidate of externalCandidates) {
          if (fs.existsSync(candidate)) {
            try {
              const stat = fs.statSync(candidate);
              if (stat.mtimeMs >= startedAt - 1000) {
                return resolve({
                  success: true,
                  ifcPath: candidate,
                  stderr: stderr || stdout || "",
                });
              }
            } catch {
              // ignore stat error
            }
          }
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
 * Read the generated IFC file as a Buffer.
 */
export async function readIfcFile(ifcPath: string): Promise<Buffer> {
  return fs.promises.readFile(ifcPath);
}
