"""
Build script for Lite-STEP tools (Blender add-on & VS Code extension).

Usage:
    python tools/build.py            # builds both
    python tools/build.py blender    # builds Blender zip
    python tools/build.py vscode     # builds VS Code VSIX
"""

import os
import sys
import zipfile
import subprocess
import shutil

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIR)
BLENDER_DIR = os.path.join(TOOLS_DIR, "blender")
VSCODE_DIR = os.path.join(TOOLS_DIR, "vscode")


def build_blender():
    zip_path = os.path.join(TOOLS_DIR, "blender-litestep.zip")
    print(f"\n[Blender] Building extension zip at: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(BLENDER_DIR):
            if "__pycache__" in root or ".pytest_cache" in root:
                continue
            for file in files:
                if file.endswith((".pyc", ".zip")):
                    continue
                abs_file = os.path.join(root, file)
                rel_file = os.path.relpath(abs_file, BLENDER_DIR)
                zf.write(abs_file, rel_file)
                print(f"  + {rel_file}")

    print(f"[Blender] Created {zip_path} ({os.path.getsize(zip_path):,} bytes)")


def build_vscode():
    print(f"\n[VS Code] Building extension in: {VSCODE_DIR}")
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    npx_cmd = "npx.cmd" if sys.platform == "win32" else "npx"

    if not os.path.exists(os.path.join(VSCODE_DIR, "node_modules")):
        print("  Running npm install...")
        subprocess.run([npm_cmd, "install"], cwd=VSCODE_DIR, check=True)

    print("  Running npm run build...")
    subprocess.run([npm_cmd, "run", "build"], cwd=VSCODE_DIR, check=True)

    print("  Packaging extension with vsce...")
    subprocess.run(
        [npx_cmd, "-y", "@vscode/vsce", "package", "--no-dependencies"],
        cwd=VSCODE_DIR,
        check=True,
    )
    print("[VS Code] Package complete.")


def main():
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if target in ("all", "blender"):
        build_blender()
    if target in ("all", "vscode"):
        build_vscode()


if __name__ == "__main__":
    main()

