#!/usr/bin/env python
"""
Package Lite-STEP assets (model.ifc, model.blend, styles/lite-step.css, textures) into
compressed distribution archives (.zip and .ifczip).

Usage:
    python -m lite_step.tests.tools.package_assets [--all] [--bundle-only] [--ifczip-only]
"""

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "lite_step" / "assets"
BUNDLE_ZIP = ASSETS_DIR / "model_bundle.zip"
IFC_ZIP = ASSETS_DIR / "model.ifczip"

REQUIRED_BUNDLE_FILES = [
    "model.ifc",
    "model.blend",
    "styles/lite-step.css",
]


def create_ifczip(ifc_path: Path, out_path: Path) -> Path:
    """Create a standard buildingSMART .ifczip archive containing model.ifc."""
    ifc_path = Path(ifc_path)
    out_path = Path(out_path)

    if not ifc_path.exists():
        raise FileNotFoundError(f"IFC file not found: {ifc_path}")
    orig_size = ifc_path.stat().st_size
    if orig_size == 0:
        raise ValueError(f"IFC file is empty (0 bytes): {ifc_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(ifc_path, arcname=ifc_path.name)

    zip_size = out_path.stat().st_size
    ratio = (1 - zip_size / orig_size) * 100
    print(f"[OK] Created {out_path} ({zip_size:,} bytes, compressed {ratio:.1f}% from {orig_size:,} bytes)")
    return out_path


def create_bundle_zip(assets_dir: Path, out_path: Path) -> Path:
    """Package model.ifc, model.blend, styles/lite-step.css, and textures/ into model_bundle.zip.

    Fails loudly if any required component is missing or empty.
    """
    assets_dir = Path(assets_dir)
    out_path = Path(out_path)

    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets directory does not exist: {assets_dir}")

    files_to_pack = []

    # 1. Core bundle files
    for rel_name in REQUIRED_BUNDLE_FILES:
        f = assets_dir / rel_name
        if not f.exists():
            raise FileNotFoundError(f"Required bundle asset missing: {f}")
        if f.stat().st_size == 0:
            raise ValueError(f"Required bundle asset is empty (0 bytes): {f}")
        files_to_pack.append(f)

    # 2. Textures directory
    textures_dir = assets_dir / "textures"
    if not textures_dir.exists():
        raise FileNotFoundError(f"Textures directory missing: {textures_dir}")

    texture_images = sorted(textures_dir.glob("*.jpg")) + sorted(textures_dir.glob("*.png"))
    if not texture_images:
        raise FileNotFoundError(f"No texture image files found in {textures_dir}")

    for img in texture_images:
        if img.stat().st_size == 0:
            raise ValueError(f"Texture image is empty (0 bytes): {img}")
        files_to_pack.append(img)

    # 3. Provenance notices (if present)
    for notice_path in [assets_dir / "NOTICE.md", textures_dir / "NOTICE.md"]:
        if notice_path.exists() and notice_path.stat().st_size > 0:
            files_to_pack.append(notice_path)

    # Dedup preserving order
    unique_files = list(dict.fromkeys(files_to_pack))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in unique_files:
            arcname = str(f.relative_to(assets_dir)).replace("\\", "/")
            zf.write(f, arcname=arcname)

    # Sanity check created archive
    with zipfile.ZipFile(out_path, "r") as zf:
        names = zf.namelist()
        if not names:
            raise RuntimeError(f"Created bundle archive {out_path} is empty!")

    total_size = out_path.stat().st_size
    print(f"[OK] Created {out_path} ({total_size:,} bytes with {len(unique_files)} items)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Package Lite-STEP assets into zip archives")
    parser.add_argument("--all", action="store_true", default=True, help="Create both bundle.zip and model.ifczip (default)")
    parser.add_argument("--bundle-only", action="store_true", help="Create only bundle zip")
    parser.add_argument("--ifczip-only", action="store_true", help="Create only .ifczip")
    args = parser.parse_args()

    ifc_file = ASSETS_DIR / "model.ifc"

    try:
        if args.ifczip_only:
            create_ifczip(ifc_file, IFC_ZIP)
        elif args.bundle_only:
            create_bundle_zip(ASSETS_DIR, BUNDLE_ZIP)
        else:
            create_ifczip(ifc_file, IFC_ZIP)
            create_bundle_zip(ASSETS_DIR, BUNDLE_ZIP)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"[ERROR] Packaging failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
