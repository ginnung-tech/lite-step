"""Headless 6-axis ThatOpen renderer for IFC files.

Usage::

    python tools/render/render_views.py path/to/output.ifc
    python tools/render/render_views.py path/to/model_dir --mode cardinal --out-dir renders

Drives ``tools/render/viewer/index.html`` (``@thatopen/components`` +
``@thatopen/fragments`` with ``LodMode.ALL_VISIBLE``) in headless Chromium and
saves six 1024x1024 PNGs (``xpos``, ``xneg``, ``ypos``, ``yneg``, ``zpos``,
``zneg`` by default). Wait for two animation frames after ``renderView`` so
heavy meshes finish painting before capture.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

VIEWER_DIR = Path(__file__).resolve().parent / "viewer"


@dataclass(frozen=True)
class RenderJob:
    """One IFC file to render into six view PNGs."""

    ifc_path: Path
    out_dir: Path
    prefix: str = "render_"


def _source_hash(src_dir: Path) -> str:
    h = hashlib.sha256()
    files = sorted(
        (p.relative_to(src_dir).as_posix(), p)
        for p in src_dir.rglob("*.mjs")
        if p.is_file()
    )
    for rel, path in files:
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        h.update(f"{rel}:{file_hash}\n".encode("utf-8"))
    return h.hexdigest()


def ensure_vendor_bundle(viewer_dir: Path = VIEWER_DIR, auto_build: bool = True) -> Path:
    """Return ``viewer_dir / 'vendor'``, building it if missing or stale."""
    vendor_dir = viewer_dir / "vendor"
    info_path = vendor_dir / "BUILD_INFO.json"
    app_js = vendor_dir / "render-app.js"
    expected_hash = _source_hash(viewer_dir / "src")

    is_fresh = False
    if app_js.is_file() and info_path.is_file():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            is_fresh = info.get("sourceHash") == expected_hash
        except (OSError, json.JSONDecodeError):
            is_fresh = False

    if is_fresh:
        return vendor_dir
    if not auto_build:
        raise FileNotFoundError(
            f"Viewer vendor bundle missing or stale at {vendor_dir}. "
            f"Run: cd {viewer_dir} && bun install && bun run build:vendor"
        )

    bun = shutil.which("bun")
    if bun is None:
        raise RuntimeError("`bun` not found on PATH; cannot build tools/render/viewer/vendor")
    if not (viewer_dir / "node_modules").is_dir():
        cmd = [bun, "install", "--frozen-lockfile"] if (viewer_dir / "bun.lock").is_file() else [bun, "install"]
        if os.name == "nt":
            cmd.append("--backend=copyfile")
        subprocess.run(cmd, cwd=viewer_dir, check=True)
    subprocess.run([bun, "run", "build:vendor"], cwd=viewer_dir, check=True)
    return vendor_dir


def _resolve_ifc_target(target: Path) -> Path:
    resolved = target.resolve()
    if resolved.is_dir():
        candidate = resolved / "output.ifc"
        if not candidate.is_file():
            raise FileNotFoundError(f"Expected output.ifc inside {resolved}")
        return candidate
    if not resolved.is_file():
        raise FileNotFoundError(f"IFC file not found: {resolved}")
    return resolved


def _stage_viewer(stage_dir: Path, viewer_dir: Path, vendor_dir: Path) -> None:
    shutil.copy2(viewer_dir / "index.html", stage_dir / "index.html")
    dst_vendor = stage_dir / "vendor"
    try:
        dst_vendor.symlink_to(vendor_dir, target_is_directory=True)
    except OSError:
        shutil.copytree(vendor_dir, dst_vendor)


async def _render_one_job(
    browser,
    stage_dir: Path,
    viewer_url: str,
    job: RenderJob,
    timeout_ms: int,
    verbose: bool,
) -> list[Path]:
    job.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(job.ifc_path, stage_dir / "output.ifc")
    page = await browser.new_page(viewport={"width": 1024, "height": 1024})
    page_errors: list[str] = []
    if verbose:
        page.on("console", lambda msg: print(f"  [browser] {msg.text}"))
    page.on("pageerror", lambda err: page_errors.append(str(err)))
    try:
        await page.goto(viewer_url)
        await page.wait_for_function("window.__ready === true || !!window.__error", timeout=timeout_ms)
        err = await page.evaluate("window.__error || null")
        if err:
            raise RuntimeError(f"viewer error for {job.ifc_path.name}: {err} | page_errors={page_errors}")
        stems = await page.evaluate("window.__viewStems")
        written: list[Path] = []
        for stem in stems:
            await page.evaluate("(s) => window.renderView(s)", stem)
            await page.evaluate("() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))")
            out_png = job.out_dir / f"{job.prefix}{stem}.png"
            await page.locator("#wrap").screenshot(path=str(out_png))
            written.append(out_png)
        return written
    finally:
        await page.close()


async def render_batch_async(
    jobs: Sequence[RenderJob],
    *,
    mode: str = "cardinal",
    elev: float = 22.0,
    extra_params: str = "",
    timeout_ms: int = 90_000,
    verbose: bool = False,
    viewer_dir: Path = VIEWER_DIR,
) -> dict[Path, list[Path]]:
    """Render one or more IFC jobs while keeping a single Chromium instance open."""
    from playwright.async_api import async_playwright

    if not jobs:
        return {}
    vendor_dir = ensure_vendor_bundle(viewer_dir)
    query = f"?mode={mode}&elev={elev:g}"
    if extra_params:
        query = f"{query}&{extra_params.lstrip('&?')}"

    results: dict[Path, list[Path]] = {}
    with tempfile.TemporaryDirectory(prefix="lite-step-render-") as tmp:
        stage_dir = Path(tmp)
        _stage_viewer(stage_dir, viewer_dir, vendor_dir)
        url = f"file:///{stage_dir.as_posix()}/index.html{query}"
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--allow-file-access-from-files", "--enable-webgl", "--ignore-gpu-blocklist"],
            )
            try:
                for job in jobs:
                    results[job.ifc_path] = await _render_one_job(
                        browser, stage_dir, url, job, timeout_ms, verbose
                    )
            finally:
                await browser.close()
    return results


def render_ifc(
    target: Path,
    *,
    out_dir: Path | None = None,
    prefix: str = "render_",
    mode: str = "cardinal",
    elev: float = 22.0,
    extra_params: str = "",
    timeout_ms: int = 90_000,
    verbose: bool = False,
) -> list[Path]:
    """Render six views of ``target`` (an ``.ifc`` file or dir containing ``output.ifc``)."""
    ifc_path = _resolve_ifc_target(target)
    dest_dir = (out_dir or ifc_path.parent).resolve()
    res = asyncio.run(
        render_batch_async(
            [RenderJob(ifc_path=ifc_path, out_dir=dest_dir, prefix=prefix)],
            mode=mode,
            elev=elev,
            extra_params=extra_params,
            timeout_ms=timeout_ms,
            verbose=verbose,
        )
    )
    return res[ifc_path]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render 6-axis ThatOpen views of an IFC file.")
    parser.add_argument("target", nargs="?", default=".", help="IFC file or directory containing output.ifc")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for PNGs (default: beside IFC)")
    parser.add_argument("--prefix", default="render_", help="Output PNG filename prefix (default: render_)")
    parser.add_argument("--mode", choices=("cardinal", "aerial"), default="cardinal", help="Camera preset (default: cardinal 6-axis)")
    parser.add_argument("--elev", type=float, default=22.0, help="Horizontal camera elevation in degrees (default: 22)")
    parser.add_argument("--params", default="", help="Extra query parameters passed to the viewer (e.g. edges=1)")
    parser.add_argument("--verbose", action="store_true", help="Print viewer console messages")
    args = parser.parse_args(list(argv) if argv is not None else None)

    written = render_ifc(
        Path(args.target),
        out_dir=args.out_dir,
        prefix=args.prefix,
        mode=args.mode,
        elev=args.elev,
        extra_params=args.params,
        verbose=args.verbose,
    )
    for png in written:
        print(f"Saved {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
