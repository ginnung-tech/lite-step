"""A mesh read from a file in the project: ``Mesh(asset="meshes/statue.ply", sha256="...")``.

A scanned or sculpted mesh is data, not code: written into a script it is hundreds of kilobytes every reader of
the script pays for. As an asset the script names it by path AND by the sha256 of its bytes, so the script still
says exactly what it builds: a changed file is a loud refusal, never a different model.

The script itself still reads nothing (the restricted namespace has no file access); the DSL reads the asset,
under these rules:

- the path is relative, inside the asset root, and stays inside it once resolved (no ``..``, no absolute path, no
  link out). The root is ``LITE_STEP_ASSET_ROOT`` when a host sets it, else the folder of the script being run,
  else the working directory (a host that runs the script as a string runs it in the project);
- ``.ply`` (ASCII or binary little-endian) or ``.stl`` (binary or ASCII);
- coordinates in integer millimetres, as everywhere in Lite-STEP: a non-integer coordinate is refused (round and
  repair the mesh before exporting it; rounding can close thin gaps, so check it after);
- at most ``MAX_FACES`` triangles (decimate outside).
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

MAX_FACES = 2_000_000
_SHA = re.compile(r"^[0-9a-f]{64}$")
_PKG = Path(__file__).resolve().parents[1]  # lite_step/


class MeshAssetError(ValueError):
    """A mesh asset that cannot be read as asked; the message says what to do."""


def asset_root() -> Path:
    env = os.environ.get("LITE_STEP_ASSET_ROOT")
    if env:
        return Path(env).resolve()
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_globals.get("__name__") == "__lite_step__":
            break  # a script run as a string by a host: it has no file, and its project is the working directory
        f = frame.f_globals.get("__file__")
        if f:
            p = Path(f).resolve()
            if not p.is_relative_to(_PKG) and "pydantic" not in p.parts and "site-packages" not in p.parts:
                return p.parent
        frame = frame.f_back
    return Path.cwd().resolve()


def resolve(asset: str, root: Path | None = None) -> Path:
    root = (root or asset_root()).resolve()
    if not asset or Path(asset).is_absolute() or ".." in Path(asset).parts or re.match(r"^[A-Za-z]:", asset):
        raise MeshAssetError(f"Mesh asset {asset!r}: a relative path inside the project, without '..'")
    p = (root / asset).resolve()
    if not p.is_relative_to(root):
        raise MeshAssetError(f"Mesh asset {asset!r} resolves outside the project ({root})")
    if p.suffix.lower() not in (".ply", ".stl"):
        raise MeshAssetError(f"Mesh asset {asset!r}: .ply or .stl (export the cleaned mesh as one of those)")
    if not p.is_file():
        raise MeshAssetError(f"Mesh asset {asset!r} is not a file under {root}")
    return p


def load(asset: str, sha256: str | None, root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(vertices (N, 3) int64 mm, faces (M, 3) int64) of the asset, after checking its bytes against `sha256`."""
    if not sha256 or not _SHA.match(sha256.lower()):
        raise MeshAssetError("Mesh(asset=...) needs sha256= the file's SHA-256 (64 hex characters): the script names "
                             "the exact bytes it builds from (`python -c \"import hashlib,sys; print(hashlib.sha256("
                             "open(sys.argv[1],'rb').read()).hexdigest())\" FILE`)")
    path = resolve(asset, root)
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256.lower():
        raise MeshAssetError(f"Mesh asset {asset!r} has changed: its sha256 is {got}, the script names {sha256}. "
                             "If the new file is meant, put its hash in the script.")
    verts, faces = _read_ply(data, asset) if path.suffix.lower() == ".ply" else _read_stl(data, asset)
    if len(faces) == 0:
        raise MeshAssetError(f"Mesh asset {asset!r} has no faces")
    if len(faces) > MAX_FACES:
        raise MeshAssetError(f"Mesh asset {asset!r} has {len(faces):,} triangles; at most {MAX_FACES:,} (decimate it "
                             "outside, keeping detail where it shows)")
    if faces.min() < 0 or faces.max() >= len(verts):
        raise MeshAssetError(f"Mesh asset {asset!r}: a face indexes a vertex that is not there")
    rounded = np.rint(verts)
    off = np.abs(verts - rounded)
    if off.max(initial=0.0) > 1e-6:
        i = int(np.argmax(off.max(axis=1)))
        raise MeshAssetError(f"Mesh asset {asset!r}: vertex {i} is {verts[i].tolist()}, not integer millimetres. Write "
                             "the mesh in millimetres and round it before export (then check it is still watertight)")
    return rounded.astype(np.int64), faces.astype(np.int64)


def is_closed(faces: np.ndarray) -> bool:
    """Every edge used exactly twice, once in each direction: closed and consistently wound."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    fwd = {tuple(x) for x in e.tolist()}
    if len(fwd) != len(e):
        return False  # an edge used twice in the same direction
    return all((b, a) in fwd for a, b in fwd)


# -- readers ----------------------------------------------------------------------------------------------------

_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
              "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def _read_ply(data: bytes, name: str) -> tuple[np.ndarray, np.ndarray]:
    end = data.find(b"end_header")
    if not data.startswith(b"ply") or end < 0:
        raise MeshAssetError(f"{name}: not a PLY file")
    nl = data.index(b"\n", end) + 1
    header = data[:nl].decode("ascii", "replace").splitlines()
    fmt = next((ln.split()[1] for ln in header if ln.startswith("format")), "")
    elements: list[tuple[str, int, list[tuple]]] = []
    for ln in header:
        p = ln.split()
        if not p:
            continue
        if p[0] == "element":
            elements.append((p[1], int(p[2]), []))
        elif p[0] == "property" and elements:
            if p[1] == "list":
                elements[-1][2].append(("list", p[2], p[3], p[4]))
            else:
                elements[-1][2].append(("scalar", p[1], p[2]))
    body = data[nl:]
    verts = faces = None
    if fmt == "ascii":
        tokens = body.decode("ascii", "replace").split()
        pos = 0
        for el, count, props in elements:
            if el == "vertex":
                names = [pr[2] for pr in props]
                vals = np.array(tokens[pos:pos + count * len(props)], dtype=np.float64).reshape(count, len(props))
                pos += count * len(props)
                verts = vals[:, [names.index(a) for a in ("x", "y", "z")]]
            elif el == "face":
                rows = []
                for _ in range(count):
                    n = int(tokens[pos])
                    rows.append([int(v) for v in tokens[pos + 1:pos + 1 + n]])
                    pos += 1 + n
                faces = _fan(rows)
            else:
                for _ in range(count):  # skip an element we do not use
                    for pr in props:
                        pos += 1 + int(tokens[pos]) if pr[0] == "list" else 1
    elif fmt == "binary_little_endian":
        off = 0
        for el, count, props in elements:
            if all(pr[0] == "scalar" for pr in props):
                dt = np.dtype([(pr[2], "<" + _PLY_TYPES[pr[1]]) for pr in props])
                arr = np.frombuffer(body, dtype=dt, count=count, offset=off)
                off += dt.itemsize * count
                if el == "vertex":
                    verts = np.stack([arr[a].astype(np.float64) for a in ("x", "y", "z")], axis=1)
                continue
            if el != "face" or len(props) != 1:
                raise MeshAssetError(f"{name}: element {el} with list properties is not supported; export vertices "
                                     "and faces only")
            _, ct, it, _ = props[0]
            cdt, idt = np.dtype("<" + _PLY_TYPES[ct]), np.dtype("<" + _PLY_TYPES[it])
            # the common case: every face a triangle, read in one go
            tri = np.dtype([("n", cdt), ("v", idt, 3)])
            arr = np.frombuffer(body, dtype=tri, count=count, offset=off) if off + tri.itemsize * count <= len(body) else None
            if arr is not None and np.all(arr["n"] == 3):
                faces = arr["v"].astype(np.int64)
                off += tri.itemsize * count
                continue
            rows = []
            for _ in range(count):
                n = int(np.frombuffer(body, cdt, 1, off)[0])
                off += cdt.itemsize
                rows.append(np.frombuffer(body, idt, n, off).tolist())
                off += idt.itemsize * n
            faces = _fan(rows)
    else:
        raise MeshAssetError(f"{name}: PLY format {fmt!r}; save it as ascii or binary_little_endian")
    if verts is None or faces is None:
        raise MeshAssetError(f"{name}: a PLY needs a vertex element (x, y, z) and a face element")
    return verts, faces


def _fan(rows: list[list[int]]) -> np.ndarray:
    out = [(r[0], r[i], r[i + 1]) for r in rows for i in range(1, len(r) - 1)]
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def _read_stl(data: bytes, name: str) -> tuple[np.ndarray, np.ndarray]:
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                                count=n, offset=84)
            return _merge(rec["v"].reshape(-1, 3).astype(np.float64))
    text = data.decode("ascii", "replace")
    if not text.lstrip().startswith("solid"):
        raise MeshAssetError(f"{name}: not an STL file")
    pts = [[float(v) for v in ln.split()[1:4]] for ln in text.splitlines() if ln.strip().startswith("vertex")]
    if len(pts) % 3:
        raise MeshAssetError(f"{name}: an ASCII STL facet without three vertices")
    return _merge(np.array(pts, dtype=np.float64))


def _merge(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """STL repeats every corner per facet: weld equal coordinates into shared vertices (exact match, the file is
    in whole millimetres or it is refused anyway)."""
    verts, inverse = np.unique(np.rint(corners * 1e6) / 1e6, axis=0, return_inverse=True)
    return verts, inverse.reshape(-1, 3)
