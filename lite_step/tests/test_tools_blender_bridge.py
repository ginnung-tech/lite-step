"""Who may connect to the Blender add-on's local server (tools/blender).

The add-on's package imports ``bpy``; its ``origin_policy``, ``protocol`` and
``server`` modules do not, so they are loaded here under a stand-in package
and the server is run on a real socket.
"""
import asyncio
import importlib.util
import json
import socket
import sys
import time
import types
from pathlib import Path

import pytest

websockets = pytest.importorskip("websockets")

BLENDER_DIR = Path(__file__).resolve().parents[2] / "tools" / "blender"
PACKAGE = "litestep_blender_under_test"


def _load(name):
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(BLENDER_DIR)]
        sys.modules[PACKAGE] = package
    full = f"{PACKAGE}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, BLENDER_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


policy = _load("origin_policy")
_load("protocol")
server_module = _load("server")

PORT = 9876
BUILT_IN = policy.parse_allowlist()


# ── the rules ────────────────────────────────────────────────────────


def test_the_built_in_origin_is_a_web_client_by_exact_match():
    assert policy.check_upgrade(f"127.0.0.1:{PORT}", "https://ginnung.tech", PORT, BUILT_IN) == policy.WEB
    assert policy.check_upgrade(f"localhost:{PORT}", "https://ginnung.tech", PORT, BUILT_IN) == policy.WEB


def test_no_origin_is_a_program_on_this_computer():
    assert policy.check_upgrade(f"127.0.0.1:{PORT}", None, PORT, BUILT_IN) == policy.LOCAL


@pytest.mark.parametrize("origin", [
    "https://ginnung.tech.evil.example",
    "https://evilginnung.tech",
    "https://www.ginnung.tech",
    "http://ginnung.tech",
    "https://ginnung.tech:8443",
    "https://GINNUNG.tech",
    "https://ginnung.tech/",
    "http://localhost:5173",
    "null",
    "https://example.com",
])
def test_look_alike_and_unlisted_origins_are_refused(origin):
    assert policy.check_upgrade(f"127.0.0.1:{PORT}", origin, PORT, BUILT_IN) is None


@pytest.mark.parametrize("host", [None, "", "evil.example", f"evil.example:{PORT}", "127.0.0.1:9999", "127.0.0.1"])
def test_a_host_that_is_not_this_loopback_port_is_refused(host):
    assert policy.check_upgrade(host, "https://ginnung.tech", PORT, BUILT_IN) is None
    assert policy.check_upgrade(host, None, PORT, BUILT_IN) is None


def test_a_user_entry_is_allowed_and_only_that_exact_origin():
    allow = policy.parse_allowlist(policy.split_entries("https://app.example.com, http://localhost:5173"))
    assert allow.rejected == []
    assert policy.check_upgrade(f"127.0.0.1:{PORT}", "https://app.example.com", PORT, allow) == policy.WEB
    assert policy.check_upgrade(f"127.0.0.1:{PORT}", "http://localhost:5174", PORT, allow) is None
    assert policy.check_upgrade(f"127.0.0.1:{PORT}", "https://app.example.com.evil.example", PORT, allow) is None


@pytest.mark.parametrize("entry", [
    "http://app.example.com",
    "https://*.example.com",
    "https://app.example.com/",
    "https://app.example.com/path",
    "https://APP.example.com",
    "https://app.example.com:443",
    "https://user@app.example.com",
    " https://app.example.com",
    "app.example.com",
    "vscode-webview://",
    "*",
    "",
    42,
    None,
])
def test_a_bad_entry_is_ignored_with_a_reason_and_never_widens_the_list(entry):
    allow = policy.parse_allowlist([entry])
    assert len(allow.rejected) == 1
    assert allow.origins == frozenset({"https://ginnung.tech"})


def test_a_web_page_may_push_ifc_only():
    assert policy.allowed_web_file("Model.IFC")
    for name in ("output.py", "scene.blend", "run.bat", "model.ifc.exe", ""):
        assert not policy.allowed_web_file(name)


# ── the door, on a real socket ───────────────────────────────────────


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def running():
    """A started server on a free port, with what it was asked to load."""
    server = server_module.LiteStepServer()
    server.handshake_timeout = 0.3
    loaded = []
    server.on_file_received = lambda data, filename, blend_path=None: loaded.append((filename, blend_path))
    port = _free_port()
    server.start(port=port, version="1.3")
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    yield server, port, loaded
    server.stop()


def _upgrade_status(port, origin=None, host=None):
    """The HTTP status line a hand-written upgrade request gets."""
    lines = [
        "GET / HTTP/1.1",
        f"Host: {host or f'127.0.0.1:{port}'}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==",
        "Sec-WebSocket-Version: 13",
    ]
    if origin is not None:
        lines.append(f"Origin: {origin}")
    with socket.create_connection(("127.0.0.1", port), timeout=3) as s:
        s.sendall(("\r\n".join(lines) + "\r\n\r\n").encode())
        reply = s.recv(4096).decode("latin-1")
    return reply.split("\r\n", 1)[0], reply


@pytest.mark.parametrize("origin", ["https://evil.example", "https://ginnung.tech.evil.example", "http://localhost:5173", "null"])
def test_a_rogue_origin_gets_a_bare_403(running, origin):
    _, port, _ = running
    status, reply = _upgrade_status(port, origin=origin)
    assert status.startswith("HTTP/1.1 403"), status
    assert "hello" not in reply


def test_a_forged_host_is_refused_even_with_the_right_origin(running):
    _, port, _ = running
    status, _ = _upgrade_status(port, origin="https://ginnung.tech", host=f"evil.example:{port}")
    assert status.startswith("HTTP/1.1 403"), status


def test_the_allowed_origin_is_upgraded(running):
    _, port, _ = running
    status, _ = _upgrade_status(port, origin="https://ginnung.tech")
    assert status.startswith("HTTP/1.1 101"), status


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, 10))


PUSH = {"type": "file.push", "id": "t1", "filename": "model.ifc", "content": "ISO-10303-21;",
        "blendPath": "C:/somewhere/else.blend"}
CONNECTED = {"type": "status", "id": "s1", "status": "connected"}


def test_a_silent_web_page_is_closed_after_the_timeout(running):
    _, port, loaded = running

    async def go():
        async with websockets.connect(f"ws://127.0.0.1:{port}", origin="https://ginnung.tech") as ws:
            assert json.loads(await ws.recv())["type"] == "hello"
            with pytest.raises(websockets.exceptions.ConnectionClosed) as closed:
                await ws.recv()
            assert closed.value.rcvd.code == 1008

    _run(go())
    assert loaded == []


def test_a_web_page_that_pushes_before_the_handshake_is_closed_and_nothing_loads(running):
    _, port, loaded = running

    async def go():
        async with websockets.connect(f"ws://127.0.0.1:{port}", origin="https://ginnung.tech") as ws:
            await ws.recv()
            await ws.send(json.dumps(PUSH))
            with pytest.raises(websockets.exceptions.ConnectionClosed):
                await ws.recv()

    _run(go())
    assert loaded == []


def test_a_web_page_after_the_handshake_loads_ifc_but_never_names_a_blend_file(running):
    _, port, loaded = running

    async def go():
        async with websockets.connect(f"ws://127.0.0.1:{port}", origin="https://ginnung.tech") as ws:
            await ws.recv()
            await ws.send(json.dumps(CONNECTED))
            await ws.send(json.dumps(PUSH))
            assert json.loads(await ws.recv())["status"] == "loaded"
            await ws.send(json.dumps({**PUSH, "id": "t2", "filename": "output.py"}))
            assert json.loads(await ws.recv())["type"] == "error"

    _run(go())
    assert loaded == [("model.ifc", None)]


def test_a_local_program_needs_no_handshake_and_may_name_a_blend_file(running):
    _, port, loaded = running

    async def go():
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            await ws.recv()
            await ws.send(json.dumps(PUSH))
            assert json.loads(await ws.recv())["status"] == "loaded"

    _run(go())
    assert loaded == [("model.ifc", "C:/somewhere/else.blend")]


# ── the open repo names no private environment ───────────────────────


def test_the_tools_name_only_the_public_origin():
    """Other environments are a user setting, never a default in this repo."""
    tools = BLENDER_DIR.parent
    private_zone = "grab" + "." + "technology"
    skip = {"node_modules", "out", "out-test", "wheels", "__pycache__"}
    hits = []
    for path in tools.rglob("*"):
        if not path.is_file() or skip & set(path.parts) or path.suffix in {".vsix", ".zip", ".wasm", ".whl"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if private_zone in text:
            hits.append(str(path.relative_to(tools)))
    assert hits == []
    assert policy.BUILT_IN_ORIGINS == ("https://ginnung.tech",)
