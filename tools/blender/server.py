import asyncio
import http
import json
import threading
import logging
from typing import Iterable, Optional, Set, Callable
import websockets

from .origin_policy import WEB, allowed_web_file, check_upgrade, parse_allowlist
from .protocol import parse_message, build_hello, build_status, build_error, build_push

logger = logging.getLogger(__name__)

#: How long a web page has to answer ``hello`` with ``status: "connected"``.
HANDSHAKE_TIMEOUT_S = 5.0

#: WebSocket close code for "you broke the rules" (RFC 6455 policy violation).
_POLICY_VIOLATION = 1008

class LiteStepServer:
    def __init__(self):
        self.host = '127.0.0.1'
        self.port = 9876
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server = None
        self.active_connections: Set[websockets.WebSocketServerProtocol] = set()
        self._version = "1.1"
        
        # Callback for when an IFC file is fully received
        self.on_file_received: Optional[Callable[[bytes, str], None]] = None
        
        self._incoming_state = {}
        self._stop_event = None
        self._allowlist = parse_allowlist()
        self.handshake_timeout = HANDSHAKE_TIMEOUT_S

    @property
    def rejected_origins(self):
        """The user's allowed-origin entries that were ignored, with reasons."""
        return list(self._allowlist.rejected)

    def start(self, port: int, version: str, allowed_origins: Iterable[str] = ()) -> None:
        """Start the WebSocket server in a background thread.

        ``allowed_origins`` are the user's extra web origins (see
        ``origin_policy``); invalid ones are ignored and reported.
        """
        if self.is_running():
            return
        self.port = port
        self._version = version
        self._allowlist = parse_allowlist(allowed_origins)
        for problem in self._allowlist.rejected:
            print(f"Lite-STEP: ignored allowed-origin entry {problem}")
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        async def serve_async():
            self._stop_event = asyncio.Event()
            async with websockets.serve(
                self._handler,
                self.host,
                self.port,
                max_size=100 * 1024 * 1024,
                process_request=self._process_request,
            ) as server:
                self._server = server
                print(f"Lite-STEP: Server listening on ws://{self.host}:{self.port}")
                logger.info(f"Lite-STEP server listening on ws://{self.host}:{self.port}")
                await self._stop_event.wait()

        try:
            self._loop.run_until_complete(serve_async())
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        except Exception:
            logger.exception("Failed to start WebSocket server")
        finally:
            try:
                if self._loop and not self._loop.is_closed():
                    self._loop.close()
            except Exception:
                pass

    def stop(self) -> None:
        """Stop the WebSocket server and background thread."""
        if not self.is_running():
            return
        
        if self._loop and self._loop.is_running() and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        self._thread = None
        self._loop = None
        self._server = None
        self._stop_event = None
        self.active_connections.clear()
        logger.info("Lite-STEP server stopped.")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _process_request(self, connection, request):
        """Refuse an upgrade that is not allowed, before the socket exists.

        A bare 403: a refused page learns nothing about what listens here.
        The role (``local`` or ``web``) is kept on the connection.
        """
        try:
            host = request.headers.get("Host")
            origin = request.headers.get("Origin")
        except Exception:  # a repeated Host or Origin header
            host, origin = None, "invalid"
        role = check_upgrade(host, origin, self.port, self._allowlist)
        if role is None:
            logger.warning("Refused connection from origin %r (host %r)", origin, host)
            return connection.respond(http.HTTPStatus.FORBIDDEN, "")
        connection.litestep_role = role
        return None

    async def _await_handshake(self, websocket) -> bool:
        """A web page must answer ``hello`` with ``status: "connected"``, in time, first."""
        try:
            first = await asyncio.wait_for(websocket.recv(), self.handshake_timeout)
            data = json.loads(first) if isinstance(first, str) else None
        except (asyncio.TimeoutError, ValueError):
            data = None
        if isinstance(data, dict) and data.get("type") == "status" and data.get("status") == "connected":
            return True
        await websocket.close(_POLICY_VIOLATION, "handshake first")
        return False

    async def _handler(self, websocket) -> None:
        """Handle individual WebSocket connections."""
        role = getattr(websocket, "litestep_role", None)
        if role is None:  # the upgrade check did not run: not allowed
            await websocket.close(_POLICY_VIOLATION, "not allowed")
            return
        logger.info(f"Client connected ({role}): {websocket.remote_address}")
        try:
            # Send initial hello handshake
            await websocket.send(build_hello(self._version))
            # A web page gets nothing more, and is not sent models, until it
            # answers. Local programs are not the threat and older ones push
            # on open without a handshake.
            if role == WEB and not await self._await_handshake(websocket):
                return
            self.active_connections.add(websocket)
            await self._message_loop(websocket)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        except Exception:
            logger.exception("WebSocket handler error")
        finally:
            self.active_connections.discard(websocket)
            if websocket in self._incoming_state:
                del self._incoming_state[websocket]

    async def _message_loop(self, websocket) -> None:
        async for msg in websocket:
            if isinstance(msg, str):
                # JSON Text Frame
                try:
                    data = parse_message(msg)
                    await self._handle_json_message(websocket, data)
                except Exception as e:
                    logger.exception("Error handling message")
                    await websocket.send(build_error(None, str(e)))
            elif isinstance(msg, bytes):
                # Binary Frame (IFC data)
                await self._handle_binary_message(websocket, msg)

    async def _handle_json_message(self, websocket, data: dict) -> None:
        msg_type = data.get("type")
        msg_id = data.get("id", "unknown")

        if msg_type in ("ifc.push", "file.push"):
            await self._handle_push(websocket, data, msg_id)
        elif msg_type == "status":
            print(f"Lite-STEP: Received status update for {msg_id}: {data.get('status')}")
            logger.info(f"Received status update for {msg_id}: {data.get('status')}")

    async def _refuse(self, websocket, msg_id, reason: str) -> None:
        """Answer a refused push with an error, then close: a web page that
        breaks the rules gets no second frame read, so a binary payload that
        follows a refused header can never be loaded."""
        logger.warning("Refused a push from a web page: %s", reason)
        try:
            await websocket.send(build_error(msg_id, reason))
        finally:
            await websocket.close(_POLICY_VIOLATION, "refused")

    async def _handle_push(self, websocket, data: dict, msg_id) -> None:
        filename = data.get("filename") or "import.ifc"
        content = data.get("content")
        blend_path = data.get("blendPath") or data.get("blend_path")
        if getattr(websocket, "litestep_role", None) == WEB:
            # A web page may not name a file on this disk for Blender to
            # open, and may push IFC only.
            blend_path = None
            if not allowed_web_file(filename):
                await self._refuse(websocket, msg_id, "Only .ifc files are accepted")
                return

        if content:
            # Protocol 1.1: Inline text content
            print(f"Lite-STEP: Received inline model '{filename}' ({len(content)} chars)")
            await self._load(websocket, content.encode("utf-8"), filename, blend_path, msg_id)
            return

        # Protocol 1.0: Expect binary payload frame next
        self._incoming_state[websocket] = {
            "id": msg_id,
            "filename": filename,
            "blend_path": blend_path,
            "sizeBytes": data.get("sizeBytes", 0)
        }
        print(f"Lite-STEP: Prepared to receive binary IFC '{filename}' ({data.get('sizeBytes', 0)} bytes)")
        logger.info(f"Prepared to receive IFC '{filename}' ({data.get('sizeBytes')} bytes)")

    async def _load(self, websocket, payload: bytes, filename: str, blend_path, msg_id) -> None:
        if not self.on_file_received:
            return
        try:
            self.on_file_received(payload, filename, blend_path=blend_path)
            await websocket.send(build_status(msg_id, "loaded"))
            print(f"Lite-STEP: Sent status ack 'loaded' for {msg_id}")
        except Exception as e:
            print(f"Lite-STEP: Error loading received model: {e}")
            logger.exception("Error loading received model")
            await websocket.send(build_error(msg_id, f"Load failed: {str(e)}"))

    async def _handle_binary_message(self, websocket, msg: bytes) -> None:
        state = self._incoming_state.pop(websocket, None)
        if not state:
            if getattr(websocket, "litestep_role", None) == WEB:
                # A web page must announce a binary frame with a header that
                # passed the checks above; a bare one is not loaded.
                await self._refuse(websocket, None, "A binary frame needs an ifc.push header first")
                return
            filename = "output.ifc"
            msg_id = "binary-push"
            blend_path = None
            print(f"Lite-STEP: Received binary frame ({len(msg)} bytes) without prior control header. Loading as {filename}...")
        else:
            filename = state["filename"]
            msg_id = state["id"]
            blend_path = state.get("blend_path")
            print(f"Lite-STEP: Received binary frame for '{filename}', size: {len(msg)} bytes.")
            logger.info(f"Received binary frame for '{filename}', size: {len(msg)} bytes.")

        await self._load(websocket, msg, filename, blend_path, msg_id)

    def send_ifc_to_clients(self, ifc_bytes: bytes, filename: str) -> None:
        """Send an IFC file to all connected clients."""
        if not self.is_running() or not self._loop:
            logger.warning("Cannot send IFC, server is not running.")
            return

        async def _send() -> None:
            import uuid
            msg_id = str(uuid.uuid4())
            push_msg = build_push(msg_id, "to-client", filename, len(ifc_bytes))
            
            for ws in self.active_connections.copy():
                try:
                    await ws.send(push_msg)
                    await ws.send(ifc_bytes)
                    logger.info(f"Sent IFC '{filename}' to {ws.remote_address}")
                except Exception:
                    logger.exception(f"Error sending to client {ws.remote_address}")

        asyncio.run_coroutine_threadsafe(_send(), self._loop)

# Global singleton instance
server_instance = LiteStepServer()
