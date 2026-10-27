import asyncio
import threading
import logging
from typing import Optional, Set, Callable
import websockets

from .protocol import parse_message, build_hello, build_status, build_error, build_push

logger = logging.getLogger(__name__)

class GinnungServer:
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

    def start(self, port: int, version: str) -> None:
        """Start the WebSocket server in a background thread."""
        if self.is_running():
            return
        self.port = port
        self._version = version
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
                origins=[None],
            ) as server:
                self._server = server
                print(f"Ginnung: Server listening on ws://{self.host}:{self.port}")
                logger.info(f"Ginnung server listening on ws://{self.host}:{self.port}")
                await self._stop_event.wait()

        try:
            self._loop.run_until_complete(serve_async())
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        except Exception as e:
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
        logger.info("Ginnung server stopped.")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    async def _handler(self, websocket) -> None:
        """Handle individual WebSocket connections."""
        self.active_connections.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        try:
            # Send initial hello handshake
            await websocket.send(build_hello(self._version))
            await self._message_loop(websocket)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        except Exception as e:
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
            direction = data.get("direction")
            filename = data.get("filename") or "import.ifc"
            content = data.get("content")
            blend_path = data.get("blendPath") or data.get("blend_path")

            if content:
                # Protocol 1.1: Inline text content
                print(f"Ginnung: Received inline model '{filename}' ({len(content)} chars)")
                if self.on_file_received:
                    try:
                        self.on_file_received(content.encode("utf-8"), filename, blend_path=blend_path)
                        await websocket.send(build_status(msg_id, "loaded"))
                        print(f"Ginnung: Sent status ack 'loaded' for {msg_id}")
                    except Exception as e:
                        print(f"Ginnung: Error loading received model: {e}")
                        await websocket.send(build_error(msg_id, f"Load failed: {str(e)}"))
            else:
                # Protocol 1.0: Expect binary payload frame next
                self._incoming_state[websocket] = {
                    "id": msg_id,
                    "filename": filename,
                    "blend_path": blend_path,
                    "sizeBytes": data.get("sizeBytes", 0)
                }
                print(f"Ginnung: Prepared to receive binary IFC '{filename}' ({data.get('sizeBytes', 0)} bytes)")
                logger.info(f"Prepared to receive IFC '{filename}' ({data.get('sizeBytes')} bytes)")
        elif msg_type == "status":
            print(f"Ginnung: Received status update for {msg_id}: {data.get('status')}")
            logger.info(f"Received status update for {msg_id}: {data.get('status')}")

    async def _handle_binary_message(self, websocket, msg: bytes) -> None:
        state = self._incoming_state.pop(websocket, None)
        if not state:
            filename = "output.ifc"
            msg_id = "binary-push"
            blend_path = None
            print(f"Ginnung: Received binary frame ({len(msg)} bytes) without prior control header. Loading as {filename}...")
        else:
            filename = state["filename"]
            msg_id = state["id"]
            blend_path = state.get("blend_path")
            print(f"Ginnung: Received binary frame for '{filename}', size: {len(msg)} bytes.")
            logger.info(f"Received binary frame for '{filename}', size: {len(msg)} bytes.")

        if self.on_file_received:
            try:
                self.on_file_received(msg, filename, blend_path=blend_path)
                await websocket.send(build_status(msg_id, "loaded"))
                print(f"Ginnung: Sent status ack 'loaded' for {msg_id}")
            except Exception as e:
                print(f"Ginnung: Error processing received IFC: {e}")
                logger.exception("Error processing received IFC")
                await websocket.send(build_error(msg_id, f"Load failed: {str(e)}"))

    def send_ifc_to_clients(self, ifc_bytes: bytes, filename: str) -> None:
        """Send an IFC file to all connected clients."""
        if not self.is_running() or not self._loop:
            logger.warning("Cannot send IFC, server is not running.")
            return

        async def _send() -> None:
            import uuid
            msg_id = str(uuid.uuid4())
            push_msg = build_push(msg_id, "to-ginnung", filename, len(ifc_bytes))
            
            for ws in self.active_connections.copy():
                try:
                    await ws.send(push_msg)
                    await ws.send(ifc_bytes)
                    logger.info(f"Sent IFC '{filename}' to {ws.remote_address}")
                except Exception as e:
                    logger.exception(f"Error sending to client {ws.remote_address}")

        asyncio.run_coroutine_threadsafe(_send(), self._loop)

# Global singleton instance
server_instance = GinnungServer()
