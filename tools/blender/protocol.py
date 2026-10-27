import json
from typing import Any, Dict, List, Optional

def parse_message(raw_msg: str) -> Dict[str, Any]:
    """Parse an incoming JSON string into a dictionary."""
    try:
        data = json.loads(raw_msg)
        if not isinstance(data, dict):
            raise ValueError("Message must be a JSON object.")
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

def build_hello(version: str) -> str:
    """Build a hello handshake message."""
    return json.dumps({
        "type": "hello",
        "version": version,
        "capabilities": ["ifc.push"]
    })

def build_status(msg_id: str, status: str) -> str:
    """Build a status update message."""
    return json.dumps({
        "type": "status",
        "version": "1.1",
        "id": msg_id,
        "status": status
    })

def build_error(msg_id: Optional[str], error: str) -> str:
    """Build an error message."""
    return json.dumps({
        "type": "error",
        "version": "1.1",
        "id": msg_id,
        "error": error
    })

def build_push(msg_id: str, direction: str, filename: str, size_bytes: int) -> str:
    """Build an ifc.push message header."""
    return json.dumps({
        "type": "ifc.push",
        "version": "1.1",
        "id": msg_id,
        "direction": direction,
        "filename": filename,
        "sizeBytes": size_bytes
    })
