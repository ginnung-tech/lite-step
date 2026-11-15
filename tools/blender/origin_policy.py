"""Who may connect to the local bridge.

The threat is a website the user did not choose, scanning 127.0.0.1 for this
port. A page cannot set or forge its ``Origin`` header, and a browser sends
``Origin: https://example.com`` only for a page it loaded from that site over
verified TLS. So an exact match on ``Origin`` is the whole defence: no token,
no pairing, no network call.

A connection with no ``Origin`` is a program on this computer (the VS Code
extension, a script). It belongs to the same user as Blender and is accepted.

No ``bpy`` import: this module is tested outside Blender. The VS Code
extension has the same rules in ``tools/vscode/src/origin-policy.ts``.
"""
from typing import FrozenSet, Iterable, List, NamedTuple, Optional
from urllib.parse import urlsplit

#: Web origins accepted without any setting.
BUILT_IN_ORIGINS = ("https://ginnung.tech",)

#: A program on this computer (no Origin) / an allowed web page.
LOCAL = "local"
WEB = "web"

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1"})
_DEFAULT_PORTS = {"https": 443, "http": 80}


class Allowlist(NamedTuple):
    origins: FrozenSet[str]
    #: User entries that were ignored, each with the reason.
    rejected: List[str]


def entry_problem(entry: object) -> Optional[str]:
    """Why ``entry`` cannot be an allowed origin, or ``None`` when it can.

    An entry is a bare origin exactly as a browser sends it: scheme, host and
    (non-default) port, nothing else. It must be https; http is accepted only
    on this computer, for a developer's own dev server. A wildcard, a path or
    a trailing slash would turn the exact match into a guess.
    """
    if not isinstance(entry, str) or not entry.strip():
        return "not a text value"
    if entry != entry.strip():
        return "has spaces around it"
    if "*" in entry:
        return "wildcards are not supported; list each origin"
    try:
        parts = urlsplit(entry)
        port = parts.port
    except ValueError:
        return "not a web address"
    host = parts.hostname
    if parts.scheme not in _DEFAULT_PORTS or not host:
        return "not a web address (https://host)"
    if parts.username is not None or parts.password is not None:
        return "must not contain a user name"
    written = f"{parts.scheme}://{host}"
    if port is not None and port != _DEFAULT_PORTS[parts.scheme]:
        written += f":{port}"
    if written != entry:
        return f'must be written exactly as "{written}" (no path, no trailing slash, lower case)'
    if parts.scheme == "https":
        return None
    if host in _LOOPBACK_HOSTS:
        return None
    return "must be https (http is accepted only on localhost or 127.0.0.1)"


def split_entries(text: str) -> List[str]:
    """The origins in a preference field: separated by commas, spaces or new lines."""
    return [part for part in text.replace(",", " ").split() if part]


def parse_allowlist(user_entries: Iterable[object] = ()) -> Allowlist:
    """The built-in origins plus the user's valid entries."""
    origins = set(BUILT_IN_ORIGINS)
    rejected: List[str] = []
    for entry in user_entries:
        problem = entry_problem(entry)
        if problem is None:
            origins.add(entry)  # type: ignore[arg-type]
        else:
            rejected.append(f"{entry!r}: {problem}")
    return Allowlist(frozenset(origins), rejected)


def check_upgrade(host: Optional[str], origin: Optional[str], port: int,
                  allowlist: Allowlist) -> Optional[str]:
    """The role of a connection asking to be upgraded, or ``None`` to refuse it.

    ``Host`` must name this computer's loopback and this port: a page that
    reached us through a DNS name pointed at 127.0.0.1 (DNS rebinding) carries
    that name here. ``Origin`` absent is a local program; present, it must
    equal an allowlist entry.
    """
    h = (host or "").lower()
    if h not in (f"127.0.0.1:{port}", f"localhost:{port}"):
        return None
    if not origin:
        return LOCAL
    return WEB if origin in allowlist.origins else None


def allowed_web_file(filename: str) -> bool:
    """Whether a web page may push this file to Blender: an ``.ifc`` name only."""
    return filename.lower().endswith(".ifc")
