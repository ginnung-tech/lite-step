"""The seat a HOST installs its sandbox into.

Lite-STEP executes author-written Python. Who is allowed to run that, and
under what containment, is a decision belonging to whatever embeds the DSL —
not to the DSL. Until now ``executor`` reached for the host's guard module by name
inside a ``try/except ImportError``, which reads as an optional dependency and
is really the opposite: an integration that silently evaporates.

This inverts it. ``lite_step`` declares the shape it can call; a host installs
an implementation. Missing hooks are then a stated property of the deployment
rather than a swallowed import error.

**A host installs its hooks at import time**, so that by the time any
import. That placement is load-bearing rather than tidy: the only production
caller reaches ``execute_lite_step_script`` the seat is filled. A host that
defers installation until first use leaves a window where the guarded path
cannot be reached with an empty seat.
``tests/test_sandbox_seat.py`` asserts that caller set, because the guarantee
is exactly as strong as it remains true.

**A public Lite-STEP has no host and therefore no sandbox**, and that must be
a documented decision rather than a discovered one: it runs the author's own
code, so the prompt-injection threat model Layer 3 exists for does not apply.
Anyone executing UNTRUSTED DSL has to supply their own containment through
this seat. What survives regardless is lite_step's own restricted namespace —
a blocked ``import socket`` is still refused, at exec time rather than before
it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Type


@dataclass(frozen=True)
class SandboxHooks:
    """What a host may supply. Every field is optional and independent.

    Frozen because a half-installed sandbox is worse than none: a caller that
    mutated one field would leave the others describing a different host.
    """

    #: Pre-exec static check. Raises ``unsafe_code_error`` to reject.
    check_code: Optional[Callable[[str], None]] = None
    #: The exception ``check_code`` raises. Required WITH ``check_code``.
    unsafe_code_error: Optional[Type[BaseException]] = None
    #: Run the script in an isolated child. Signature is the host's.
    run_sandboxed: Optional[Callable[..., Any]] = None
    #: Whether the isolated-child path is switched on right now.
    subprocess_enabled: Optional[Callable[[], bool]] = None

    def __post_init__(self) -> None:
        if bool(self.check_code) != bool(self.unsafe_code_error):
            raise ValueError(
                "check_code and unsafe_code_error must be installed together "
                "— a checker whose rejection type is unknown would have its "
                "refusal caught as an ordinary error and reported as a "
                "compile failure, which reads as a broken model rather than "
                "a blocked one.")


_INSTALLED = SandboxHooks()


def set_sandbox(hooks: SandboxHooks) -> None:
    """Install (or, with a bare ``SandboxHooks()``, clear) the host's sandbox."""
    global _INSTALLED
    if not isinstance(hooks, SandboxHooks):
        raise TypeError(f"expected SandboxHooks, got {type(hooks).__name__}")
    _INSTALLED = hooks


def get_sandbox() -> SandboxHooks:
    """The installed hooks. Always a ``SandboxHooks`` — never ``None``, so
    callers read fields instead of branching on the container."""
    return _INSTALLED


#: A deployment that REQUIRES containment sets this. See ``assert_installed``.
REQUIRE_ENV = "LITE_STEP_REQUIRE_SANDBOX"


def assert_installed() -> None:
    """Refuse to execute unguarded where the deployment says containment is
    mandatory. No-op unless ``LITE_STEP_REQUIRE_SANDBOX=1``.

    This exists because the seat is strictly weaker than what it replaced, in
    one specific way. Importing the host's guard lazily INSIDE the call gives
every caller the guards whatever the import order. Installing
    at host-import time instead means a path that reaches
    ``execute_lite_step_script`` without the host ever being imported runs with
    an empty seat — no AST check, no subprocess isolation, and nothing raising.
    ``lite_step.transpiler`` is such a path today: it is reached through
    ``patch_main``, which has no in-repo production caller but is a documented
    public entry point.

    Relying on "every caller happens to sit under the host package" is the assumption
    that would fail silently and invisibly. So production states the
    requirement instead (``docker/Dockerfile`` sets the env) and gets a loud
    refusal, while a public Lite-STEP — which runs its author's own code and
    has no host — leaves it unset and is unaffected.
    """
    import os

    if os.environ.get(REQUIRE_ENV) != "1":
        return
    hooks = get_sandbox()
    if hooks.check_code is None and hooks.run_sandboxed is None:
        raise RuntimeError(
            f"{REQUIRE_ENV}=1 but no sandbox is installed — refusing to exec "
            f"untrusted source unguarded. The host must call "
            f"lite_step.compiler.sandbox.set_sandbox() before executing; in "
            f"a host, that typically happens at import time.")
