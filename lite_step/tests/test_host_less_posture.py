"""What a HOST-LESS ``lite_step`` refuses on its own — and what it does not.

The dunder-escape cases moved out of ``test_restricted_builtins.py``, which
asserted ``error_type == "UnsafeCodeError"``, which only Layer 1 raises — and
Layer 1 belongs to the HOST, seated through ``lite_step.compiler.sandbox``.
They passed in the tree they came from for one reason: that tree also shipped
a host. They belong on the side that supplies the guard, not here.

**This file is the other half, and it had never been written down.** It states
what the package refuses with an EMPTY seat, which is the security posture of
lite_step shipped standalone. That is not a footnote — it is the thing a reader
would otherwise have to discover by being wrong about it.

Found by running this suite standalone, where three tests failed because there
was no host to raise the error they expected. An import-level boundary check
does not catch that: those tests never imported a host, they depended on one
THROUGH the seat.
"""
from __future__ import annotations

import pytest

from lite_step.compiler.executor import execute_lite_step_script


@pytest.fixture()
def host_less(monkeypatch):
    """Clear the seat for the duration of one test.

    Explicit rather than assumed: a host may be importable and something earlier
    in the session may already have seated it, so a test that merely declined to
    install would be testing whatever ran before it.
    """
    from lite_step.compiler import sandbox

    monkeypatch.setattr(sandbox, "_INSTALLED", sandbox.SandboxHooks())


class TestWhatSurvivesExtraction:

    def test_a_blocked_import_is_still_refused(self, host_less):
        """The import allowlist is lite_step's OWN — a curated namespace, not
        the AST guard — so it survives having no host. Enforced at exec time
        rather than before it, which is later and still a refusal."""
        outcome = execute_lite_step_script(
            "import socket\nresult = Project(name='X')\n")
        assert not outcome.success
        assert "socket" in (outcome.error or "")

    def test_the_model_still_compiles(self, host_less):
        """The refusals must not cost the ordinary path. A package that is
        safe because nothing works is not the claim."""
        outcome = execute_lite_step_script(
            "result = Project(name='X')\n"
            "_v = getattr(result, 'name', None)\n"
            "assert _v == 'X'\n")
        assert outcome.success, outcome.error

    def test_setattr_is_refused_by_the_NAMESPACE_not_the_guard(self, host_less):
        """Measured, and it corrected this file's first draft.

        ``setattr`` was grouped with the two dunder escapes below as
        "guard-only". It is not: it is absent from
        ``get_restricted_builtins()`` entirely, so a host-less lite_step
        refuses it by NameError — no AST pass involved. Mutation is simply not
        in the whitelist, which is a stronger guarantee than a heuristic scan
        and one that survives extraction.
        """
        outcome = execute_lite_step_script(
            "result = Project(name='X')\nsetattr(result, 'name', 'Y')\n")
        assert not outcome.success
        assert "setattr" in (outcome.error or "")


class TestWhatDoesNot:
    """Uncomfortable, and the point of writing it down."""

    # NOT `setattr` — that one the namespace refuses on its own; see above.
    # These two are READS through a permitted builtin, which is exactly the
    # gap the AST guard's dunder blocklist exists to cover.
    @pytest.mark.parametrize("label,script", [
        ("getattr_globals_literal",
         "result = Project(name='X')\n_g = getattr(result, '__globals__', None)\n"),
        ("subclasses_attribute",
         "result = Project(name='X')\n_s = type(result).__subclasses__()\n"),
    ])
    def test_dunder_escapes_execute_without_a_host(self, label, script,
                                                   host_less):
        """With no sandbox seated, these three RUN.

        The AST guard is what refuses them, and a host-less lite_step has none.
        So a deployment executing UNTRUSTED DSL must seat one via
        ``set_sandbox()`` and set ``LITE_STEP_REQUIRE_SANDBOX=1``, which turns
        an empty seat into a loud refusal instead of this.

        Asserted rather than documented, because a README sentence cannot fail.
        If the namespace ever grows a guard of its own, this breaks and someone
        deletes it with evidence in hand — which is the right way to learn that
        the posture changed.
        """
        outcome = execute_lite_step_script(script)
        assert outcome.success, (
            f"{label} was refused with an empty seat — the namespace gained a "
            f"guard this file says it does not have. Good news; update the "
            f"docstring and the README rather than deleting the assertion.")
