"""``base_version_refusal`` — the one decision both continuation paths read.

``compile_main`` (adjacent ``base_ifc``) and ``patch_main`` (a base passed in
memory) each ask this function whether a stored base may be continued. The
path-level behaviour is exercised in ``test_compile_main`` and
``test_patch_executor``; this file pins the DECISION and the MESSAGE, so a
refusal that stops naming both versions, or a comparison that inverts, fails
here by name rather than somewhere downstream.
"""
import pytest

import lite_step.ifc.embedder as emb
from lite_step.compiler.patch_executor import patch_main
from lite_step.ifc.embedder import LitestepMeta, base_version_refusal


def _meta(version: str) -> LitestepMeta:
    return LitestepMeta(source="", manifest={}, version=version, hash="", hash_valid=True)


def _older() -> str:
    return str(int(emb.LITESTEP_META_VERSION) - 1)


def _newer() -> str:
    return str(int(emb.LITESTEP_META_VERSION) + 1)


def test_current_version_is_accepted():
    assert base_version_refusal(_meta(emb.LITESTEP_META_VERSION)) is None


def test_no_meta_is_accepted_as_a_cold_import():
    # A third-party IFC carries no LITESTEP_META: not a stale model.
    assert base_version_refusal(None) is None


@pytest.mark.parametrize("version", [_older(), _newer(), "1", "99"])
def test_any_other_version_is_refused_naming_both_versions_and_the_way_out(version):
    msg = base_version_refusal(_meta(version), base_name="The base IFC (x.ifc)")
    assert msg is not None
    assert "The base IFC (x.ifc)" in msg
    assert f"version {version};" in msg  # what the base carries
    assert f"writes version {emb.LITESTEP_META_VERSION}" in msg  # what this compiler wants
    assert "start a fresh build" in msg  # what the author does next


def test_patch_path_names_the_current_version_too(monkeypatch):
    """The path-level refusal must carry the same actionable text, not a
    paraphrase: it is what the author reads."""
    from lite_step.tests.test_patch_executor import _base_stamped, DESIRED_SOURCE

    base = _base_stamped(_older(), monkeypatch)
    result = patch_main(base, DESIRED_SOURCE)
    assert result.success is False
    assert f"version {_older()};" in result.error
    assert f"writes version {emb.LITESTEP_META_VERSION}" in result.error
