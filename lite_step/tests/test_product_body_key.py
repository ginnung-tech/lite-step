"""Product sharing grouped by a key decided before emission (``occurrence_body_key``).

Grouping Product occurrences by fingerprinting their emitted geometry read 13.5
million values on a 52-storey tower. The key groups them from the DSL instead,
and the fingerprint only checks it. What is pinned is that the key can never
make the output worse than the fingerprint did: a WRONG key falls back to the
fingerprint (correct bodies, a warning), and an over-cautious key loses no
sharing.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("ifcopenshell")

from lite_step.ifc import product_types  # noqa: E402
from lite_step.tests.test_product_geometry_sharing import (  # noqa: E402
    PLACEMENTS, anchored_facade, compile_ifc, model_of, profile_dims, spread_units,
)


def _maps(model):
    return sum(len(t.RepresentationMaps or ()) for t in model.by_type("IfcTypeProduct"))


def test_occurrences_placed_apart_get_one_key(monkeypatch):
    seen = []
    real = product_types.occurrence_body_key
    monkeypatch.setattr(product_types, "occurrence_body_key", lambda e: seen.append(real(e)) or seen[-1])
    model = model_of(compile_ifc(spread_units(), "ifcopenshell"))
    assert len(seen) == len(PLACEMENTS) and len(set(seen)) == 1
    assert _maps(model) == 5


def test_a_wrong_key_falls_back_to_the_fingerprint_and_says_so(monkeypatch, caplog):
    """Every occurrence given one key, though the facade has two orientations.

    The south and east windows alternate on this facade, so a guard that only
    fingerprints a group's first and last member passes them (it did): the
    extents check must catch it.
    """
    monkeypatch.setattr(product_types, "occurrence_body_key", lambda e: "same")
    with caplog.at_level(logging.WARNING, logger="lite_step.ifc.product_types"):
        model = model_of(compile_ifc(anchored_facade(), "ifcopenshell"))
    dims = profile_dims(model)
    assert (1.4, 0.06) in dims and (0.06, 1.4) in dims, "both orientations kept their own body"
    assert any("shares a body key with others but not their extents" in r.getMessage() for r in caplog.records)


def test_an_over_cautious_key_loses_no_sharing(monkeypatch):
    """A key unique to every occurrence: the fingerprints of the groups' first members merge them back."""
    counter = iter(range(10_000))
    monkeypatch.setattr(product_types, "occurrence_body_key", lambda e: str(next(counter)))
    model = model_of(compile_ifc(spread_units(), "ifcopenshell"))
    assert _maps(model) == 5
    assert len(model.by_type("IfcMappedItem")) == 5 * len(PLACEMENTS)


def test_without_keys_the_plan_is_the_fingerprints():
    """plan_shared_geometry called the old way still groups by fingerprint alone."""
    import inspect

    assert inspect.signature(product_types.plan_shared_geometry).parameters["keys"].default is None
