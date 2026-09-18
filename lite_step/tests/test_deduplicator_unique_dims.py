"""Deduplicator × unique_dims contract (the missing-cladding fix, v19).

Three properties keep the geometry-signature noise alive end-to-end:

1. ``IFCRECTANGLEPROFILEDEF`` floats are EXEMPT from the deduplicator's
   0.1 mm rounding (``DEDUP_NO_FLOAT_ROUND_TYPES``) — the 0.02 mm
   ``unique_dims`` nudge must reach the serialized file, or mirror facade
   leaves re-collapse to byte-identical dims and the SPA render worker's
   instancing race can drop one again.
2. Genuinely identical rect profiles STILL dedup (full-precision floats
   serialize byte-identically, so the merge sees them as equal).
3. Other safe types (points, directions) keep the 0.1 mm rounding —
   the exemption is surgical.

Plus the EntityCache side: ``_claim_unique_dims`` is deterministic and
same-name repeats share the claim (slices within one wall reuse one
profile; only cross-name collisions nudge).
"""

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")

from lite_step.ifc.deduplicator import (
    DEDUP_NO_FLOAT_ROUND_TYPES,
    DEDUP_SAFE_TYPES,
    deduplicate_ifc_step,
)
from lite_step.ifc.entity_cache import EntityCache


_HEADER = (
    "ISO-10303-21;\n"
    "HEADER;\n"
    "FILE_DESCRIPTION((''),'2;1');\n"
    "FILE_NAME('','',(''),(''),'','','');\n"
    "FILE_SCHEMA(('IFC4'));\n"
    "ENDSEC;\n"
    "DATA;\n"
)
_FOOTER = "ENDSEC;\nEND-ISO-10303-21;\n"


def _step(*entity_lines: str) -> str:
    return _HEADER + "\n".join(entity_lines) + "\n" + _FOOTER


class TestDeduplicatorExemption:
    def test_rect_profile_is_dedup_safe_but_round_exempt(self):
        assert "IFCRECTANGLEPROFILEDEF" in DEDUP_SAFE_TYPES
        assert "IFCRECTANGLEPROFILEDEF" in DEDUP_NO_FLOAT_ROUND_TYPES

    def test_nudged_profile_dims_survive_dedup(self):
        """A 0.02 mm dims difference (the unique_dims nudge) must NOT be
        rounded away and merged — the whole point of the exemption."""
        content = _step(
            "#1=IFCRECTANGLEPROFILEDEF(.AREA.,'wall:facade_south',$,12.656,0.108);",
            "#2=IFCRECTANGLEPROFILEDEF(.AREA.,'wall:facade_north',$,12.65602,0.108);",
        )
        result, stats = deduplicate_ifc_step(content)
        assert stats["entities_removed"] == 0
        assert "12.656," in result
        assert "12.65602," in result

    def test_identical_anon_profiles_still_merge(self):
        """Property 2: the exemption removes rounding, not deduplication —
        byte-identical profiles still collapse to one entity."""
        content = _step(
            "#1=IFCRECTANGLEPROFILEDEF(.AREA.,$,$,0.045,0.195);",
            "#2=IFCRECTANGLEPROFILEDEF(.AREA.,$,$,0.045,0.195);",
        )
        _result, stats = deduplicate_ifc_step(content)
        assert stats["entities_removed"] == 1

    def test_points_still_round_and_merge(self):
        """Property 3: a sub-0.1mm difference on a CARTESIANPOINT still
        rounds + merges — the exemption is scoped to rect profiles."""
        content = _step(
            "#1=IFCCARTESIANPOINT((1.00001,2.,3.));",
            "#2=IFCCARTESIANPOINT((1.,2.,3.));",
        )
        _result, stats = deduplicate_ifc_step(content)
        assert stats["entities_removed"] == 1


class TestUniqueDimsClaim:
    def _cache(self):
        return EntityCache(ifcopenshell.file(schema="IFC4"))

    def test_cross_name_collision_nudges_by_step(self):
        c = self._cache()
        p1 = c.get_or_create_rect_profile(x_dim=12.656, y_dim=0.108,
                                          name="wall:a", unique_dims=True)
        p2 = c.get_or_create_rect_profile(x_dim=12.656, y_dim=0.108,
                                          name="wall:b", unique_dims=True)
        assert p1.XDim == 12.656
        assert p2.XDim == pytest.approx(12.656 + EntityCache._UNIQUE_DIMS_STEP)
        assert p1.YDim == p2.YDim == 0.108

    def test_same_name_repeat_reuses_claim_and_entity(self):
        """Slices within one wall share one profile — no self-nudge."""
        c = self._cache()
        p1 = c.get_or_create_rect_profile(x_dim=0.108, y_dim=9.656,
                                          name="wall:w", unique_dims=True)
        p2 = c.get_or_create_rect_profile(x_dim=0.108, y_dim=9.656,
                                          name="wall:w", unique_dims=True)
        assert p1 is p2

    def test_three_way_collision_all_distinct(self):
        c = self._cache()
        dims = set()
        for n in ("wall:a", "wall:b", "wall:c"):
            p = c.get_or_create_rect_profile(x_dim=0.19, y_dim=9.656,
                                             name=n, unique_dims=True)
            dims.add((p.XDim, p.YDim))
        assert len(dims) == 3

    def test_step_survives_float32_at_facade_scale(self):
        """The nudge must stay visible in float32 half-dim vertex coords at
        facade coordinate scale (~6.3 m) — the SPA mesh's actual storage."""
        import struct
        f32 = lambda v: struct.unpack('f', struct.pack('f', v))[0]
        x = 12.656
        nudged = x + EntityCache._UNIQUE_DIMS_STEP
        assert f32(x / 2) != f32(nudged / 2)
