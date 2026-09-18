"""The Product geometry purge is one dict-delete, not 60,189 file scans.. ``ifcopenshell.file.remove`` nulls out every inverse reference
to the entity it deletes, which costs a scan of the whole file — so purging the
follower occurrences' orphaned bodies one at a time was O(removals x file
size): 60,189 removals at 6.3 ms on ``unitized-curtain-wall``, 381.8 s, about
72% of that model's compile.

The entities are now condemned by id and dropped in the pass the deduplicator
already makes over the parsed file.

**What must not change is the OUTPUT.** WR11 is the reason the purge exists (an
``IfcShapeRepresentation`` used by no product shape, no representation map and
no shape aspect is a schema error), and ``ifcopenshell.validate`` does NOT
report WR11 — so the corpus conformance gate cannot catch a regression here.
``test_product_geometry_sharing.py`` counts the owners itself; these tests
cover the drop mechanism.
"""
from __future__ import annotations

import pytest

from lite_step.ifc.deduplicator import _drop_purged, deduplicate_ifc_step

MINIMAL = """ISO-10303-21;
HEADER;
ENDSEC;
DATA;
#1=IFCCARTESIANPOINT((0.,0.,0.));
#2=IFCDIRECTION((0.,0.,1.));
#3=IFCAXIS2PLACEMENT3D(#1,#2,$);
#4=IFCCARTESIANPOINT((5.,0.,0.));
ENDSEC;
END-ISO-10303-21;
"""


class TestTheDrop:

    def test_condemned_ids_are_gone_and_the_rest_survive(self):
        out, _stats = deduplicate_ifc_step(MINIMAL, drop_ids={4})
        assert "#4=" not in out
        assert "#3=" in out and "#1=" in out

    def test_dropping_nothing_changes_nothing(self):
        untouched, _ = deduplicate_ifc_step(MINIMAL)
        dropped_none, _ = deduplicate_ifc_step(MINIMAL, drop_ids=set())
        assert untouched == dropped_none

    def test_an_id_that_is_not_in_the_file_is_not_an_error(self):
        """The condemned set is collected during generation and the file is
        rewritten afterwards; an id that has already gone (deduplicated into a
        twin, say) is ordinary, not a bug."""
        out, _ = deduplicate_ifc_step(MINIMAL, drop_ids={4, 99999})
        assert "#4=" not in out

    def test_the_purge_survives_a_file_with_nothing_to_deduplicate(self):
        """The early return that skips the rewrite when no duplicates are found
        must not return the ORIGINAL text. Reached with a non-empty purge set,
        that resurrects every condemned entity — silently, because the
        file still parses and still renders."""
        out, _ = deduplicate_ifc_step(MINIMAL, drop_ids={4})
        assert "#4=" not in out, (
            "the no-duplicates early return handed back the original text")


class TestTheClosureAssertion:
    """``purgeable_after_sharing`` runs a fixpoint that drops any candidate
    with a referrer outside the set, so what it returns is reference-closed —
    which is what makes deleting outright equivalent to ``file.remove``'s
    null-out-the-references behaviour.

    That is a property of another module, so this one checks rather than
    trusts: a dangling ``#id`` still PARSES and still renders, so nothing
    downstream would report it.
    """

    def test_purging_a_referenced_entity_raises(self):
        with pytest.raises(ValueError, match="not reference-closed"):
            # #3 still points at #1.
            deduplicate_ifc_step(MINIMAL, drop_ids={1})

    def test_purging_the_referrer_with_it_is_fine(self):
        out, _ = deduplicate_ifc_step(MINIMAL, drop_ids={1, 3})
        assert "#1=" not in out and "#3=" not in out
        assert "#2=" in out

    def test_the_message_names_a_culprit(self):
        """A closure violation is a bug in the purge SET, one module away. An
        error that does not say which entity still points where sends the
        reader to the wrong file."""
        with pytest.raises(ValueError) as exc:
            deduplicate_ifc_step(MINIMAL, drop_ids={1})
        assert "3" in str(exc.value)


class TestDropPurgedDirectly:

    def test_it_reports_how_many_it_dropped(self):
        entities = {1: ("IFCFOO", ""), 2: ("IFCBAR", "")}
        assert _drop_purged(entities, {1}) == 1
        assert set(entities) == {2}

    def test_none_and_empty_are_both_no_ops(self):
        entities = {1: ("IFCFOO", "")}
        assert _drop_purged(entities, None) == 0
        assert _drop_purged(entities, set()) == 0
        assert set(entities) == {1}
