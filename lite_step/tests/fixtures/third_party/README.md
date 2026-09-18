# Third-party IFC corpus

Scaffold directory for real-vendor IFC exports (Revit, ArchiCAD,
buildingSMART reference models) used as fixtures for cold-import
round-trip and coverage tests in [`test_third_party_corpus.py`](../../test_third_party_corpus.py).

Currently empty. The corresponding tests synthesize vendor-style IFCs
inline via `ifcopenshell.file(schema="IFC4")` + `create_entity` calls,
so they have zero external dependency and run in CI without network.

When you drop a real-vendor `.ifc` file here, also add a matching
`pytest` case in `test_third_party_corpus.py` that loads it via
`Path(__file__).parent / "fixtures" / "third_party" / "<file>.ifc"` and
asserts the cold-import coverage report shape you expect for that
vendor's output. See the synthesised cases as the template.

Licensing notes:
- `Duplex_A_20110907.ifc` and friends from buildingSMART are
  Apache 2.0 compatible (CC-licensed reference models).
- Revit / ArchiCAD-exported `.ifc` files generated from in-house
  projects are case-by-case — check with the file owner before
  committing.
