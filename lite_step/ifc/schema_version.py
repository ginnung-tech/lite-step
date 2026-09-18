"""The one place the IFC output schema is declared.

It is passed to ``ifcopenshell.file(schema=...)``, which writes
``FILE_SCHEMA (('IFC4X3_ADD2'))``, and stamped verbatim into the STEP
header.

IFC4X3_ADD2 is ISO 16739-1:2024 — the published IFC4.3. The DSL's semantic
vocabulary (``ELEMENT_IFC43_CLASSES``) emits natively under it; before the
flip those classes rode IFC4 carriers (IfcGeographicElement / proxy with
ObjectType truth). Old IFC4 files (stored bases, patch-mode inputs) still
open fine — ifcopenshell reads any schema; only OUTPUT is pinned here.
"""

#: Passed to ``ifcopenshell.file(schema=...)`` and stamped into the
#: ``FILE_SCHEMA`` header.
IFC_OUTPUT_SCHEMA = "IFC4X3_ADD2"
