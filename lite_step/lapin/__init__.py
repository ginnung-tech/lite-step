"""Lite-STEP models as Lapin operations.

Lapin (https://github.com/Ciuntuc-BIM-Studio/lapin, MIT) is a desktop BIM
modeller whose engine takes commands over a local WebSocket. This package
holds the part of the adapter that needs no Lapin: :mod:`.ops` reads a
compiled IFC and says which Lapin operation makes each element.

Applying those operations to a running Lapin lives in ``tools/lapin`` (it
needs the ``lapinbim`` package, which the compiler does not).
"""

from .ops import FORMAT, VERSION, LapinOps, Op, ops_from_ifc, ops_from_json

__all__ = ["FORMAT", "VERSION", "LapinOps", "Op", "ops_from_ifc", "ops_from_json"]
