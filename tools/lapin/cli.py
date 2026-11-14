"""``lite-step lapin …``: Lite-STEP models in a running Lapin.

Lapin (https://github.com/Ciuntuc-BIM-Studio/lapin, MIT) is a desktop BIM
modeller. This tool sits beside the Blender and VS Code tools: it talks to
another program on this computer, which the compiler itself never does.

    lite-step lapin ops model.py            # what each element becomes, no Lapin needed
    lite-step lapin ops model.py -o ops.json

The mapping (which element becomes an editable box, which a body) is
``lite_step/lapin/ops.py``; this file only compiles, calls it and prints.
"""

from __future__ import annotations

import argparse
import collections
import runpy
import sys
import tempfile
from pathlib import Path


def load_ops(source: Path):
    """The operations for a model file (compiled first) or for an IFC already compiled."""
    from lite_step.lapin import ops_from_ifc

    if source.suffix.lower() == ".ifc":
        return ops_from_ifc(source)

    from lite_step.compiler.executor import compile_main

    # Not "__main__": the model's own compile_main() tail must not write beside the source.
    namespace = runpy.run_path(str(source), run_name="__lite_step__")
    result = namespace.get("result")
    if result is None:
        raise SystemExit(f"error: {source} did not define module-scope `result`.")
    with tempfile.TemporaryDirectory(prefix="lite-step-lapin-") as tmp:
        ifc = Path(tmp) / "model.ifc"
        compile_main(result=result, source_path=str(source), output_name=str(ifc))
        return ops_from_ifc(ifc)


def print_summary(ops, out=sys.stdout, notes: bool = False) -> None:
    s = ops.summary()
    plain = s["primitive"] - s["primitive_carved"]
    print(f"{s['elements']} elements for Lapin:", file=out)
    print(f"  {plain:6}  editable boxes and cylinders", file=out)
    print(f"  {s['primitive_carved']:6}  editable, cut by other elements (live in Lapin)", file=out)
    print(f"  {s['body']:6}  bodies (shape set by the script only)", file=out)
    for name, elevation in ops.storeys:
        print(f"  storey {name or '(unnamed)'} at {elevation:g} m", file=out)
    for path, reason in ops.skipped:
        print(f"  LEFT OUT {path}: {reason}", file=out)
    if notes:
        for note in ops.notes:
            print(f"  note: {note}", file=out)
    elif ops.notes:
        reasons = collections.Counter(n.split(" because ")[-1] if " because " in n else n.split(": ", 1)[-1]
                                      for n in ops.notes)
        for reason, count in reasons.most_common(5):
            print(f"  {count:6}  {reason}", file=out)
        print("  (--notes lists every one)", file=out)


def cmd_ops(args) -> int:
    ops = load_ops(Path(args.source).resolve())
    if args.output:
        Path(args.output).write_text(ops.to_json(), encoding="utf-8")
        print_summary(ops, out=sys.stderr, notes=args.notes)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print_summary(ops, notes=args.notes)
    return 1 if ops.skipped else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lite-step lapin", description="Lite-STEP models in a running Lapin.")
    sub = parser.add_subparsers(dest="command", required=True)

    ops = sub.add_parser("ops", help="what each element becomes in Lapin; no Lapin needed")
    ops.add_argument("source", help="a model file (.py, compiled first) or a compiled .ifc")
    ops.add_argument("-o", "--output", help="write the operations as JSON here")
    ops.add_argument("--notes", action="store_true", help="list every note, not the five commonest")
    ops.set_defaults(run=cmd_ops)

    args = parser.parse_args(argv)
    if not Path(args.source).is_file():
        parser.error(f"not a file: {args.source}")
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
