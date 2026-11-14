"""``lite-step lapin …``: Lite-STEP models in a running Lapin.

Lapin (https://github.com/Ciuntuc-BIM-Studio/lapin, MIT) is a desktop BIM
modeller. This tool sits beside the Blender and VS Code tools: it talks to
another program on this computer, which the compiler itself never does.

    lite-step lapin status [model.py]       # the Lapins open, and what this model owns there
    lite-step lapin push model.py           # compile, then make Lapin match (one undo step)
    lite-step lapin push model.py --watch   # again at every save
    lite-step lapin diff model.py           # what a push would do; changes nothing
    lite-step lapin list model.py           # this model's elements in Lapin, and their state
    lite-step lapin fetch -o context.ifc    # the open Lapin project as IFC, to build against
    lite-step lapin remove model.py         # delete what this model owns in Lapin, nothing else
    lite-step lapin ops model.py            # what each element becomes; no Lapin needed

Talking to Lapin needs its ``lapinbim`` package (MIT, standard library only).
The mapping is ``lite_step/lapin/ops.py``; the delta is ``lapin_sync.py`` here.
"""

from __future__ import annotations

import argparse
import base64
import collections
import runpy
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lapin_sync  # noqa: E402

LAPINBIM_HINT = (
    "error: talking to Lapin needs its `lapinbim` package.\n"
    "  pip install lapinbim\n"
    "or, until it is on PyPI, from a Lapin checkout:\n"
    "  pip install <lapin checkout>/python"
)


class Failed(Exception):
    """Something to tell the user in one line; no traceback."""


# --------------------------------------------------------------------------- the model


def load_ops(source: Path):
    """The operations for a model file (compiled first) or for an IFC already compiled."""
    from lite_step.lapin import ops_from_ifc

    if source.suffix.lower() == ".ifc":
        return ops_from_ifc(source)

    from lite_step.compiler.executor import LiteStepCompileError, compile_main

    # Not "__main__": the model's own compile_main() tail must not write beside the source.
    namespace = runpy.run_path(str(source), run_name="__lite_step__")
    result = namespace.get("result")
    if result is None:
        raise Failed(f"{source} did not define module-scope `result`.")
    with tempfile.TemporaryDirectory(prefix="lite-step-lapin-") as tmp:
        ifc = Path(tmp) / "model.ifc"
        try:
            compile_main(result=result, source_path=str(source), output_name=str(ifc))
        except LiteStepCompileError:
            raise Failed(f"{source} did not compile (the errors are above).") from None
        return ops_from_ifc(ifc)


def model_id(args) -> str:
    """What marks this model's elements in Lapin: ``--id``, else the file's name without its ending."""
    return args.id or Path(args.source).stem


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


# --------------------------------------------------------------------------- Lapin


def lapinbim():
    try:
        import lapinbim as module
    except ImportError:
        raise Failed(LAPINBIM_HINT[len("error: "):]) from None
    return module


def connect(args):
    """The Lapin to work in: ``--port``, ``--project``, else the only one open."""
    module = lapinbim()
    try:
        return module.Lapin(args.port, project=args.project)
    except (module.LapinError, LookupError, OSError) as e:
        raise Failed(str(e)) from None


def print_plan(the_plan, out=sys.stdout, verbose: bool = False) -> None:
    c = the_plan.counts()
    print(f"create {c['create']}, change {c['update']}, make again {c['replace']}, "
          f"delete {c['delete']}, untouched {c['keep']}", file=out)
    for path in the_plan.hand_edited:
        print(f"  changed in Lapin, the script's version is put back: {path}", file=out)
    if verbose:
        for label, paths in (("create", [o.path for o in the_plan.create]),
                             ("change", [o.path for o, _ in the_plan.update]),
                             ("make again", [o.path for o, _ in the_plan.replace]),
                             ("delete", [h.path for h in the_plan.delete])):
            for path in paths:
                print(f"  {label}: {path}", file=out)


def push_once(args, client) -> int:
    ops = load_ops(Path(args.source).resolve())
    the_plan = lapin_sync.plan(ops, lapin_sync.read_owned(client, model_id(args)))
    for path, reason in ops.skipped:
        print(f"LEFT OUT {path}: {reason}", file=sys.stderr)
    print_plan(the_plan, verbose=args.verbose or args.dry_run)
    if args.dry_run:
        print("(dry run: nothing was changed)")
        return 0
    started = time.time()
    lapin_sync.apply(client, ops, the_plan, model_id(args), lapinbim().global_id)
    if the_plan.empty:
        print("Lapin already matches the model.")
    else:
        print(f"done in {time.time() - started:.1f} s; Ctrl+Z in Lapin undoes this push.")
    return 1 if ops.skipped else 0


def cmd_push(args) -> int:
    client = connect(args)
    with client:
        code = push_once(args, client)
        if not args.watch:
            return code
        source = Path(args.source).resolve()
        seen = source.stat().st_mtime
        print(f"watching {source.name}; Ctrl+C stops.")
        try:
            while True:
                time.sleep(0.5)
                now = source.stat().st_mtime
                if now == seen:
                    continue
                seen = now
                try:
                    push_once(args, client)
                except Failed as e:
                    print(f"error: {e}", file=sys.stderr)
                except Exception as e:  # noqa: BLE001 - a bad save must not end the watch
                    print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        except KeyboardInterrupt:
            return 0


def cmd_diff(args) -> int:
    args.dry_run, args.verbose, args.watch = True, True, False
    with connect(args) as client:
        return push_once(args, client)


def cmd_list(args) -> int:
    with connect(args) as client:
        owned = lapin_sync.read_owned(client, model_id(args))
        ops = load_ops(Path(args.source).resolve())
    the_plan = lapin_sync.plan(ops, owned)
    state = {o.path: "not in Lapin yet" for o in the_plan.create}
    state.update({o.path: "changed in the script" for o, _ in [*the_plan.update, *the_plan.replace]})
    state.update({o.path: "in sync" for o, _ in the_plan.keep})
    state.update({h.path: "gone from the script" for h in the_plan.delete})
    state.update({p: "changed in Lapin" for p in the_plan.hand_edited})
    kinds = {o.path: (o.kind, o.ifc_class, o.storey or "") for o in ops.ops}
    for path in sorted(state):
        kind, cls, storey = kinds.get(path, ("", "", ""))
        print(f"{state[path]:22} {kind:9} {cls:28} {storey:16} {path}")
    print(f"{len(owned)} elements of `{model_id(args)}` in Lapin; {len(ops.ops)} in the model.")
    return 0


def cmd_status(args) -> int:
    module = lapinbim()
    # A port named is looked at directly: an engine started with a port of its own is not found by looking.
    engines = [] if args.port else module.engines()
    if not engines and not args.port:
        print("No Lapin is open on this computer (start the app, or set LAPIN_PORT).")
        return 1
    for e in engines:
        print(f"Lapin on port {e['port']}: {e.get('productName') or e.get('product')} {e.get('version', '')}, "
              f"project {e.get('project') or '(not saved yet)'}")
    if len(engines) > 1 and not (args.port or args.project):
        print("Several are open: name one with --port or --project to see inside it.")
        return 0
    with connect(args) as client:
        if args.port:
            e = client.call("engine.hello", {})
            print(f"Lapin on port {args.port}: {e['productName']} {e['version']}, project {e['project'] or '(not saved yet)'}")
        for s in sorted((e for e in client.model if e.type == "IfcBuildingStorey"),
                        key=lambda s: float(s.attrs.get("elevation") or 0)):
            print(f"  storey {s.name} at {float(s.attrs.get('elevation') or 0):g} m")
        all_marks = lapin_sync.marks(client)
        if not all_marks:
            print("  no Lite-STEP elements in this project")
        for name, owned in sorted(all_marks.items()):
            edited = sum(1 for o in owned.values() if o.hand_edited)
            print(f"  model `{name}`: {len(owned)} elements" + (f", {edited} changed in Lapin" if edited else ""))
    return 0


def cmd_fetch(args) -> int:
    with connect(args) as client:
        answer = client.call("project.exportIfc", {})
    data = base64.b64decode(answer["data"])
    Path(args.output).write_bytes(data)
    print(f"wrote {args.output} ({len(data)} bytes): the Lapin project as IFC. "
          f"It is context to build against; nothing reads it back into Lapin.")
    return 0


def cmd_remove(args) -> int:
    with connect(args) as client:
        count = lapin_sync.remove(client, model_id(args))
    print(f"deleted {count} elements of `{model_id(args)}` from Lapin; Ctrl+Z there brings them back."
          if count else f"`{model_id(args)}` has no elements in Lapin.")
    return 0


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

    def command(name, run, help, source=None, lapin=True):
        p = sub.add_parser(name, help=help)
        if source:
            p.add_argument("source", nargs=None if source == "required" else "?",
                           help="a model file (.py, compiled first) or a compiled .ifc")
            p.add_argument("--id", help="the name this model's elements are marked with in Lapin "
                                        "(default: the file's name without its ending)")
        if lapin:
            p.add_argument("--port", type=int, help="the Lapin on this port")
            p.add_argument("--project", help="the Lapin with this project open (its file name or a part of its path)")
        p.set_defaults(run=run)
        return p

    push = command("push", cmd_push, "compile, then make Lapin match the model (one undo step)", "required")
    push.add_argument("--watch", action="store_true", help="push again whenever the file is saved")
    push.add_argument("--dry-run", action="store_true", help="say what would change, change nothing")
    push.add_argument("-v", "--verbose", action="store_true", help="name every element touched")
    command("diff", cmd_diff, "what a push would do; changes nothing", "required")
    command("list", cmd_list, "this model's elements in Lapin and their state", "required")
    command("status", cmd_status, "the Lapins open on this computer and what they hold")
    fetch = command("fetch", cmd_fetch, "the open Lapin project as IFC, to build against")
    fetch.add_argument("-o", "--output", required=True, help="the IFC file to write")
    command("remove", cmd_remove, "delete what this model owns in Lapin; nothing else", "required")
    ops = command("ops", cmd_ops, "what each element becomes in Lapin; no Lapin needed", "required", lapin=False)
    ops.add_argument("-o", "--output", help="write the operations as JSON here")
    ops.add_argument("--notes", action="store_true", help="list every note, not the five commonest")

    args = parser.parse_args(argv)
    if getattr(args, "source", None) and not Path(args.source).is_file():
        parser.error(f"not a file: {args.source}")
    try:
        return args.run(args)
    except Failed as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
