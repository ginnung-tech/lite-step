"""``lite-step lapin …``: a Lite-STEP model in a running Lapin.

Lapin (https://github.com/Ciuntuc-BIM-Studio/lapin, MIT) is a desktop BIM
modeller. This tool sits beside the Blender and VS Code tools: it talks to
another program on this computer, which the compiler itself never does.

    lite-step lapin status                 # the Lapins open, and the Lite-STEP models linked in them
    lite-step lapin push model.py          # compile to .lapin, then link into Lapin or reload the link
    lite-step lapin push model.py format=ifc  # link as IFC instead of .lapin
    lite-step lapin push model.py --watch  # again at every save
    lite-step lapin pull model.py          # write the parameters changed in Lapin into the script
    lite-step lapin fetch -o context.ifc   # the open Lapin project as IFC, to build against
    lite-step lapin remove model.py        # unlink the model from Lapin

**Parameters.** Every ``NAME = param(value, unit, label)`` in the script is
published to Lapin as a project parameter (Manage -> Project information), in
the group ``Lite-STEP <model>``. Changed there, it comes back: ``pull`` writes
it into the script, and ``push --watch`` does so by itself, then compiles and
reloads. Only the value in that one line changes. A parameter the script
computes (``param(WIDTH * 2, ...)``) is not published: it can only change in
the script.

A push hands the compiled IFC to Lapin's own importer as a **link**: Lapin
reads it fast, keeps shared shapes shared, recognises walls, slabs and doors,
and every element is there to select, read-only (the script owns it). The
next push reloads the link. Lite-STEP's GlobalIds are derived from the model,
so a reloaded element is the same element to Lapin: what the user dimensioned
or tagged in Lapin stays attached.

Talking to Lapin needs its ``lapinbim`` package (MIT, standard library only).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import os
import runpy
import sys
import time
from pathlib import Path

LAPINBIM_HINT = (
    "talking to Lapin needs its `lapinbim` package.\n"
    "  pip install lapinbim\n"
    "or, until it is on PyPI, from a Lapin checkout:\n"
    "  pip install <lapin checkout>/python"
)

#: How long Lapin may take to read a large file (a 10 MB IFC takes about 20 s).
IMPORT_TIMEOUT = 900


class Failed(Exception):
    """Something to tell the user in one line; no traceback."""


# --------------------------------------------------------------------------- the model


_LAPIN_EXT = ".lapin"
_IFC_EXT = ".ifc"


def ifc_of(source: Path) -> Path:
    """The IFC a push links: ``source`` itself, or the model compiled beside it (``<name>.ifc``)."""
    return source if source.suffix.lower() == _IFC_EXT else source.with_suffix(_IFC_EXT)


def lapin_of(source: Path) -> Path:
    """The `.lapin` package a push links: ``source`` itself, or ``<name>.lapin`` beside it."""
    return source if source.suffix.lower() == _LAPIN_EXT else source.with_suffix(_LAPIN_EXT)


def compile_lapin(source: Path, quiet: bool) -> Path:
    """Compile ``source`` (``.py`` or ``.ifc``) into ``<name>.lapin`` beside it."""
    target = lapin_of(source)
    if source.suffix.lower() == _LAPIN_EXT:
        return target

    if source.suffix.lower() == _IFC_EXT:
        from lite_step.compiler.lapin_package import build_lapin_package

        build_lapin_package(source, target, source_path=None)
        return target

    from lite_step.compiler.executor import LiteStepCompileError, compile_main

    started = time.time()
    print(f"compiling {source.name}…", flush=True)
    with _quiet(quiet):
        namespace = runpy.run_path(str(source), run_name="__lite_step__")
    result = namespace.get("result")
    if result is None:
        raise Failed(f"{source} did not define module-scope `result`.")
    try:
        with _quiet(quiet):
            compile_main(result=result, source_path=str(source), output_name=target.name, format="lapin")
    except LiteStepCompileError:
        raise Failed(f"{source.name} did not compile (run `lite-step {source.name}` to see why).") from None
    print(f"compiled {source.name} -> {target.name} in {time.time() - started:.0f} s")
    return target


def compile_model(source: Path, quiet: bool) -> Path:
    """The model file compiled to ``<name>.ifc`` beside it, as ``lite-step model.py -o <name>.ifc`` does."""
    target = ifc_of(source)
    suffix = source.suffix.lower()
    if suffix == _IFC_EXT:
        return target
    if suffix == _LAPIN_EXT:
        raise Failed(f"cannot push {source.name} with format=ifc (pass the .py or .ifc source, or omit format=ifc)")

    from lite_step.compiler.executor import LiteStepCompileError, compile_main

    started = time.time()
    print(f"compiling {source.name}…", flush=True)
    # Not "__main__": the model's own compile_main() tail must not write a second file.
    with _quiet(quiet):
        namespace = runpy.run_path(str(source), run_name="__lite_step__")
    result = namespace.get("result")
    if result is None:
        raise Failed(f"{source} did not define module-scope `result`.")
    try:
        with _quiet(quiet):
            compile_main(result=result, source_path=str(source), output_name=target.name)
    except LiteStepCompileError:
        raise Failed(f"{source.name} did not compile (run `lite-step {source.name}` to see why).") from None
    print(f"compiled {source.name} -> {target.name} in {time.time() - started:.0f} s")
    return target


@contextlib.contextmanager
def _quiet(quiet: bool):
    """The compiler's reports (tree, carves) kept off the screen unless asked for."""
    if not quiet:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink), \
            contextlib.redirect_stderr(sink):
        yield


# --------------------------------------------------------------------------- Lapin


def lapinbim():
    try:
        import lapinbim as module
    except ImportError:
        raise Failed(LAPINBIM_HINT) from None
    return module


def connect(args):
    """The Lapin to work in: ``--port``, ``--project``, else the only one open."""
    module = lapinbim()
    try:
        return module.Lapin(args.port, project=args.project)
    except (module.LapinError, LookupError, OSError) as e:
        raise Failed(str(e)) from None


def _same_file(a: str, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(str(b)))


def find_link(client, ifc: Path):
    """Lapin's link to this file, or ``None``. A link remembers the path it was made from."""
    for link in client.query("link.list") or []:
        path = link.get("path") if isinstance(link, dict) else getattr(link, "path", None)
        if path and _same_file(path, ifc):
            return link
    return None


def _field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


# --------------------------------------------------------------------------- parameters


def model_id(args) -> str:
    """What this model's parameters are grouped under in Lapin: the script's name without its ending."""
    return Path(args.source).stem


def _group(model: str) -> str:
    return f"Lite-STEP {model}"


def _lapin_params(client) -> dict:
    """Lapin's project parameters, ``{key: {key, label, type, group, value}}``."""
    info = client.query("project.info") or {}
    return {_field(c, "key"): c for c in (_field(info, "custom") or [])}


def _script_params(source: Path):
    from lite_step.params import read_params

    if source.suffix.lower() != ".py":
        return []
    return [p for p in read_params(source.read_text(encoding="utf-8")) if p.editable]


def _shown(p) -> dict:
    """A parameter as Lapin keeps it: its label carries the unit, its value is text."""
    label = p.label or p.name.replace("_", " ").capitalize()
    is_number = isinstance(p.value, (int, float)) and not isinstance(p.value, bool)
    return {"label": f"{label} ({p.unit})" if p.unit else label,
            "type": "number" if is_number else "text", "value": str(p.value)}


def _same_value(script_value, lapin_value) -> bool:
    from lite_step.params import coerce

    try:
        return coerce(script_value, lapin_value) == script_value
    except (TypeError, ValueError):
        return False


def publish_params(client, source: Path, model: str) -> None:
    """The script's parameters into Lapin's project parameters; those it no longer has removed.

    The script wins: a value changed in Lapin and not pulled is overwritten, and said so.
    """
    group, there = _group(model), _lapin_params(client)
    wanted = _script_params(source)
    for p in wanted:
        have = there.get(p.name)
        if have is not None and _field(have, "group") != group:
            print(f"parameter {p.name}: Lapin already has one in group {_field(have, 'group')!r}; not published",
                  file=sys.stderr)
            continue
        shown = _shown(p)
        if have is not None:
            same_value = _same_value(p.value, _field(have, "value"))
            same_rest = all(str(_field(have, k)) == shown[k] for k in ("label", "type"))
            if same_value and same_rest:
                continue
            if not same_value:
                print(f"parameter {p.name}: {_field(have, 'value')} in Lapin replaced by the script's {p.value} "
                      f"(`lite-step lapin pull` keeps a value changed in Lapin)")
        try:
            client.execute("project.setParameter", {"key": p.name, "group": group, **shown})
        except lapinbim().LapinError as e:
            # Lapin keeps some keys for its own project fields (a constant called NAME, say).
            print(f"parameter {p.name}: Lapin refused it ({e}); rename the constant to publish it",
                  file=sys.stderr)
    names = {p.name for p in wanted}
    for key, have in there.items():
        if _field(have, "group") == group and key not in names:
            client.execute("project.removeParameter", {"key": key})


def pull_params(client, source: Path, model: str) -> list:
    """The values changed in Lapin written into the script: ``[(name, old, new)]``, in file order.

    Writes the file only when something changed; each change is one literal.
    """
    from lite_step.params import read_params, write_param

    if source.suffix.lower() != ".py":
        return []
    group, there = _group(model), _lapin_params(client)
    text = source.read_text(encoding="utf-8")
    changes = []
    for p in read_params(text):
        have = there.get(p.name)
        if have is None or _field(have, "group") != group or not p.editable:
            continue
        value = _field(have, "value")
        if _same_value(p.value, value):
            continue
        try:
            text = write_param(text, p.name, value)
        except ValueError as e:
            print(f"parameter {p.name}: {value!r} from Lapin not written: {e}", file=sys.stderr)
            continue
        changes.append((p.name, p.value, value))
    if changes:
        with open(source, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    return changes


def _resolved_format(source: Path, requested: str | None) -> str:
    if requested is not None:
        return requested
    return "ifc" if source.suffix.lower() == _IFC_EXT else "lapin"


def push_once(args, client) -> int:
    source = Path(args.source).resolve()
    fmt = _resolved_format(source, args.format)
    started = time.time()
    if fmt == "ifc":
        target = compile_model(source, quiet=not args.verbose)
        other_target = lapin_of(source)
        method = "project.importIfc"
        call_params = {"path": str(target), "mode": "link"}
    else:
        target = compile_lapin(source, quiet=not args.verbose)
        other_target = ifc_of(source)
        method = "project.linkLapin"
        call_params = {"path": str(target)}

    link = find_link(client, target)
    other_link = find_link(client, other_target)
    if other_link is not None:
        print(
            f"warning: Lapin also has {other_target.name} linked; run `lite-step lapin remove {source.name}` "
            "or remove the old link to avoid duplicate geometry",
            file=sys.stderr,
        )
    try:
        if link is None:
            answer = client.call(method, call_params, timeout=IMPORT_TIMEOUT)
            link_id = _field(answer, "linkId")
            verb = "linked"
        else:
            link_id = _field(link, "id")
            client.call("project.reloadLink", {"id": link_id}, timeout=IMPORT_TIMEOUT)
            verb = "reloaded"
    except lapinbim().LapinError as e:
        raise Failed(f"Lapin refused the file: {e}") from None

    now = find_link(client, target) or {}
    print(f"{verb} {target.name} in Lapin in {time.time() - started:.0f} s: "
          f"{_field(now, 'elements', '?')} elements on {_field(now, 'storeys', '?')} storeys (link {link_id})")
    if source.suffix.lower() == ".py":
        publish_params(client, source, model_id(args))
    return 0


def cmd_push(args) -> int:
    with connect(args) as client:
        code = push_once(args, client)
        if not args.watch:
            return code
        source = Path(args.source).resolve()
        seen = source.stat().st_mtime
        print(f"watching {source.name} and its parameters in Lapin; Ctrl+C stops (a plain push exits by itself).")
        try:
            while True:
                time.sleep(0.5)
                now = source.stat().st_mtime
                if now == seen:
                    # The script is as pushed: a parameter changed in Lapin is written into it,
                    # which changes the file, which the next turn compiles and reloads.
                    try:
                        for name, old, new in pull_params(client, source, model_id(args)):
                            print(f"{name}: {old} -> {new} from Lapin, written to {source.name}")
                    except Exception as e:  # noqa: BLE001 - a failed look must not end the watch
                        print(f"error reading Lapin's parameters: {type(e).__name__}: {e}", file=sys.stderr)
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


def cmd_pull(args) -> int:
    source = Path(args.source).resolve()
    with connect(args) as client:
        changes = pull_params(client, source, model_id(args))
    for name, old, new in changes:
        print(f"{name}: {old} -> {new}")
    print(f"wrote {len(changes)} parameter(s) from Lapin into {source.name}" if changes
          else f"{source.name} already has Lapin's parameter values")
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
        links = [x for x in client.query("link.list") or [] if _field(x, "kind") in ("lapin", "ifc")]
        if not links:
            print("  no Lapin or IFC links in this project")
        for x in links:
            print(f"  link {_field(x, 'name')} ({_field(x, 'kind')}): {_field(x, 'elements')} elements, {_field(x, 'storeys')} storeys, "
                  f"from {_field(x, 'path') or '(the browser: no path, cannot reload)'}")
    return 0


def cmd_fetch(args) -> int:
    with connect(args) as client:
        answer = client.call("project.exportIfc", {}, timeout=IMPORT_TIMEOUT)
    data = base64.b64decode(_field(answer, "data"))
    Path(args.output).write_bytes(data)
    print(f"wrote {args.output} ({len(data)} bytes): the Lapin project as IFC, to build against. "
          f"Nothing reads it back into Lapin.")
    return 0


def _remove_candidates(source: Path, fmt: str | None) -> list[Path]:
    if fmt == "ifc":
        return [ifc_of(source)]
    if fmt == "lapin":
        return [lapin_of(source)]
    return [lapin_of(source), ifc_of(source)]


def cmd_remove(args) -> int:
    source = Path(args.source).resolve()
    candidates = _remove_candidates(source, args.format)
    removed: list[tuple[Path, object]] = []
    with connect(args) as client:
        for candidate in candidates:
            link = find_link(client, candidate)
            if link is None:
                continue
            client.execute("link.unlink", {"id": _field(link, "id")})
            removed.append((candidate, link))
    if not removed:
        print(f"{candidates[0].name} is not linked in this Lapin.")
        return 0
    for target, link in removed:
        print(f"unlinked {target.name} ({_field(link, 'elements')} elements); Ctrl+Z in Lapin brings it back.")
    return 0


def _normalize_argv(argv: list[str] | None) -> list[str]:
    raw = sys.argv[1:] if argv is None else list(argv)
    out: list[str] = []
    for item in raw:
        if item.startswith("format="):
            out.extend(["--format", item.split("=", 1)[1]])
        else:
            out.append(item)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lite-step lapin", description="A Lite-STEP model in a running Lapin.")
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, run, help, source=False):
        p = sub.add_parser(name, help=help)
        if source:
            p.add_argument("source", help="a model file (.py, compiled first), a .lapin, or a compiled .ifc")
        p.add_argument("--port", type=int, help="the Lapin on this port")
        p.add_argument("--project", help="the Lapin with this project open (its file name or a part of its path)")
        p.set_defaults(run=run)
        return p

    push = command("push", cmd_push, "compile to .lapin (or format=ifc), then link into Lapin or reload the link", source=True)
    push.add_argument("--format", choices=("lapin", "ifc"), default=None, help="push format (default: lapin; pass format=ifc for legacy IFC link)")
    push.add_argument("--watch", action="store_true", help="push again whenever the file is saved")
    push.add_argument("-v", "--verbose", action="store_true", help="show the compiler's reports")
    command("pull", cmd_pull, "write the parameters changed in Lapin into the script", source=True)
    command("status", cmd_status, "the Lapins open on this computer and the models linked in them")
    fetch = command("fetch", cmd_fetch, "the open Lapin project as IFC, to build against")
    fetch.add_argument("-o", "--output", required=True, help="the IFC file to write")
    remove = command("remove", cmd_remove, "unlink the model from Lapin", source=True)
    remove.add_argument("--format", choices=("lapin", "ifc"), default=None, help="which linked format to unlink (default: auto)")

    args = parser.parse_args(_normalize_argv(argv))
    if getattr(args, "source", None) and not Path(args.source).is_file():
        parser.error(f"not a file: {args.source}")
    try:
        return args.run(args)
    except Failed as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        # Ctrl+C mid-compile or mid-reload: say so, no traceback. A reload
        # Lapin had already started finishes there; nothing is half-written.
        print("stopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
