"""``lite-step`` CLI — compile a Lite-STEP output file to IFC4 from any cwd.

Two equivalent invocations once the package is installed
(after ``pip install -r requirements.txt``, from the repo root):

    lite-step path/to/output.py
    python -m lite_step path/to/output.py

Both load the model file, read its module-scope ``result`` Project, and
hand it to :func:`lite_step.compile_main`. The model's own
``if __name__ == "__main__": compile_main()`` tail is intentionally
skipped (we run with ``run_name="__lite_step__"``) so flags passed on
the command line drive the single compile, not a second one.

This exists so that an IDE-less developer who just cloned the repo can
type one command and get an IFC, instead of debugging
``ModuleNotFoundError: No module named 'lite_step'`` from a stray cwd.

The module-as-script form (``python -m lite_step ...``) keeps working
even before the console script is on PATH, which is the failure mode
right after ``pip install -e .`` on a shell that hasn't rehashed.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from lite_step.compiler.executor import compile_main, LiteStepCompileError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lite-step",
        description="Compile a Lite-STEP output.py file to IFC4.",
    )
    parser.add_argument(
        "path",
        help="Path to the model file (must define module-scope `result`).",
    )
    parser.add_argument(
        "-o",
        "--output-name",
        default=None,
        help="Output filename written next to the source (default: output.ifc). "
        "Absolute paths are honored as-is.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print Project name and element count before compiling.",
    )
    args = parser.parse_args(argv)

    model_path = Path(args.path).resolve()
    if not model_path.is_file():
        parser.error(f"not a file: {model_path}")

    # Run the model with a non-__main__ run_name so its compile_main()
    # tail is skipped — we drive compile_main ourselves below so CLI
    # flags (--output-name, --verbose) actually take effect.
    namespace = runpy.run_path(str(model_path), run_name="__lite_step__")

    result = namespace.get("result")
    if result is None:
        print(
            f"error: {model_path} did not define module-scope `result`. "
            "Add `result = generate_project()` before the __main__ tail.",
            file=sys.stderr,
        )
        return 2

    kwargs: dict = {"result": result, "source_path": str(model_path)}
    if args.output_name is not None:
        kwargs["output_name"] = args.output_name
    if args.verbose:
        kwargs["verbose"] = True

    # compile_main raises TypeError when `result` resolves to a non-None
    # non-Project value — typically `result = generate_project` (the
    # function itself) instead of `result = generate_project()`. Catch
    # it here so the user sees a one-line stderr message and rc=2,
    # matching the missing-`result` path above, instead of a traceback.
    # Validation / IFC-generation failures raise LiteStepCompileError (loud,
    # never a silent None) — map those to rc=1; compile_main already printed
    # the per-error diagnostics to stderr.
    try:
        compile_main(**kwargs)
    except TypeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except LiteStepCompileError:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
