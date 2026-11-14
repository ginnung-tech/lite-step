"""Model parameters: constants a modelling tool may show and edit.

A model script names the values it is built from::

    from lite_step import param

    ROOF_SLAB_THICK = param(250, "mm", "Roof slab thickness")

At run time :func:`param` only checks its arguments and returns the value, so
``ROOF_SLAB_THICK`` is ``250`` and the rest of the script is untouched.

Which constants are parameters is read from the SOURCE, not collected while the
script runs: :func:`read_params` finds every module-level ``NAME = param(...)``,
and :func:`write_param` changes the value in one of them and nothing else in
the file (comments, layout and every other line stay byte for byte). That is
what lets a tool such as Lapin show a model's parameters and write an edit
back into the script it came from.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Optional, Union

Value = Union[bool, int, float, str]

#: Longest unit and label accepted: they are shown beside the value, not read as text.
_UNIT_MAX = 16
_LABEL_MAX = 80


def param(value: Value, unit: str = "", label: str = "") -> Value:
    """A model parameter: returns ``value`` unchanged.

    ``unit`` is how the value is read (``"mm"``, ``"deg"``, ``"%"``, ``""``),
    ``label`` what a person calls it. Both are shown by tools that list the
    model's parameters; neither changes the value. Lengths follow the rest of
    the DSL: integer millimetres.
    """
    if not isinstance(value, (bool, int, float, str)):
        raise TypeError(f"param(): the value must be a number, text or True/False, got {type(value).__name__}")
    if not isinstance(unit, str) or len(unit) > _UNIT_MAX:
        raise ValueError(f"param(): unit must be text of at most {_UNIT_MAX} characters, got {unit!r}")
    if not isinstance(label, str) or len(label) > _LABEL_MAX:
        raise ValueError(f"param(): label must be text of at most {_LABEL_MAX} characters, got {label!r}")
    return value


@dataclass(frozen=True)
class Param:
    """One ``NAME = param(...)`` of a model file."""

    name: str
    #: The value as written; ``None`` when it is not a plain literal (then it cannot be edited).
    value: Optional[Value]
    unit: str
    label: str
    line: int

    @property
    def editable(self) -> bool:
        return self.value is not None


def _is_param_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (isinstance(f, ast.Name) and f.id == "param") or (isinstance(f, ast.Attribute) and f.attr == "param")


def _literal(node: Optional[ast.AST]):
    """A plain literal's value (a negative number included), else ``None``."""
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None
    return value if isinstance(value, (bool, int, float, str)) else None


def _argument(call: ast.Call, index: int, keyword: str):
    if len(call.args) > index:
        return call.args[index]
    return next((k.value for k in call.keywords if k.arg == keyword), None)


def read_params(source: str) -> List[Param]:
    """The model's parameters: every module-level ``NAME = param(...)``, in file order."""
    found = []
    for node in ast.parse(source).body:
        if not (isinstance(node, (ast.Assign, ast.AnnAssign)) and _is_param_call(node.value)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        call = node.value
        unit = _literal(_argument(call, 1, "unit"))
        label = _literal(_argument(call, 2, "label"))
        found.append(Param(
            name=targets[0].id,
            value=_literal(_argument(call, 0, "value")),
            unit=unit if isinstance(unit, str) else "",
            label=label if isinstance(label, str) else "",
            line=node.lineno,
        ))
    return found


def coerce(old: Value, new) -> Value:
    """``new`` (as a tool sends it, often text) in the type the script wrote ``old`` in.

    A whole number stays whole: a parameter written ``250`` takes ``400`` or
    ``"400"`` or ``400.0``, and refuses ``400.5`` rather than turning the
    script's integer millimetres into a float.
    """
    if isinstance(old, bool):
        if isinstance(new, bool):
            return new
        text = str(new).strip().lower()
        if text in ("true", "1", "yes"):
            return True
        if text in ("false", "0", "no"):
            return False
        raise ValueError(f"expected True or False, got {new!r}")
    if isinstance(old, int):
        number = float(new) if not isinstance(new, (int, float)) else new
        if float(number) != int(float(number)):
            raise ValueError(f"expected a whole number, got {new!r}")
        return int(float(number))
    if isinstance(old, float):
        return float(new)
    return str(new)


def write_param(source: str, name: str, value) -> str:
    """``source`` with parameter ``name`` set to ``value``; nothing else changes.

    ``value`` is coerced to the type the script wrote (see :func:`coerce`).
    Raises ``KeyError`` when the file has no such parameter and ``ValueError``
    when its value is not a plain literal or ``value`` does not fit it.
    """
    import libcst as cst

    current = {p.name: p for p in read_params(source)}
    if name not in current:
        raise KeyError(f"the model has no parameter {name}")
    if not current[name].editable:
        raise ValueError(f"{name} is computed in the script, so it can only be changed there")
    new_value = coerce(current[name].value, value)

    def spelled(v: Value) -> str:
        return repr(v) if not isinstance(v, float) or v != int(v) else f"{v:.1f}"

    class Rewrite(cst.CSTTransformer):
        done = False

        def leave_Assign(self, original, updated):
            target = updated.targets[0].target if len(updated.targets) == 1 else None
            call = updated.value
            if not (isinstance(target, cst.Name) and target.value == name and isinstance(call, cst.Call)):
                return updated
            args = list(call.args)
            index = next((i for i, a in enumerate(args) if a.keyword is None or a.keyword.value == "value"), None)
            if index is None or Rewrite.done:
                return updated
            Rewrite.done = True
            args[index] = args[index].with_changes(value=cst.parse_expression(spelled(new_value)))
            return updated.with_changes(value=call.with_changes(args=args))

    module = cst.parse_module(source)
    # Only module-level statements: a parameter is a module constant.
    body = []
    for statement in module.body:
        if Rewrite.done:
            body.append(statement)
        else:
            body.append(statement.visit(Rewrite()))
    return module.with_changes(body=body).code
