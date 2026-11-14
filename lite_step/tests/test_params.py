"""``param()``: a model constant a tool may show and edit (``lite_step.params``).

What is pinned is the contract an editor relies on: the script runs exactly as
it would with the bare value, the parameters are found where they are written,
and writing one back changes that value and not a byte else.
"""
from __future__ import annotations

import pytest

from lite_step import param
from lite_step.params import coerce, read_params, write_param

SOURCE = '''from lite_step import param

# Slabs
ROOF_SLAB_THICK = param(250, "mm", "Roof slab thickness")   # cast in place
PITCH = param(30.0, "deg", label="Roof pitch")
BUILDING = param("Hoyt", label="Building name")
OPEN_PLAN = param(True)
DOUBLE = param(ROOF_SLAB_THICK * 2, "mm", "Twice the slab")
PLAIN = 5


def generate():
    LOCAL = param(1, "mm")
    return LOCAL
'''


def test_param_returns_the_value_unchanged():
    assert param(250, "mm", "Roof slab thickness") == 250
    assert type(param(250, "mm")) is int, "integer millimetres stay integers"
    assert param(True) is True and param("Hoyt") == "Hoyt" and param(-1.5, "deg") == -1.5


@pytest.mark.parametrize("args, message", [
    (([250],), "value must be a number, text or True/False"),
    ((250, 7), "unit must be text"),
    ((250, "millimetres-long-unit"), "unit must be text of at most 16"),
    ((250, "mm", "x" * 81), "label must be text of at most 80"),
])
def test_param_refuses_what_a_tool_could_not_show(args, message):
    with pytest.raises((TypeError, ValueError), match=message):
        param(*args)


def test_the_parameters_are_the_module_level_param_assignments():
    found = {p.name: p for p in read_params(SOURCE)}
    assert list(found) == ["ROOF_SLAB_THICK", "PITCH", "BUILDING", "OPEN_PLAN", "DOUBLE"]
    roof = found["ROOF_SLAB_THICK"]
    assert (roof.value, roof.unit, roof.label, roof.line, roof.editable) == (250, "mm", "Roof slab thickness", 4, True)
    assert (found["PITCH"].unit, found["PITCH"].label) == ("deg", "Roof pitch"), "keyword arguments count too"
    assert found["DOUBLE"].value is None and not found["DOUBLE"].editable, "a computed value is not a literal"
    assert "PLAIN" not in found and "LOCAL" not in found


def test_writing_a_parameter_changes_its_value_and_nothing_else():
    out = write_param(SOURCE, "ROOF_SLAB_THICK", "400")
    assert 'ROOF_SLAB_THICK = param(400, "mm", "Roof slab thickness")   # cast in place' in out
    assert out.replace("param(400,", "param(250,") == SOURCE
    assert read_params(out)[0].value == 400


@pytest.mark.parametrize("name, sent, written", [
    ("PITCH", "32.5", 'PITCH = param(32.5, "deg", label="Roof pitch")'),
    ("PITCH", 31, 'PITCH = param(31.0, "deg", label="Roof pitch")'),
    ("BUILDING", "Hoyt Street", "BUILDING = param('Hoyt Street', label=\"Building name\")"),
    ("OPEN_PLAN", "false", "OPEN_PLAN = param(False)"),
    ("ROOF_SLAB_THICK", "400.0", 'ROOF_SLAB_THICK = param(400, "mm", "Roof slab thickness")'),
])
def test_a_value_is_written_in_the_type_the_script_used(name, sent, written):
    assert written in write_param(SOURCE, name, sent)


def test_what_cannot_be_written_is_refused_and_the_file_left_alone():
    with pytest.raises(ValueError, match="whole number"):
        write_param(SOURCE, "ROOF_SLAB_THICK", "400.5")
    with pytest.raises(ValueError, match="computed in the script"):
        write_param(SOURCE, "DOUBLE", 600)
    with pytest.raises(KeyError, match="no parameter PLAIN"):
        write_param(SOURCE, "PLAIN", 6)
    with pytest.raises(ValueError, match="True or False"):
        coerce(True, "maybe")


def test_a_model_using_param_compiles_in_the_sandbox():
    """Model scripts run in a restricted namespace; param must be importable there."""
    from lite_step.compiler.executor import execute_lite_step_script_inproc

    script = '''from lite_step import param
from lite_step.models import Box, Point, Project, Storey

HEIGHT = param(2400, "mm", "Wall height")

def generate_project():
    proj = Project(name="p")
    s = Storey(name="ground")
    s.add(Box(name="wall", start=Point(x=0, y=0, z=0), end=Point(x=4000, y=200, z=HEIGHT)))
    proj.add_storey(s)
    return proj

result = generate_project()
'''
    outcome = execute_lite_step_script_inproc(script)
    assert outcome.success, outcome.error
