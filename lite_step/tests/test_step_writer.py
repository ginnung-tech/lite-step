"""Tests for the pure Python STEP P21 writer."""

import pytest

from lite_step.ifc.step_writer import (
    STEP_NULL,
    STEP_DERIVED,
    StepEnum,
    StepRef,
    IdAllocator,
    new_guid,
    encode_value,
    encode_real,
    normalize_step_floats,
    encode_string,
    format_entity,
    step_header,
    step_footer,
    assemble_step_file,
)

# The writer's output must be parseable by the regexes the deduplicator reads
# it back with — these tests are what pins that agreement.
from lite_step.ifc.deduplicator import (
    _ENTITY_RE as DEDUP_ENTITY_RE,
    _FLOAT_RE as DEDUP_FLOAT_RE,
    _REF_RE as DEDUP_REF_RE,
)


# ============================================================================
# Float encoding
# ============================================================================

class TestEncodeReal:
    def test_zero(self):
        assert encode_real(0.0) == '0.'

    def test_negative_zero(self):
        # Negative zero is treated as zero for STEP purposes
        assert encode_real(-0.0) == '0.'

    def test_integer_values(self):
        assert encode_real(3.0) == '3.'
        assert encode_real(5.0) == '5.'
        assert encode_real(19.0) == '19.'
        assert encode_real(-1.0) == '-1.'

    def test_fractional_values(self):
        assert encode_real(3.5) == '3.5'
        assert encode_real(0.15) == '0.15'
        assert encode_real(-1.2) == '-1.2'
        assert encode_real(0.2) == '0.2'

    def test_scientific_notation_negative_exp(self):
        result = encode_real(1e-5)
        assert result == '1.E-05'

    def test_scientific_notation_positive_exp(self):
        result = encode_real(1e200)
        assert result == '1.E+200'

    def test_scientific_notation_large_exp(self):
        result = encode_real(1e-100)
        assert 'E-' in result
        assert '.' in result.split('E')[0]

    def test_always_has_decimal_point(self):
        """STEP P21 reals must always contain a decimal point."""
        for v in [0.0, 1.0, -1.0, 3.5, 0.001, 1e-5, 1e10]:
            result = encode_real(v)
            mantissa = result.split('E')[0] if 'E' in result else result
            assert '.' in mantissa, f"encode_real({v}) = {result!r} has no decimal point"

    def test_no_trailing_zeros(self):
        # 3.50 should be 3.5, not 3.50
        assert encode_real(3.5) == '3.5'
        assert encode_real(0.1) == '0.1'

    def test_full_precision_color_values(self):
        """Color values like 0.517647058823529 need full precision."""
        v = 132 / 255.0
        result = encode_real(v)
        assert '.' in result
        # Should round-trip back to the same value
        assert float(result) == pytest.approx(v, rel=1e-14)


# ============================================================================
# Value encoding
# ============================================================================

class TestEncodeValue:
    def test_null(self):
        assert encode_value(None) == '$'
        assert encode_value(STEP_NULL) == '$'

    def test_derived(self):
        assert encode_value(STEP_DERIVED) == '*'

    def test_bool(self):
        assert encode_value(True) == '.T.'
        assert encode_value(False) == '.F.'

    def test_bool_before_int(self):
        """bool is a subclass of int — must be checked first."""
        assert encode_value(True) == '.T.'  # not '1'
        assert encode_value(False) == '.F.'  # not '0'

    def test_int(self):
        assert encode_value(3) == '3'
        assert encode_value(0) == '0'
        assert encode_value(-5) == '-5'

    def test_float(self):
        assert encode_value(0.0) == '0.'
        assert encode_value(3.5) == '3.5'

    def test_string(self):
        assert encode_value('Body') == "'Body'"
        assert encode_value('Model') == "'Model'"

    def test_string_with_apostrophe(self):
        assert encode_value("it's") == "'it''s'"

    def test_empty_string(self):
        assert encode_value('') == "''"

    def test_enum(self):
        assert encode_value(StepEnum('METRE')) == '.METRE.'
        assert encode_value(StepEnum('AREA')) == '.AREA.'
        assert encode_value(StepEnum('BOTH')) == '.BOTH.'
        assert encode_value(StepEnum('MODEL_VIEW')) == '.MODEL_VIEW.'

    def test_ref(self):
        assert encode_value(StepRef(42)) == '#42'
        assert encode_value(StepRef(1)) == '#1'

    def test_list(self):
        assert encode_value([StepRef(3), StepRef(4), StepRef(2)]) == '(#3,#4,#2)'

    def test_empty_list(self):
        assert encode_value([]) == '()'

    def test_nested_list(self):
        """IfcCartesianPoint has Coordinates as a list: ((0.,0.,0.))"""
        coords = [0.0, 0.0, 0.0]
        # The entity has one attribute which is a list of reals
        assert encode_value(coords) == '(0.,0.,0.)'

    def test_tuple_same_as_list(self):
        assert encode_value((1, 2, 3)) == '(1,2,3)'

    def test_list_of_strings(self):
        assert encode_value(['ViewDefinition [CoordinationView]']) == \
            "('ViewDefinition [CoordinationView]')"

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="Cannot encode"):
            encode_value(object())


# ============================================================================
# NumPy scalar coercion (regression: numpy>=2.0 repr leaked 'np.float64(..)'
# into STEP, producing IFCCARTESIANPOINT((np.float64(-0.7),...)) which web-ifc
# mis-parsed → ThatOpen Fragments skipped the geometry as ">100000 m from
# origin" → blank project. Coords MUST serialise as bare reals.)
# ============================================================================

class TestNumpyScalarCoercion:
    def test_float64_encodes_as_bare_real(self):
        np = pytest.importorskip("numpy")
        assert encode_value(np.float64(-0.7)) == '-0.7'
        assert encode_value(np.float64(0.55)) == '0.55'
        assert encode_value(np.float64(0.0)) == '0.'

    def test_float64_never_emits_numpy_repr(self):
        np = pytest.importorskip("numpy")
        for v in [-0.7, 0.55, 1.2, 20.6, 132 / 255.0, 1e-5]:
            out = encode_value(np.float64(v))
            assert 'np.float64' not in out
            assert 'numpy' not in out
            assert float(out.replace('E', 'e')) == pytest.approx(v, rel=1e-12)

    def test_encode_real_accepts_numpy_scalar(self):
        np = pytest.importorskip("numpy")
        assert encode_real(np.float64(0.15)) == '0.15'
        assert 'np.float64' not in encode_real(np.float64(3.5))

    def test_numpy_int_encodes_as_int(self):
        np = pytest.importorskip("numpy")
        assert encode_value(np.int64(42)) == '42'
        assert encode_value(np.int32(-5)) == '-5'

    def test_coordinate_list_of_numpy_floats(self):
        """The real failure mode: a profile contour row from numpy math."""
        np = pytest.importorskip("numpy")
        contour = np.array([-0.7, -0.55])
        coords = [contour[0], contour[1]]
        out = encode_value(coords)
        assert out == '(-0.7,-0.55)'
        assert 'np.float64' not in out


# ============================================================================
# String encoding
# ============================================================================

class TestEncodeString:
    def test_simple(self):
        assert encode_string('hello') == "'hello'"

    def test_apostrophe(self):
        assert encode_string("it's") == "'it''s'"

    def test_multiple_apostrophes(self):
        assert encode_string("'hello'") == "'''hello'''"

    def test_empty(self):
        assert encode_string('') == "''"


# ============================================================================
# Entity formatting
# ============================================================================

class TestFormatEntity:
    def test_simple_entity(self):
        line = format_entity(6, 'IFCCARTESIANPOINT', [[0.0, 0.0, 0.0]])
        assert line == '#6=IFCCARTESIANPOINT((0.,0.,0.));'

    def test_direction(self):
        line = format_entity(7, 'IFCDIRECTION', [[0.0, 0.0, 1.0]])
        assert line == '#7=IFCDIRECTION((0.,0.,1.));'

    def test_si_unit(self):
        line = format_entity(2, 'IFCSIUNIT', [
            STEP_DERIVED, StepEnum('LENGTHUNIT'), None, StepEnum('METRE')
        ])
        assert line == '#2=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);'

    def test_axis2_placement(self):
        line = format_entity(9, 'IFCAXIS2PLACEMENT3D', [
            StepRef(6), StepRef(7), StepRef(8)
        ])
        assert line == '#9=IFCAXIS2PLACEMENT3D(#6,#7,#8);'

    def test_context(self):
        line = format_entity(10, 'IFCGEOMETRICREPRESENTATIONCONTEXT', [
            None, 'Model', 3, 1e-5, StepRef(9), None
        ])
        assert line == "#10=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-05,#9,$);"

    def test_subcontext(self):
        line = format_entity(11, 'IFCGEOMETRICREPRESENTATIONSUBCONTEXT', [
            'Body', 'Model',
            STEP_DERIVED, STEP_DERIVED, STEP_DERIVED, STEP_DERIVED,
            StepRef(10), None, StepEnum('MODEL_VIEW'), None,
        ])
        assert line == "#11=IFCGEOMETRICREPRESENTATIONSUBCONTEXT('Body','Model',*,*,*,*,#10,$,.MODEL_VIEW.,$);"

    def test_unit_assignment(self):
        line = format_entity(5, 'IFCUNITASSIGNMENT', [
            [StepRef(3), StepRef(4), StepRef(2)]
        ])
        assert line == '#5=IFCUNITASSIGNMENT((#3,#4,#2));'

    def test_project(self):
        line = format_entity(1, 'IFCPROJECT', [
            'someGUID22chars_____X', None, 'Carport',
            None, None, None, None,
            [StepRef(10)], StepRef(5),
        ])
        assert line == "#1=IFCPROJECT('someGUID22chars_____X',$,'Carport',$,$,$,$,(#10),#5);"

    def test_extruded_area_solid(self):
        line = format_entity(28, 'IFCEXTRUDEDAREASOLID', [
            StepRef(24), StepRef(26), StepRef(27), 5.0
        ])
        assert line == '#28=IFCEXTRUDEDAREASOLID(#24,#26,#27,5.);'


# ============================================================================
# Regex compatibility (deduplicator & parallel merger)
# ============================================================================

class TestRegexCompatibility:
    """Verify output lines parse correctly with existing regex patterns."""

    SAMPLE_ENTITIES = [
        (6, 'IFCCARTESIANPOINT', [[0.0, 0.0, 0.0]]),
        (7, 'IFCDIRECTION', [[0.0, 0.0, 1.0]]),
        (2, 'IFCSIUNIT', [STEP_DERIVED, StepEnum('LENGTHUNIT'), None, StepEnum('METRE')]),
        (9, 'IFCAXIS2PLACEMENT3D', [StepRef(6), StepRef(7), StepRef(8)]),
        (10, 'IFCGEOMETRICREPRESENTATIONCONTEXT', [None, 'Model', 3, 1e-5, StepRef(9), None]),
        (28, 'IFCEXTRUDEDAREASOLID', [StepRef(24), StepRef(26), StepRef(27), 5.0]),
        (37, 'IFCCOLOURRGB', [None, 0.517647058823529, 0.580392156862745, 0.658823529411765]),
    ]

    def test_dedup_entity_regex_parses(self):
        for eid, etype, attrs in self.SAMPLE_ENTITIES:
            line = format_entity(eid, etype, attrs)
            m = DEDUP_ENTITY_RE.match(line)
            assert m is not None, f"Dedup regex failed to parse: {line}"
            assert int(m.group(1)) == eid
            assert m.group(2) == etype

    def test_parallel_entity_regex_parses(self):
        for eid, etype, attrs in self.SAMPLE_ENTITIES:
            line = format_entity(eid, etype, attrs)
            m = DEDUP_ENTITY_RE.match(line)
            assert m is not None, f"Parallel regex failed to parse: {line}"
            assert int(m.group(1)) == eid
            assert m.group(2) == etype

    def test_ref_regex_finds_refs(self):
        line = format_entity(9, 'IFCAXIS2PLACEMENT3D', [
            StepRef(6), StepRef(7), StepRef(8)
        ])
        refs = [int(m.group(1)) for m in DEDUP_REF_RE.finditer(line)]
        # Should find refs 6, 7, 8 (not 9 — that's the entity ID before =)
        # Actually the regex searches the full line, so it finds #9 too.
        assert 6 in refs
        assert 7 in refs
        assert 8 in refs

    def test_float_regex_matches_reals(self):
        line = format_entity(37, 'IFCCOLOURRGB', [None, 0.5, 0.3, 0.8])
        floats = DEDUP_FLOAT_RE.findall(line)
        assert len(floats) == 3
        assert float(floats[0]) == pytest.approx(0.5)

    def test_float_regex_matches_zero(self):
        line = format_entity(6, 'IFCCARTESIANPOINT', [[0.0, 0.0, 0.0]])
        floats = DEDUP_FLOAT_RE.findall(line)
        assert len(floats) == 3
        for f in floats:
            assert float(f) == 0.0


# ============================================================================
# GUID generation
# ============================================================================

class TestGuid:
    def test_length(self):
        g = new_guid()
        assert len(g) == 22

    def test_alphabet(self):
        valid = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$')
        for _ in range(100):
            g = new_guid()
            assert set(g).issubset(valid), f"Invalid chars in GUID: {g}"

    def test_first_char_range(self):
        """First char represents top 4 bits of 132-bit encoding; only 0-3 for 128-bit UUID."""
        first_chars = set()
        for _ in range(200):
            g = new_guid()
            first_chars.add(g[0])
        # All first chars should be in '0', '1', '2', '3'
        assert first_chars.issubset({'0', '1', '2', '3'})

    def test_uniqueness(self):
        guids = {new_guid() for _ in range(1000)}
        assert len(guids) == 1000

    def test_valid_step_string(self):
        """GUIDs are used as STEP string values — must encode cleanly."""
        g = new_guid()
        encoded = encode_value(g)
        assert encoded.startswith("'")
        assert encoded.endswith("'")
        # $ in GUID shouldn't double (only apostrophes are doubled)
        assert "''" not in encoded or "'" in g


# ============================================================================
# ID Allocator
# ============================================================================

class TestIdAllocator:
    def test_sequential(self):
        alloc = IdAllocator(start=1)
        assert alloc.next_id() == 1
        assert alloc.next_id() == 2
        assert alloc.next_id() == 3

    def test_reserve_range(self):
        alloc = IdAllocator(start=100)
        start, end = alloc.reserve_range(50)
        assert start == 100
        assert end == 150
        assert alloc.current == 150

    def test_skip_to(self):
        alloc = IdAllocator(start=1)
        alloc.next_id()  # 1
        alloc.skip_to(100)
        assert alloc.next_id() == 100

    def test_skip_to_no_backwards(self):
        alloc = IdAllocator(start=100)
        alloc.skip_to(50)
        assert alloc.current == 100  # unchanged


# ============================================================================
# File structure
# ============================================================================

class TestFileStructure:
    def test_header_structure(self):
        lines = step_header()
        assert lines[0] == 'ISO-10303-21;'
        assert lines[1] == 'HEADER;'
        assert 'FILE_DESCRIPTION' in lines[2]
        assert 'FILE_NAME' in lines[3]
        assert 'FILE_SCHEMA' in lines[4]
        assert lines[5] == 'ENDSEC;'
        assert lines[6] == 'DATA;'

    def test_header_schema(self):
        lines = step_header(schema='IFC4')
        assert "'IFC4'" in lines[4]

    def test_footer_structure(self):
        lines = step_footer()
        assert lines[0] == 'ENDSEC;'
        assert lines[1] == 'END-ISO-10303-21;'

    def test_assemble_complete_file(self):
        header = step_header()
        data = [
            format_entity(1, 'IFCPROJECT', ['guid22', None, 'Test', None, None, None, None, [], StepRef(5)]),
        ]
        footer = step_footer()
        result = assemble_step_file(header, data, footer)

        assert result.startswith('ISO-10303-21;\n')
        assert result.endswith('END-ISO-10303-21;\n')
        assert 'DATA;\n' in result
        assert '#1=IFCPROJECT' in result

    def test_assemble_carport_boilerplate(self):
        """Recreate the first few lines of a carport IFC to verify structure."""
        header = step_header()
        guid = '1FgUiryzL7rhZ5xKyHS4HW'  # fixed for comparison

        data = [
            format_entity(1, 'IFCPROJECT', [guid, None, 'Carport', None, None, None, None, [StepRef(10)], StepRef(5)]),
            format_entity(2, 'IFCSIUNIT', [STEP_DERIVED, StepEnum('LENGTHUNIT'), None, StepEnum('METRE')]),
            format_entity(3, 'IFCSIUNIT', [STEP_DERIVED, StepEnum('AREAUNIT'), None, StepEnum('SQUARE_METRE')]),
            format_entity(4, 'IFCSIUNIT', [STEP_DERIVED, StepEnum('VOLUMEUNIT'), None, StepEnum('CUBIC_METRE')]),
            format_entity(5, 'IFCUNITASSIGNMENT', [[StepRef(3), StepRef(4), StepRef(2)]]),
            format_entity(6, 'IFCCARTESIANPOINT', [[0.0, 0.0, 0.0]]),
            format_entity(7, 'IFCDIRECTION', [[0.0, 0.0, 1.0]]),
            format_entity(8, 'IFCDIRECTION', [[1.0, 0.0, 0.0]]),
            format_entity(9, 'IFCAXIS2PLACEMENT3D', [StepRef(6), StepRef(7), StepRef(8)]),
            format_entity(10, 'IFCGEOMETRICREPRESENTATIONCONTEXT', [
                None, 'Model', 3, 1e-5, StepRef(9), None,
            ]),
            format_entity(11, 'IFCGEOMETRICREPRESENTATIONSUBCONTEXT', [
                'Body', 'Model', STEP_DERIVED, STEP_DERIVED, STEP_DERIVED, STEP_DERIVED,
                StepRef(10), None, StepEnum('MODEL_VIEW'), None,
            ]),
            format_entity(12, 'IFCGEOMETRICREPRESENTATIONSUBCONTEXT', [
                'Sketch', 'Model', STEP_DERIVED, STEP_DERIVED, STEP_DERIVED, STEP_DERIVED,
                StepRef(10), None, StepEnum('SKETCH_VIEW'), None,
            ]),
        ]
        footer = step_footer()
        result = assemble_step_file(header, data, footer)

        # Verify key lines match IfcOpenShell output format
        assert "#2=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);" in result
        assert "#6=IFCCARTESIANPOINT((0.,0.,0.));" in result
        assert "#7=IFCDIRECTION((0.,0.,1.));" in result
        assert "#9=IFCAXIS2PLACEMENT3D(#6,#7,#8);" in result
        assert "#10=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-05,#9,$);" in result
        assert "#11=IFCGEOMETRICREPRESENTATIONSUBCONTEXT('Body','Model',*,*,*,*,#10,$,.MODEL_VIEW.,$);" in result


class TestNormalizeStepFloats:
    """Re-encode IfcOpenShell's raw full-precision doubles to
    shortest STEP reals, bit-exact, without touching strings/GUIDs."""

    def test_cleans_serializer_noise(self):
        step = "#33=IFCEXTRUDEDAREASOLID(#29,#35,#7,0.29999999999999999);"
        assert normalize_step_floats(step) == "#33=IFCEXTRUDEDAREASOLID(#29,#35,#7,0.3);"

    def test_cleans_scientific_precision_context(self):
        step = "#10=IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.0000000000000001E-05,#9,$);"
        assert "1.E-05" in normalize_step_floats(step)
        assert "1.0000000000000001E-05" not in normalize_step_floats(step)

    def test_preserves_integers_and_refs(self):
        # `3` (count) and `#29`/`#35` (refs) have no decimal point → untouched.
        step = "#33=IFCEXTRUDEDAREASOLID(#29,#35,#7,2.3999999999999999);"
        out = normalize_step_floats(step)
        assert out == "#33=IFCEXTRUDEDAREASOLID(#29,#35,#7,2.4);"

    def test_never_touches_quoted_strings_or_guids(self):
        # A float-looking substring inside a name/GUID must survive verbatim.
        step = "#5=IFCWALL('2Yp0ZnEZv0cuPbUsLcBGsb',$,'v2.4:wall:south',$,$,#6,#7,$,$);"
        assert normalize_step_floats(step) == step

    def test_bit_exact_roundtrip(self):
        import re
        vals = ["0.29999999999999999", "2.3999999999999999", "1.0000000000000001E-05",
                "0.20000000000000018", "-0.69999999999999996", "3.", "0."]
        step = "F(" + ",".join(vals) + ");"
        out = normalize_step_floats(step)
        # Every emitted real parses back to the identical double.
        got = re.findall(r"(?<![\w#])(-?\d+\.\d*(?:[eE][+-]?\d+)?)", out)
        assert [float(g) for g in got] == [float(v) for v in vals]

    def test_idempotent_on_clean_reals(self):
        step = "F(0.3,2.4,1.E-05,3.,0.);"
        assert normalize_step_floats(step) == normalize_step_floats(normalize_step_floats(step))

    def test_output_reals_keep_decimal_point(self):
        # STEP REAL must always carry a '.' — even the exponent form.
        out = normalize_step_floats("F(1.0000000000000001E-05,5.0000000000000001);")
        for tok in out[2:-2].split(","):
            assert "." in tok


class TestSnapToPrecision:
    """Rec-3: snap linear dimensions to 6 decimals at the
    subtraction, killing meter-space drift (3.2 - 3.0) at micron precision."""

    def test_kills_subtraction_drift(self):
        from lite_step.ifc.entity_cache import snap_to_precision
        assert (3.2 - 3.0) != 0.2                     # the drift is real
        assert snap_to_precision(3.2 - 3.0) == 0.2    # snapped clean

    def test_lossless_at_mm_scale(self):
        from lite_step.ifc.entity_cache import snap_to_precision
        # mm-authored values (multiples of 0.001 m) are unchanged.
        for v in [0.2, 3.0, 0.008, 1.4, 12.596, -4.5]:
            assert snap_to_precision(v) == v

    def test_variable_precision(self):
        from lite_step.ifc.entity_cache import snap_to_precision
        assert snap_to_precision(0.123456789, 2) == 0.12   # cm
        assert snap_to_precision(0.123456789, 3) == 0.123  # mm
        assert snap_to_precision(0.123456789, 4) == 0.1235 # 0.1 mm

    def test_default_is_six_decimals(self):
        from lite_step.ifc.entity_cache import snap_to_precision, GEOMETRY_DECIMALS
        assert GEOMETRY_DECIMALS == 6
        assert snap_to_precision(0.1234567) == 0.123457
