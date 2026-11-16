"""Pure-Python Apache Arrow IPC file writer for `.lapin` entity and relation tables.

Writes standard ``ARROW1`` IPC files matching ``Lapin.Package.ArrowTables``
without requiring ``pyarrow`` as a runtime dependency.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any, Sequence

_MAGIC = b"ARROW1"
_CONTINUATION = 0xFFFFFFFF
_METADATA_V5 = 4

# FlatBuffers Arrow Type union discriminants (Schema.fbs)
_TYPE_INT = 2
_TYPE_FLOATING_POINT = 3
_TYPE_UTF8 = 5
_TYPE_BOOL = 6
_TYPE_LIST = 12

# MessageHeader union discriminants (Message.fbs)
_MSG_SCHEMA = 1
_MSG_RECORD_BATCH = 3


@dataclass(frozen=True)
class ArrowColumn:
    """One column to serialize into an Arrow IPC RecordBatch."""

    name: str
    kind: str  # "utf8" | "int32" | "float64" | "bool" | "list_float64" | "list_utf8"
    values: Sequence[Any]
    nullable: bool = True
    metadata: tuple[tuple[str, str], ...] = ()


class _FlatBuilder:
    """Minimal little-endian FlatBuffers builder (writes backwards from end)."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._vtables: list[tuple[bytes, int]] = []

    def offset(self) -> int:
        return len(self._buf)

    def _pad(self, size: int, align: int) -> None:
        total = len(self._buf) + size
        pad = (-total) % align
        if pad:
            self._buf.extend(b"\x00" * pad)

    def prepend_bytes(self, data: bytes) -> None:
        # ByteBuffer grows backwards, so reverse-prepend data slices.
        self._buf.extend(data[::-1])

    def prepend_u8(self, v: int) -> None:
        self._buf.append(v & 0xFF)

    def prepend_i16(self, v: int) -> None:
        self._pad(2, 2)
        self._buf.extend(struct.pack("<h", v)[::-1])

    def prepend_u16(self, v: int) -> None:
        self._pad(2, 2)
        self._buf.extend(struct.pack("<H", v)[::-1])

    def prepend_i32(self, v: int) -> None:
        self._pad(4, 4)
        self._buf.extend(struct.pack("<i", v)[::-1])

    def prepend_u32(self, v: int) -> None:
        self._pad(4, 4)
        self._buf.extend(struct.pack("<I", v)[::-1])

    def prepend_i64(self, v: int) -> None:
        self._pad(8, 8)
        self._buf.extend(struct.pack("<q", v)[::-1])

    def prepend_offset(self, target_off: int) -> None:
        self._pad(4, 4)
        rel = self.offset() + 4 - target_off
        self._buf.extend(struct.pack("<I", rel)[::-1])

    def create_string(self, text: str) -> int:
        raw = text.encode("utf-8")
        self._pad(len(raw) + 1 + 4, 4)
        self._buf.append(0)  # NUL terminator
        self._buf.extend(raw[::-1])
        self._buf.extend(struct.pack("<I", len(raw))[::-1])
        return self.offset()

    def create_offset_vector(self, offsets: Sequence[int]) -> int:
        self._pad(len(offsets) * 4 + 4, 4)
        for off in reversed(offsets):
            self.prepend_offset(off)
        self.prepend_u32(len(offsets))
        return self.offset()

    def start_struct_vector(self, count: int, elem_size: int, align: int) -> None:
        self._pad(count * elem_size, align)

    def end_vector(self, count: int) -> int:
        self.prepend_u32(count)
        return self.offset()

    def build_table(self, num_fields: int, field_writers: Sequence[tuple[int, int, Any]]) -> int:
        """Build a FlatBuffer table.

        ``field_writers`` is a sequence of ``(slot_index, size_align, writer_fn)``
        in arbitrary order; slots are written in reverse slot order or tracked by
        buffer offset before writing the vtable.
        """
        slot_offsets = [0] * num_fields
        # Align table start so fields don't shift relative to object start later.
        table_start_pre = self.offset()
        for slot, _align, fn in sorted(field_writers, key=lambda x: x[0], reverse=True):
            fn()
            slot_offsets[slot] = self.offset()
        self._pad(4, 4)
        obj_start = self.offset()
        # Placeholder for signed vtable offset (filled after vtable is built/matched)
        vtable_ref_pos = len(self._buf)
        self._buf.extend(b"\x00\x00\x00\x00")
        obj_len = obj_start - table_start_pre + 4

        vtable_slots = []
        for s_off in slot_offsets:
            vtable_slots.append(0 if s_off == 0 else (obj_start + 4) - s_off)
        while vtable_slots and vtable_slots[-1] == 0:
            vtable_slots.pop()
        vt_size = 4 + len(vtable_slots) * 2
        vt_bytes = struct.pack(f"<HH{len(vtable_slots)}H", vt_size, obj_len, *vtable_slots)

        vt_off = None
        for cached_bytes, cached_off in self._vtables:
            if cached_bytes == vt_bytes:
                vt_off = cached_off
                break
        if vt_off is None:
            self._pad(len(vt_bytes), 2)
            self._buf.extend(vt_bytes[::-1])
            vt_off = self.offset()
            self._vtables.append((vt_bytes, vt_off))

        # Signed offset from table_start (obj_start + 4) to vt_off: vt_off - table_start
        rel_vt = vt_off - (obj_start + 4)
        self._buf[vtable_ref_pos : vtable_ref_pos + 4] = struct.pack("<i", rel_vt)[::-1]
        return obj_start + 4

    def finish(self, root_off: int) -> bytes:
        self._pad(4, 8)
        self.prepend_offset(root_off)
        return bytes(self._buf[::-1])


def _pad8(data: bytes) -> bytes:
    rem = len(data) % 8
    return data if rem == 0 else data + b"\x00" * (8 - rem)


def _pack_validity(valid: Sequence[bool]) -> tuple[bytes, int]:
    n = len(valid)
    null_count = sum(1 for v in valid if not v)
    if null_count == 0:
        return b"", 0
    buf = bytearray((n + 7) // 8)
    for i, ok in enumerate(valid):
        if ok:
            buf[i >> 3] |= 1 << (i & 7)
    return bytes(buf), null_count


def _pack_utf8(values: Sequence[str | None]) -> tuple[bytes, int, list[bytes]]:
    valid = [v is not None for v in values]
    vbuf, nulls = _pack_validity(valid)
    offsets = bytearray()
    data = bytearray()
    pos = 0
    offsets.extend(struct.pack("<i", 0))
    for v in values:
        if v is not None:
            enc = v.encode("utf-8")
            data.extend(enc)
            pos += len(enc)
        offsets.extend(struct.pack("<i", pos))
    return vbuf, nulls, [bytes(offsets), bytes(data)]


def _pack_int32(values: Sequence[int | None]) -> tuple[bytes, int, list[bytes]]:
    valid = [v is not None for v in values]
    vbuf, nulls = _pack_validity(valid)
    data = bytearray()
    for v in values:
        data.extend(struct.pack("<i", 0 if v is None else int(v)))
    return vbuf, nulls, [bytes(data)]


def _pack_float64(values: Sequence[float | None], nullable: bool = True) -> tuple[bytes, int, list[bytes]]:
    valid: list[bool] = []
    floats: list[float] = []
    for v in values:
        if v is None:
            valid.append(False)
            floats.append(0.0)
            continue
        f = float(v)
        if not math.isfinite(f):
            if not nullable:
                raise ValueError(f"Non-finite float64 value {v!r} in non-nullable column")
            valid.append(False)
            floats.append(0.0)
        else:
            valid.append(True)
            floats.append(f)
    vbuf, nulls = _pack_validity(valid)
    data = bytearray()
    for f in floats:
        data.extend(struct.pack("<d", f))
    return vbuf, nulls, [bytes(data)]


def _pack_bool(values: Sequence[bool | None]) -> tuple[bytes, int, list[bytes]]:
    valid = [v is not None for v in values]
    vbuf, nulls = _pack_validity(valid)
    buf = bytearray((len(values) + 7) // 8)
    for i, v in enumerate(values):
        if v:
            buf[i >> 3] |= 1 << (i & 7)
    return vbuf, nulls, [bytes(buf)]


def _pack_column(col: ArrowColumn) -> tuple[list[tuple[int, int]], list[bytes]]:
    """Return ``([(length, null_count), ...], [buffer_bytes, ...])`` for one column."""
    if col.kind == "utf8":
        vbuf, nulls, bufs = _pack_utf8(col.values)
        return [(len(col.values), nulls)], [vbuf, *bufs]
    if col.kind == "int32":
        vbuf, nulls, bufs = _pack_int32(col.values)
        return [(len(col.values), nulls)], [vbuf, *bufs]
    if col.kind == "float64":
        vbuf, nulls, bufs = _pack_float64(col.values, nullable=col.nullable)
        return [(len(col.values), nulls)], [vbuf, *bufs]
    if col.kind == "bool":
        vbuf, nulls, bufs = _pack_bool(col.values)
        return [(len(col.values), nulls)], [vbuf, *bufs]
    if col.kind in ("list_float64", "list_utf8"):
        valid = [v is not None for v in col.values]
        vbuf, nulls = _pack_validity(valid)
        offsets = bytearray(struct.pack("<i", 0))
        flat: list[Any] = []
        for v in col.values:
            if v is not None:
                flat.extend(v)
            offsets.extend(struct.pack("<i", len(flat)))
        if col.kind == "list_float64":
            cvbuf, cnulls, cbufs = _pack_float64(flat)
        else:
            cvbuf, cnulls, cbufs = _pack_utf8(flat)
        nodes = [(len(col.values), nulls), (len(flat), cnulls)]
        return nodes, [vbuf, bytes(offsets), cvbuf, *cbufs]
    raise ValueError(f"Unsupported Arrow column kind: {col.kind!r}")


def _build_kv_vector(fb: _FlatBuilder, items: Sequence[tuple[str, str]]) -> int:
    offs = []
    for k, v in items:
        k_off = fb.create_string(k)
        v_off = fb.create_string(v)
        offs.append(fb.build_table(2, [
            (0, 4, lambda o=k_off: fb.prepend_offset(o)),
            (1, 4, lambda o=v_off: fb.prepend_offset(o)),
        ]))
    return fb.create_offset_vector(offs)


def _build_type(fb: _FlatBuilder, kind: str) -> tuple[int, int, list[int]]:
    """Build Arrow Type table and child Field offsets."""
    if kind == "utf8":
        return _TYPE_UTF8, fb.build_table(0, []), []
    if kind == "int32":
        t = fb.build_table(2, [
            (0, 4, lambda: fb.prepend_i32(32)),
            (1, 1, lambda: fb.prepend_u8(1)),
        ])
        return _TYPE_INT, t, []
    if kind == "float64":
        t = fb.build_table(1, [(0, 2, lambda: fb.prepend_i16(2))])  # IEEE float64 precision (enum ordinal 2)
        return _TYPE_FLOATING_POINT, t, []
    if kind == "bool":
        return _TYPE_BOOL, fb.build_table(0, []), []
    if kind in ("list_float64", "list_utf8"):
        child_kind = "float64" if kind == "list_float64" else "utf8"
        child = _build_field(fb, "item", child_kind, nullable=True, metadata=())
        return _TYPE_LIST, fb.build_table(0, []), [child]
    raise ValueError(f"Unsupported Arrow column kind: {kind!r}")


def _build_field(
    fb: _FlatBuilder,
    name: str,
    kind: str,
    nullable: bool,
    metadata: Sequence[tuple[str, str]],
) -> int:
    name_off = fb.create_string(name)
    type_id, type_off, children = _build_type(fb, kind)
    children_off = fb.create_offset_vector(children) if children else 0
    meta_off = _build_kv_vector(fb, metadata) if metadata else 0
    writers: list[tuple[int, int, Any]] = [
        (0, 4, lambda: fb.prepend_offset(name_off)),
        (1, 1, lambda: fb.prepend_u8(1 if nullable else 0)),
        (2, 1, lambda: fb.prepend_u8(type_id)),
        (3, 4, lambda: fb.prepend_offset(type_off)),
    ]
    if children_off:
        writers.append((5, 4, lambda: fb.prepend_offset(children_off)))
    if meta_off:
        writers.append((6, 4, lambda: fb.prepend_offset(meta_off)))
    return fb.build_table(7, writers)


def _build_schema(fb: _FlatBuilder, columns: Sequence[ArrowColumn]) -> int:
    field_offs = [_build_field(fb, c.name, c.kind, c.nullable, c.metadata) for c in columns]
    fields_vec = fb.create_offset_vector(field_offs)
    return fb.build_table(4, [
        (0, 2, lambda: fb.prepend_i16(0)),  # Endianness.Little
        (1, 4, lambda: fb.prepend_offset(fields_vec)),
    ])


def _encapsulate_message(flatbuf: bytes) -> bytes:
    padded = _pad8(flatbuf)
    return struct.pack("<Ii", _CONTINUATION, len(padded)) + padded


def _build_schema_message(columns: Sequence[ArrowColumn]) -> bytes:
    fb = _FlatBuilder()
    schema_off = _build_schema(fb, columns)
    msg_off = fb.build_table(5, [
        (0, 2, lambda: fb.prepend_i16(_METADATA_V5)),
        (1, 1, lambda: fb.prepend_u8(_MSG_SCHEMA)),
        (2, 4, lambda: fb.prepend_offset(schema_off)),
        (3, 8, lambda: fb.prepend_i64(0)),
    ])
    return _encapsulate_message(fb.finish(msg_off))


def _build_record_batch_message(
    num_rows: int,
    nodes: Sequence[tuple[int, int]],
    buffers: Sequence[tuple[int, int]],
    body_len: int,
) -> bytes:
    fb = _FlatBuilder()

    def write_nodes() -> int:
        fb.start_struct_vector(len(nodes), 16, 8)
        for length, null_count in reversed(nodes):
            fb.prepend_i64(null_count)
            fb.prepend_i64(length)
        return fb.end_vector(len(nodes))

    def write_buffers() -> int:
        fb.start_struct_vector(len(buffers), 16, 8)
        for offset, length in reversed(buffers):
            fb.prepend_i64(length)
            fb.prepend_i64(offset)
        return fb.end_vector(len(buffers))

    nodes_vec = write_nodes()
    buffers_vec = write_buffers()
    batch_off = fb.build_table(4, [
        (0, 8, lambda: fb.prepend_i64(num_rows)),
        (1, 4, lambda: fb.prepend_offset(nodes_vec)),
        (2, 4, lambda: fb.prepend_offset(buffers_vec)),
    ])
    msg_off = fb.build_table(5, [
        (0, 2, lambda: fb.prepend_i16(_METADATA_V5)),
        (1, 1, lambda: fb.prepend_u8(_MSG_RECORD_BATCH)),
        (2, 4, lambda: fb.prepend_offset(batch_off)),
        (3, 8, lambda: fb.prepend_i64(body_len)),
    ])
    return _encapsulate_message(fb.finish(msg_off))


def _build_footer(columns: Sequence[ArrowColumn], batches: Sequence[tuple[int, int, int]]) -> bytes:
    fb = _FlatBuilder()
    schema_off = _build_schema(fb, columns)
    fb.start_struct_vector(len(batches), 24, 8)
    for offset, meta_len, body_len in reversed(batches):
        fb.prepend_i64(body_len)
        fb.prepend_i32(0)  # padding between int32 metaDataLength and int64 bodyLength
        fb.prepend_i32(meta_len)
        fb.prepend_i64(offset)
    batches_vec = fb.end_vector(len(batches))
    footer_off = fb.build_table(5, [
        (0, 2, lambda: fb.prepend_i16(_METADATA_V5)),
        (1, 4, lambda: fb.prepend_offset(schema_off)),
        (3, 4, lambda: fb.prepend_offset(batches_vec)),
    ])
    return fb.finish(footer_off)


def write_arrow_file(columns: Sequence[ArrowColumn], num_rows: int) -> bytes:
    """Serialize ``columns`` (all of length ``num_rows``) as an Apache Arrow IPC file."""
    for col in columns:
        if len(col.values) != num_rows:
            raise ValueError(f"Column {col.name!r} has {len(col.values)} rows, expected {num_rows}")

    out = bytearray()
    out.extend(_MAGIC + b"\x00\x00")
    out.extend(_build_schema_message(columns))

    nodes: list[tuple[int, int]] = []
    raw_buffers: list[bytes] = []
    for col in columns:
        col_nodes, col_bufs = _pack_column(col)
        nodes.extend(col_nodes)
        raw_buffers.extend(col_bufs)

    body = bytearray()
    buffer_metas: list[tuple[int, int]] = []
    for raw in raw_buffers:
        offset = len(body)
        length = len(raw)
        buffer_metas.append((offset, length))
        body.extend(_pad8(raw))

    batch_offset = len(out)
    batch_meta = _build_record_batch_message(num_rows, nodes, buffer_metas, len(body))
    out.extend(batch_meta)
    out.extend(body)

    out.extend(struct.pack("<Ii", _CONTINUATION, 0))  # EOS
    footer = _build_footer(columns, [(batch_offset, len(batch_meta), len(body))])
    out.extend(footer)
    out.extend(struct.pack("<i", len(footer)))
    out.extend(_MAGIC)
    return bytes(out)
