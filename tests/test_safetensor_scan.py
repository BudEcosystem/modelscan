"""Tests for the safetensors structural scanner.

Each test constructs an in-memory safetensors blob (8-byte LE u64 header
size + UTF-8 JSON header + raw payload) and runs the scanner against it via
a `Model` wrapping a BytesIO. No real safetensors files on disk are needed.

Test groupings:
  - Happy paths (Qwen regression, FP8, BOOL).
  - Polyglot defenses (gap_at_start, gap, overlap, trailing_bytes).
  - Header bounds (truncated file, oversized header_size, past-EOF header).
  - Header content (invalid UTF-8, duplicate keys, non-object root).
  - Per-tensor validation (dtype, shape, offsets, name).
  - `__metadata__` shape.
  - Format spoofing (pickle, GGUF, HDF5).
  - Resource caps (tensor count via monkeypatch).
"""

import io
import json
import struct
from typing import Any, Dict, List, Optional


from modelscan.issues import IssueCode, IssueSeverity, LayoutIssueDetails
from modelscan.model import Model
from modelscan.scanners.safetensor import scan as st_scan
from modelscan.scanners.safetensor.scan import (
    MAX_HEADER_SIZE,
    SafetensorUnsafeScan,
    compute_data_segment_sha256,
)
from modelscan.settings import SupportedModelFormats


# --- helpers ----------------------------------------------------------------


def _build(
    tensors: Dict[str, Dict[str, Any]],
    metadata: Optional[Any] = None,
    trailing: bytes = b"",
    header_bytes_override: Optional[bytes] = None,
    header_size_override: Optional[int] = None,
) -> bytes:
    """Construct a safetensors blob.

    `tensors` becomes the JSON header (insertion order preserved). Pass
    `metadata` for an `__metadata__` block. `trailing` appends bytes past
    the computed payload (used for trailing_bytes tests). Use the
    *_override args to bypass normal header construction.
    """
    header: Dict[str, Any] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    header.update(tensors)
    header_bytes = (
        header_bytes_override
        if header_bytes_override is not None
        else json.dumps(header).encode("utf-8")
    )
    # Payload size: max declared `end` across tensors.
    max_end = 0
    for t in tensors.values():
        if isinstance(t, dict) and "data_offsets" in t:
            try:
                _, e = t["data_offsets"]
                if isinstance(e, int) and e > max_end:
                    max_end = e
            except Exception:
                pass
    payload = b"\x00" * max_end + trailing
    size_field = (
        header_size_override if header_size_override is not None else len(header_bytes)
    )
    return struct.pack("<Q", size_field) + header_bytes + payload


def _scan(blob: bytes, name: str = "test.safetensors"):
    scanner = SafetensorUnsafeScan(settings={})
    model = Model(source=name, stream=io.BytesIO(blob))
    model.open()
    model.set_context("formats", [SupportedModelFormats.SAFETENSORS])
    return scanner.scan(model)


def _kinds(results) -> List[str]:
    return sorted(
        i.details.kind
        for i in results.issues
        if isinstance(i.details, LayoutIssueDetails)
    )


# === Happy paths ============================================================


def test_qwen_style_alphabetical_keys_not_offset_order():
    """The Qwen regression: JSON keys are alphabetical but on-disk offsets
    are not. The old scanner flagged every such tensor MEDIUM. The new
    scanner must report zero issues.
    """
    tensors = {
        "a": {"dtype": "F32", "shape": [2], "data_offsets": [16, 24]},
        "b": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]},
        "c": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
    }
    r = _scan(_build(tensors))
    assert r.issues == []
    assert r.skipped == []


def test_dtype_lowercase_accepted():
    """Existing budmodel report renders use lowercase (`f32`, `bf16`).
    The normalizer should accept them via case-insensitive lookup.
    """
    tensors = {"x": {"dtype": "f32", "shape": [4], "data_offsets": [0, 16]}}
    r = _scan(_build(tensors))
    assert r.issues == []


def test_fp8_e4m3_recognized():
    tensors = {"w": {"dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 16]}}
    r = _scan(_build(tensors))
    assert r.issues == []


def test_bool_subbyte_packs_to_bytes():
    """16 BOOL elements = 16 bits = 2 bytes — valid sub-byte packing."""
    tensors = {"mask": {"dtype": "BOOL", "shape": [16], "data_offsets": [0, 2]}}
    r = _scan(_build(tensors))
    assert r.issues == []


def test_empty_header_no_tensors_passes():
    """A safetensors file with no tensors is structurally legal."""
    header_bytes = b"{}"
    blob = struct.pack("<Q", len(header_bytes)) + header_bytes
    r = _scan(blob)
    assert r.issues == []


# === Sub-byte alignment =====================================================


def test_f4_misaligned_subbyte_flagged():
    """3 F4 elements = 12 bits ≠ whole bytes — must be flagged."""
    tensors = {"q": {"dtype": "F4", "shape": [3], "data_offsets": [0, 2]}}
    r = _scan(_build(tensors))
    assert "misaligned_subbyte" in _kinds(r)


# === Polyglot defenses (HIGH severity) ======================================


def test_gap_at_start_flagged():
    tensors = {"x": {"dtype": "F32", "shape": [2], "data_offsets": [16, 24]}}
    r = _scan(_build(tensors))
    assert "gap_at_start" in _kinds(r)
    for i in r.issues:
        if (
            isinstance(i.details, LayoutIssueDetails)
            and i.details.kind == "gap_at_start"
        ):
            assert i.severity == IssueSeverity.HIGH


def test_gap_between_tensors_flagged():
    tensors = {
        "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "b": {"dtype": "F32", "shape": [2], "data_offsets": [16, 24]},
    }
    r = _scan(_build(tensors))
    assert "gap" in _kinds(r)


def test_real_overlap_flagged():
    tensors = {
        "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
        "b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},
    }
    r = _scan(_build(tensors))
    assert "overlap" in _kinds(r)


def test_trailing_bytes_flagged():
    """The canonical Trail-of-Bits polyglot finding."""
    tensors = {"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    r = _scan(_build(tensors, trailing=b"PAYLOAD"))
    assert "trailing_bytes" in _kinds(r)


# === Header bounds (skip) ===================================================


def test_file_too_short_skipped():
    r = _scan(b"123")  # < 8 bytes
    assert r.issues == []
    assert len(r.skipped) == 1


def test_header_size_too_large_skipped():
    blob = struct.pack("<Q", MAX_HEADER_SIZE + 1) + b"x" * 16
    r = _scan(blob)
    assert r.issues == []
    assert len(r.skipped) == 1


def test_header_runs_past_eof_skipped():
    blob = struct.pack("<Q", 1000) + b"x" * 92
    r = _scan(blob)
    assert r.issues == []
    assert len(r.skipped) == 1


def test_zero_header_size_skipped():
    blob = struct.pack("<Q", 0) + b"\x00" * 8
    r = _scan(blob)
    assert r.issues == []
    assert len(r.skipped) == 1


# === Header content =========================================================


def test_invalid_utf8_header_flagged():
    blob = _build({}, header_bytes_override=b"\xff\xfe\xfd")
    r = _scan(blob)
    assert any(i.code == IssueCode.INVALID_ENCODING for i in r.issues)


def test_duplicate_keys_flagged():
    raw = (
        b'{"a": {"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
        b' "a": {"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    )
    blob = _build({}, header_bytes_override=raw)
    r = _scan(blob)
    assert any(
        i.code == IssueCode.INVALID_HEADER
        and isinstance(i.details, LayoutIssueDetails)
        and i.details.kind == "duplicate_key"
        for i in r.issues
    )


def test_non_object_root_skipped():
    blob = _build({}, header_bytes_override=b'"not an object"')
    r = _scan(blob)
    assert r.issues == []
    assert len(r.skipped) == 1


def test_json_parse_error_flagged():
    blob = _build({}, header_bytes_override=b"{not valid json")
    r = _scan(blob)
    assert any(i.code == IssueCode.JSON_PARSING_FAILED for i in r.issues)


# === Per-tensor validation ==================================================


def test_unknown_dtype_flagged():
    tensors = {"x": {"dtype": "BLOOP128", "shape": [4], "data_offsets": [0, 16]}}
    r = _scan(_build(tensors))
    assert "unknown_dtype" in _kinds(r)


def test_size_mismatch_flagged():
    # F32 x 4 = 16 bytes but we claim 32
    tensors = {"x": {"dtype": "F32", "shape": [4], "data_offsets": [0, 32]}}
    r = _scan(_build(tensors))
    assert "size_mismatch" in _kinds(r)


def test_missing_keys_flagged():
    header = {"x": {"dtype": "F32"}}
    blob = (
        struct.pack("<Q", len(json.dumps(header).encode()))
        + json.dumps(header).encode()
    )
    r = _scan(blob)
    assert "missing_keys" in _kinds(r)


def test_unknown_extra_keys_flagged_but_continues():
    """Extra keys are LOW; the rest of validation still runs."""
    tensors = {
        "x": {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [0, 8],
            "fancy_extension": "evil",
        }
    }
    r = _scan(_build(tensors))
    kinds = _kinds(r)
    assert "unknown_keys" in kinds
    # The valid tensor should still pass the layout pass — no other findings.
    assert not any(
        k in kinds for k in ("gap_at_start", "gap", "overlap", "trailing_bytes")
    )


def test_negative_shape_flagged():
    tensors = {"x": {"dtype": "F32", "shape": [-2, 4], "data_offsets": [0, 0]}}
    r = _scan(_build(tensors))
    assert "bad_shape" in _kinds(r)


def test_non_int_shape_flagged():
    tensors = {"x": {"dtype": "F32", "shape": ["a", 4], "data_offsets": [0, 16]}}
    r = _scan(_build(tensors))
    assert "bad_shape" in _kinds(r)


def test_out_of_bounds_data_offsets_flagged():
    """End offset past the file's data segment."""
    header = {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 1000]}}
    header_bytes = json.dumps(header).encode()
    blob = struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00" * 8
    r = _scan(blob)
    assert "bad_data_offsets" in _kinds(r)


def test_bad_data_offsets_shape_flagged():
    """data_offsets must be [start, end] — not a 3-element list."""
    header = {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4, 8]}}
    header_bytes = json.dumps(header).encode()
    blob = struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00" * 8
    r = _scan(blob)
    assert "bad_data_offsets" in _kinds(r)


def test_tensor_name_with_nul_flagged():
    """NUL bytes in tensor names. Hard to express in JSON literally,
    so use json.dumps with ensure_ascii=False then a manual replacement.
    """
    weird = "bad\x00name"
    header = {weird: {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    header_bytes = json.dumps(header).encode("utf-8")
    blob = struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00" * 8
    r = _scan(blob)
    assert "bad_name" in _kinds(r)


# === __metadata__ validation ================================================


def test_metadata_string_only_passes():
    tensors = {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    r = _scan(_build(tensors, metadata={"author": "test", "version": "1"}))
    assert r.issues == []


def test_metadata_nested_object_flagged():
    tensors = {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    r = _scan(_build(tensors, metadata={"bad": {"nested": "value"}}))
    assert "bad_metadata" in _kinds(r)


def test_metadata_must_be_dict():
    header = {
        "__metadata__": ["not", "a", "dict"],
        "x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
    }
    header_bytes = json.dumps(header).encode()
    blob = struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00" * 8
    r = _scan(blob)
    assert "bad_metadata" in _kinds(r)


# === Format spoofing ========================================================


def test_pickle_spoof_detected():
    blob = b"\x80\x04" + b"\x00" * 100
    r = _scan(blob)
    fm = [i for i in r.issues if i.code == IssueCode.FORMAT_MISMATCH]
    assert len(fm) == 1
    assert fm[0].severity == IssueSeverity.HIGH  # Stage 2.2 — was LOW before


def test_pytorch_zip_spoof_detected():
    blob = b"PK\x03\x04" + b"\x00" * 100
    r = _scan(blob)
    assert any(i.code == IssueCode.FORMAT_MISMATCH for i in r.issues)


def test_gguf_spoof_detected():
    blob = b"GGUF" + b"\x00" * 100
    r = _scan(blob)
    assert any(i.code == IssueCode.FORMAT_MISMATCH for i in r.issues)


def test_hdf5_spoof_detected():
    blob = b"\x89HDF\r\n\x1a\n" + b"\x00" * 100
    r = _scan(blob)
    assert any(i.code == IssueCode.FORMAT_MISMATCH for i in r.issues)


# === Resource caps ==========================================================


def test_too_many_tensors_capped(monkeypatch):
    """Use a low cap so we don't have to construct 1M entries."""
    monkeypatch.setattr(st_scan, "MAX_TENSORS", 2)
    tensors = {
        "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "b": {"dtype": "F32", "shape": [2], "data_offsets": [8, 16]},
        "c": {"dtype": "F32", "shape": [2], "data_offsets": [16, 24]},
    }
    r = _scan(_build(tensors))
    assert "too_many_tensors" in _kinds(r)


def test_shape_too_large_flagged(monkeypatch):
    """Use a low cap so we don't have to actually compute a 2**40 product."""
    monkeypatch.setattr(st_scan, "MAX_NELEMENTS", 100)
    tensors = {"x": {"dtype": "F32", "shape": [1000], "data_offsets": [0, 4000]}}
    r = _scan(_build(tensors))
    assert "shape_too_large" in _kinds(r)


def test_tensor_name_too_long_flagged(monkeypatch):
    monkeypatch.setattr(st_scan, "MAX_TENSOR_NAME_LEN", 4)
    tensors = {
        "longname": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
    }
    r = _scan(_build(tensors))
    assert "bad_name" in _kinds(r)


# === Stage 2.4 — provenance attestation ====================================


def test_sha256_of_known_payload(tmp_path):
    """Hash of `b'PAYLOAD'` against the openssl-attested SHA-256 of the
    same bytes. The header is just b'{}' so the hash covers only the
    payload bytes — no scanner-internal state leaks into the digest.
    """
    import hashlib

    payload = b"PAYLOAD-FOR-PROVENANCE-TEST"
    header_bytes = b"{}"
    blob = struct.pack("<Q", len(header_bytes)) + header_bytes + payload
    f = tmp_path / "x.safetensors"
    f.write_bytes(blob)

    expected = hashlib.sha256(payload).hexdigest()
    assert compute_data_segment_sha256(f) == expected


def test_sha256_returns_none_when_over_cap(tmp_path):
    """Cap is honored: hashing refuses if the data segment exceeds it."""
    header_bytes = b"{}"
    payload = b"A" * 100
    blob = struct.pack("<Q", len(header_bytes)) + header_bytes + payload
    f = tmp_path / "y.safetensors"
    f.write_bytes(blob)

    assert compute_data_segment_sha256(f, max_bytes=50) is None


def test_sha256_returns_none_for_truncated_file(tmp_path):
    """Anything shorter than the 8-byte header is rejected."""
    f = tmp_path / "z.safetensors"
    f.write_bytes(b"123")
    assert compute_data_segment_sha256(f) is None


def test_sha256_returns_none_when_header_runs_past_eof(tmp_path):
    """File claims a huge header but the bytes don't exist on disk."""
    blob = struct.pack("<Q", 1000) + b"x" * 32  # claims 1000-byte header, has 32
    f = tmp_path / "w.safetensors"
    f.write_bytes(blob)
    assert compute_data_segment_sha256(f) is None


def test_sha256_empty_data_segment(tmp_path):
    """A safetensors file with a real header but no tensors still has a
    well-defined (empty) data segment. SHA-256 of zero bytes is the
    well-known `e3b0c4...` digest.
    """
    header_bytes = b"{}"
    blob = struct.pack("<Q", len(header_bytes)) + header_bytes
    f = tmp_path / "empty.safetensors"
    f.write_bytes(blob)
    assert (
        compute_data_segment_sha256(f)
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
