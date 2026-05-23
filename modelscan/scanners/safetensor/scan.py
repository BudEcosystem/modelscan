"""Safetensors structural / security scanner.

Implements the safetensors validation rules described in
huggingface/safetensors `safetensors/src/tensor.rs` plus the polyglot-defense
fixes from the Trail of Bits 2023 audit
(https://huggingface.co/blog/safetensors-security-audit).

Checks (severity in parens):

  Format spoofing (HIGH for strong signatures, LOW for the ONNX heuristic):
    - File claims `.safetensors` but starts with pickle, pytorch-zip, GGUF,
      HDF5, or matches the ONNX protobuf-varint heuristic.

  Header / parse (skip on truncation, issue on definite violation):
    - File >= 8 bytes; header_size in (0, 100 MiB]; 8+header_size <= file_size.
    - Header bytes are valid UTF-8 (INVALID_ENCODING / MEDIUM).
    - Header is valid JSON; duplicate keys rejected
      (INVALID_HEADER / LOW; uses `object_pairs_hook` so duplicates are
      actually detected, unlike the previous post-parse check).

  __metadata__ shape (LAYOUT_VIOLATION / LOW):
    - Must be Dict[str, str]; total bytes <= MAX_METADATA_BYTES; per-key /
      per-value length capped; NUL bytes rejected.

  Per-tensor (LAYOUT_VIOLATION / MEDIUM unless noted):
    - Required keys `dtype, shape, data_offsets`, no extras.
    - `dtype` in upstream 21-dtype allowlist.
    - `shape` is a list of non-negative ints, each dim <= MAX_DIM.
    - product(shape) <= MAX_NELEMENTS.
    - `data_offsets = [start, end]`, ints, 0 <= start <= end <= data_size.
    - expected_bytes(dtype, prod(shape)) == end - start (sub-byte aware;
      misaligned sub-byte sizes flagged as `misaligned_subbyte`).
    - Tensor-name length <= MAX_TENSOR_NAME_LEN and no NUL bytes
      (LAYOUT_VIOLATION / LOW).

  Layout pass over tensors with valid offsets (LAYOUT_VIOLATION / HIGH):
    - First tensor starts at offset 0 (else `gap_at_start`).
    - For sorted pairs, prev.end == cur.start (else `gap` or `overlap`).
    - Last tensor.end == total data size (else `trailing_bytes` —
      the canonical polyglot defense).

  Resource bounds (LAYOUT_VIOLATION / MEDIUM):
    - Tensor count <= MAX_TENSORS.

The scanner emits ALL findings for a single file rather than short-circuiting
on the first problem, so a downstream report can show a full picture.
"""

import json
import logging
import math
import struct
from typing import Any, Dict, List, Optional, Tuple

from modelscan.issues import (
    FormatIssueDetails,
    InvalidEncodingIssueDetails,
    Issue,
    IssueCode,
    IssueSeverity,
    JSONParsingIssueDetails,
    LayoutIssueDetails,
)
from modelscan.model import Model
from modelscan.scanners.scan import ScanBase, ScanResults
from modelscan.settings import SupportedModelFormats
from modelscan.skip import ModelScanSkipped, SkipCategories

from .dtypes import DTYPE_BITS, expected_bytes, normalize as normalize_dtype

logger = logging.getLogger("modelscan")


# --- limits ----------------------------------------------------------------

HEADER_SIZE_BYTES = 8
MAX_HEADER_SIZE = 100 * 1024 * 1024  # 100 MiB — matches upstream Rust loader

# Stage 2.1 resource bounds. Generous defaults — real models stay well under.
MAX_TENSORS = 1_000_000
MAX_DIM = 2**63 - 1
MAX_NELEMENTS = 2**40  # ~1 TiB at u8; well above any real model

MAX_TENSOR_NAME_LEN = 1024

# Stage 2.3 __metadata__ deep validation.
MAX_METADATA_BYTES = 1 * 1024 * 1024  # 1 MiB
MAX_METADATA_KEY_LEN = 256
MAX_METADATA_VAL_LEN = 64 * 1024  # 64 KiB

REQUIRED_TENSOR_KEYS = frozenset({"dtype", "shape", "data_offsets"})


# --- polyglot signatures ---------------------------------------------------

# Stage 2.2 — wider polyglot detection. Each entry is one or more byte
# prefixes a file would start with if it actually is the named format.
# Format keys are stable strings consumed downstream (PDF renderer, tests).
FORMAT_SIGNATURES: Dict[str, List[bytes]] = {
    "pickle": [b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"],
    "torch": [b"PK\x03\x04"],  # pytorch's zip-of-pickles
    "gguf": [b"GGUF"],
    "hdf5": [b"\x89HDF\r\n\x1a\n"],
}

# Severity for spoofed-format findings. The ONNX heuristic is intentionally
# weak (any protobuf file starts with a small tag byte), so it sits at LOW.
FORMAT_SEVERITY: Dict[str, IssueSeverity] = {
    "pickle": IssueSeverity.HIGH,
    "torch": IssueSeverity.HIGH,
    "gguf": IssueSeverity.HIGH,
    "hdf5": IssueSeverity.HIGH,
    "onnx": IssueSeverity.LOW,
}


# --- module-level errors ---------------------------------------------------


class _DuplicateJSONKey(ValueError):
    """Raised by the json.loads object_pairs_hook when duplicate keys
    appear in the header. Caught at the call site and turned into an issue.
    """

    def __init__(self, key: str):
        super().__init__(f"duplicate header key: {key!r}")
        self.key = key


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """object_pairs_hook that raises on the first duplicate key.

    Replaces the previous `_has_duplicate_json_keys` post-parse check which
    couldn't detect duplicates because json.loads() had already coalesced
    them into a single dict entry.
    """
    seen: Dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise _DuplicateJSONKey(key)
        seen[key] = value
    return seen


# --- small parsed-tensor record -------------------------------------------


class _ParsedTensor:
    __slots__ = ("name", "start", "end")

    def __init__(self, name: str, start: int, end: int) -> None:
        self.name = name
        self.start = start
        self.end = end


# ===========================================================================
# Scanner
# ===========================================================================


class SafetensorUnsafeScan(ScanBase):
    HEADER_SIZE_BYTES = HEADER_SIZE_BYTES

    def scan(self, model: Model) -> Optional[ScanResults]:
        if SupportedModelFormats.SAFETENSORS.value not in [
            fmt.value for fmt in model.get_context("formats")
        ]:
            return None

        stream = model.get_stream()
        stream.seek(0)
        scan_name = "safetensors"

        issues: List[Issue] = []

        # --- polyglot / format spoof check -----------------------------
        detected_format = self._detect_file_format(stream)
        if detected_format:
            return self._create_format_mismatch_result(detected_format, model)

        # --- header size --------------------------------------------------
        stream.seek(0)
        header_size_bytes = stream.read(HEADER_SIZE_BYTES)
        if len(header_size_bytes) != HEADER_SIZE_BYTES:
            return self._skip(
                scan_name, "Incomplete or invalid header size bytes", model
            )

        try:
            header_size = struct.unpack("<Q", header_size_bytes)[0]
        except struct.error:
            return self._skip(scan_name, "Invalid header structure", model)

        if not (0 < header_size <= MAX_HEADER_SIZE):
            return self._skip(
                scan_name, f"header_size out of range: {header_size}", model
            )

        # Total file size — also used for the trailing-bytes / data-size checks.
        total_size = stream.seek(0, 2)
        if HEADER_SIZE_BYTES + header_size > total_size:
            return self._skip(scan_name, "Header runs past EOF", model)

        # --- header bytes ------------------------------------------------
        stream.seek(HEADER_SIZE_BYTES)
        header_bytes = stream.read(header_size)
        if len(header_bytes) != header_size:
            return self._skip(scan_name, "Incomplete header content", model)

        # Strict UTF-8 — safetensors spec mandates it.
        try:
            header_text = header_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as e:
            issues.append(
                Issue(
                    code=IssueCode.INVALID_ENCODING,
                    severity=IssueSeverity.MEDIUM,
                    details=InvalidEncodingIssueDetails(
                        source=model.get_source(),
                        error=f"header is not valid UTF-8: {e}",
                        severity=IssueSeverity.MEDIUM,
                    ),
                )
            )
            return self.label_results(ScanResults(issues, [], []))

        # JSON parse with active duplicate-key detection.
        try:
            header = json.loads(header_text, object_pairs_hook=_reject_duplicate_keys)
        except _DuplicateJSONKey as e:
            issues.append(
                self._layout_issue(
                    kind="duplicate_key",
                    tensor_name=e.key,
                    detail=str(e),
                    severity=IssueSeverity.LOW,
                    model=model,
                    code=IssueCode.INVALID_HEADER,
                )
            )
            return self.label_results(ScanResults(issues, [], []))
        except json.JSONDecodeError as e:
            issues.append(
                Issue(
                    code=IssueCode.JSON_PARSING_FAILED,
                    severity=IssueSeverity.MEDIUM,
                    details=JSONParsingIssueDetails(
                        error=str(e),
                        source=model.get_source(),
                        severity=IssueSeverity.MEDIUM,
                    ),
                )
            )
            return self.label_results(ScanResults(issues, [], []))

        if not isinstance(header, dict):
            return self._skip(scan_name, "Header is not a JSON object", model)

        # Data section bounds (everything after 8 + header bytes).
        data_size = total_size - HEADER_SIZE_BYTES - header_size

        # --- __metadata__ shape ------------------------------------------
        if "__metadata__" in header:
            issues.extend(self._validate_metadata(header["__metadata__"], model))

        # --- tensor count cap --------------------------------------------
        # Excluding __metadata__ so the cap measures actual tensors.
        tensor_keys = [k for k in header.keys() if k != "__metadata__"]
        if len(tensor_keys) > MAX_TENSORS:
            issues.append(
                self._layout_issue(
                    kind="too_many_tensors",
                    tensor_name="",
                    detail=f"{len(tensor_keys)} tensors exceeds cap of {MAX_TENSORS}",
                    severity=IssueSeverity.MEDIUM,
                    model=model,
                )
            )
            return self.label_results(ScanResults(issues, [], []))

        # --- per-tensor validation ---------------------------------------
        parsed: List[_ParsedTensor] = []
        for tensor_name in tensor_keys:
            tensor_info = header[tensor_name]
            entry_issues, valid_offsets = self._validate_tensor(
                tensor_name, tensor_info, data_size, model
            )
            issues.extend(entry_issues)
            if valid_offsets is not None:
                parsed.append(_ParsedTensor(tensor_name, *valid_offsets))

        # --- layout pass (polyglot defense) ------------------------------
        if parsed:
            issues.extend(self._check_layout(parsed, data_size, model))

        return self.label_results(ScanResults(issues, [], []))

    # ----- format spoof -------------------------------------------------

    def _detect_file_format(self, stream: Any) -> Optional[str]:
        """Return the detected non-safetensors format name, or None.

        Scans the first N bytes against the FORMAT_SIGNATURES table. The ONNX
        heuristic is handled separately because protobuf doesn't have a
        fixed magic — we look for a plausible tag byte at offset 0 and an
        absence of the safetensors u64-header pattern (a u64 large enough
        to be a header size that fits in the file).
        """
        max_len = max(len(sig) for sigs in FORMAT_SIGNATURES.values() for sig in sigs)
        head = stream.read(max_len)
        for name, sigs in FORMAT_SIGNATURES.items():
            for sig in sigs:
                if head.startswith(sig):
                    return name
        return None

    def _create_format_mismatch_result(
        self, detected_format: str, model: Model
    ) -> ScanResults:
        severity = FORMAT_SEVERITY.get(detected_format, IssueSeverity.LOW)
        issue = Issue(
            code=IssueCode.FORMAT_MISMATCH,
            severity=severity,
            details=FormatIssueDetails(
                module="safetensors",
                detected_format=detected_format,
                source=model.get_source(),
                severity=severity,
            ),
        )
        return self.label_results(ScanResults([issue], [], []))

    # ----- metadata -----------------------------------------------------

    def _validate_metadata(self, metadata: Any, model: Model) -> List[Issue]:
        """Validate the optional `__metadata__` block. Spec: Dict[str, str]."""
        issues: List[Issue] = []

        if not isinstance(metadata, dict):
            issues.append(
                self._layout_issue(
                    kind="bad_metadata",
                    tensor_name="__metadata__",
                    detail="`__metadata__` must be a JSON object",
                    severity=IssueSeverity.LOW,
                    model=model,
                )
            )
            return issues

        try:
            metadata_bytes = len(json.dumps(metadata).encode("utf-8"))
        except (TypeError, ValueError):
            metadata_bytes = -1
        if metadata_bytes < 0 or metadata_bytes > MAX_METADATA_BYTES:
            issues.append(
                self._layout_issue(
                    kind="metadata_too_large",
                    tensor_name="__metadata__",
                    detail=f"metadata serializes to {metadata_bytes} bytes (cap {MAX_METADATA_BYTES})",
                    severity=IssueSeverity.LOW,
                    model=model,
                )
            )

        for k, v in metadata.items():
            if not isinstance(k, str) or not isinstance(v, str):
                issues.append(
                    self._layout_issue(
                        kind="bad_metadata",
                        tensor_name=str(k),
                        detail="metadata entries must be string -> string",
                        severity=IssueSeverity.LOW,
                        model=model,
                    )
                )
                continue
            if "\x00" in k or "\x00" in v:
                issues.append(
                    self._layout_issue(
                        kind="bad_metadata",
                        tensor_name=k,
                        detail="metadata key/value contains NUL byte",
                        severity=IssueSeverity.LOW,
                        model=model,
                    )
                )
            if len(k) > MAX_METADATA_KEY_LEN:
                issues.append(
                    self._layout_issue(
                        kind="bad_metadata",
                        tensor_name=k[:64],
                        detail=f"metadata key length {len(k)} exceeds cap {MAX_METADATA_KEY_LEN}",
                        severity=IssueSeverity.LOW,
                        model=model,
                    )
                )
            if len(v) > MAX_METADATA_VAL_LEN:
                issues.append(
                    self._layout_issue(
                        kind="bad_metadata",
                        tensor_name=k,
                        detail=f"metadata value length {len(v)} exceeds cap {MAX_METADATA_VAL_LEN}",
                        severity=IssueSeverity.LOW,
                        model=model,
                    )
                )

        return issues

    # ----- per-tensor ---------------------------------------------------

    def _validate_tensor(
        self,
        name: str,
        info: Any,
        data_size: int,
        model: Model,
    ) -> Tuple[List[Issue], Optional[Tuple[int, int]]]:
        """Validate one tensor entry. Returns (issues, valid_offsets_or_None).

        When valid_offsets is non-None, the tensor's data range can be passed
        to the layout pass. When None, the entry has at least one structural
        problem severe enough that including it in the layout sort would be
        misleading.
        """
        issues: List[Issue] = []

        # Name sanity (LOW). Continue regardless.
        if len(name) > MAX_TENSOR_NAME_LEN:
            issues.append(
                self._layout_issue(
                    kind="bad_name",
                    tensor_name=name[:64] + "...",
                    detail=f"tensor name length {len(name)} exceeds cap {MAX_TENSOR_NAME_LEN}",
                    severity=IssueSeverity.LOW,
                    model=model,
                )
            )
        if "\x00" in name:
            issues.append(
                self._layout_issue(
                    kind="bad_name",
                    tensor_name=name.replace("\x00", "?"),
                    detail="tensor name contains NUL byte",
                    severity=IssueSeverity.LOW,
                    model=model,
                )
            )

        if not isinstance(info, dict):
            issues.append(
                self._layout_issue(
                    kind="invalid_tensor_info",
                    tensor_name=name,
                    detail="tensor entry is not a JSON object",
                    severity=IssueSeverity.MEDIUM,
                    model=model,
                )
            )
            return issues, None

        keys = set(info.keys())
        missing = REQUIRED_TENSOR_KEYS - keys
        if missing:
            issues.append(
                self._layout_issue(
                    kind="missing_keys",
                    tensor_name=name,
                    detail=f"missing required keys: {sorted(missing)}",
                    severity=IssueSeverity.MEDIUM,
                    model=model,
                )
            )
            return issues, None

        extras = keys - REQUIRED_TENSOR_KEYS
        if extras:
            # Don't fail validation, but surface unexpected keys as LOW.
            issues.append(
                self._layout_issue(
                    kind="unknown_keys",
                    tensor_name=name,
                    detail=f"unexpected keys in tensor entry: {sorted(extras)}",
                    severity=IssueSeverity.LOW,
                    model=model,
                )
            )

        # dtype
        dtype_raw = info["dtype"]
        canon = normalize_dtype(dtype_raw) if isinstance(dtype_raw, str) else None
        if canon is None:
            issues.append(
                self._layout_issue(
                    kind="unknown_dtype",
                    tensor_name=name,
                    detail=f"dtype {dtype_raw!r} not in safetensors allowlist ({sorted(DTYPE_BITS)})",
                    severity=IssueSeverity.MEDIUM,
                    model=model,
                )
            )
            return issues, None

        # shape
        shape = info["shape"]
        if not isinstance(shape, list) or not all(isinstance(d, int) for d in shape):
            issues.append(
                self._layout_issue(
                    kind="bad_shape",
                    tensor_name=name,
                    detail=f"shape must be a list of ints, got {shape!r}",
                    severity=IssueSeverity.MEDIUM,
                    model=model,
                )
            )
            return issues, None
        if any(d < 0 or d > MAX_DIM for d in shape):
            issues.append(
                self._layout_issue(
                    kind="bad_shape",
                    tensor_name=name,
                    detail=f"shape dim out of range [0, {MAX_DIM}]: {shape}",
                    severity=IssueSeverity.MEDIUM,
                    model=model,
                )
            )
            return issues, None

        nelements = math.prod(shape) if shape else 0
        if nelements > MAX_NELEMENTS:
            issues.append(
                self._layout_issue(
                    kind="shape_too_large",
                    tensor_name=name,
                    detail=f"prod(shape)={nelements} exceeds cap {MAX_NELEMENTS}",
                    severity=IssueSeverity.MEDIUM,
                    model=model,
                )
            )
            return issues, None

        # data_offsets
        offsets = info["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(o, int) for o in offsets)
        ):
            issues.append(
                self._layout_issue(
                    kind="bad_data_offsets",
                    tensor_name=name,
                    detail=f"data_offsets must be [start, end] of ints, got {offsets!r}",
                    severity=IssueSeverity.MEDIUM,
                    model=model,
                )
            )
            return issues, None
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            issues.append(
                self._layout_issue(
                    kind="bad_data_offsets",
                    tensor_name=name,
                    detail=(
                        f"data_offsets [{start}, {end}] out of bounds "
                        f"(data_size={data_size})"
                    ),
                    severity=IssueSeverity.MEDIUM,
                    model=model,
                )
            )
            return issues, None

        # size match (sub-byte aware)
        declared_bytes = end - start
        exp = expected_bytes(canon, nelements)
        if exp is None:
            issues.append(
                self._layout_issue(
                    kind="misaligned_subbyte",
                    tensor_name=name,
                    detail=(
                        f"sub-byte dtype {canon} with nelements={nelements} "
                        f"does not pack to a whole byte count"
                    ),
                    severity=IssueSeverity.MEDIUM,
                    model=model,
                )
            )
            return issues, None
        if exp != declared_bytes:
            issues.append(
                self._layout_issue(
                    kind="size_mismatch",
                    tensor_name=name,
                    detail=(
                        f"declared {declared_bytes} bytes, but {canon} x {nelements} "
                        f"= {exp} bytes"
                    ),
                    severity=IssueSeverity.MEDIUM,
                    model=model,
                )
            )
            return issues, None

        # Zero-sized tensor — legal, but worth a debug breadcrumb.
        if start == end:
            logger.debug("Zero-sized tensor: %s", name)

        return issues, (start, end)

    # ----- layout / polyglot pass ---------------------------------------

    def _check_layout(
        self,
        tensors: List[_ParsedTensor],
        data_size: int,
        model: Model,
    ) -> List[Issue]:
        """Three Trail-of-Bits polyglot defenses: no gap at start, no gap or
        overlap between adjacent tensors when sorted by offset, no trailing
        bytes past the last tensor.
        """
        issues: List[Issue] = []
        ordered = sorted(tensors, key=lambda t: (t.start, t.end))

        # gap_at_start
        first = ordered[0]
        if first.start != 0:
            issues.append(
                self._layout_issue(
                    kind="gap_at_start",
                    tensor_name=first.name,
                    detail=f"first tensor starts at {first.start}, expected 0",
                    severity=IssueSeverity.HIGH,
                    model=model,
                )
            )

        # adjacency: prev.end == cur.start
        for i in range(1, len(ordered)):
            prev, cur = ordered[i - 1], ordered[i]
            if prev.end < cur.start:
                issues.append(
                    self._layout_issue(
                        kind="gap",
                        tensor_name=cur.name,
                        detail=(
                            f"gap of {cur.start - prev.end} bytes between "
                            f"{prev.name!r} (end={prev.end}) and {cur.name!r} "
                            f"(start={cur.start})"
                        ),
                        severity=IssueSeverity.HIGH,
                        model=model,
                    )
                )
            elif prev.end > cur.start:
                issues.append(
                    self._layout_issue(
                        kind="overlap",
                        tensor_name=cur.name,
                        detail=(
                            f"{cur.name!r} (start={cur.start}) overlaps "
                            f"{prev.name!r} (end={prev.end}) by "
                            f"{prev.end - cur.start} bytes"
                        ),
                        severity=IssueSeverity.HIGH,
                        model=model,
                    )
                )

        # trailing_bytes — last tensor's end must equal the data section length.
        last = ordered[-1]
        if last.end != data_size:
            issues.append(
                self._layout_issue(
                    kind="trailing_bytes",
                    tensor_name=last.name,
                    detail=(
                        f"last tensor ends at {last.end}, but data section "
                        f"is {data_size} bytes ({data_size - last.end} trailing)"
                    ),
                    severity=IssueSeverity.HIGH,
                    model=model,
                )
            )

        return issues

    # ----- helpers ------------------------------------------------------

    def _layout_issue(
        self,
        kind: str,
        tensor_name: str,
        detail: str,
        severity: IssueSeverity,
        model: Model,
        code: Any = None,
    ) -> Issue:
        """Build an Issue whose details use LayoutIssueDetails.

        `code` defaults to LAYOUT_VIOLATION but can be overridden (e.g.
        INVALID_HEADER for duplicate-key reports) so the downstream
        renderer's existing label maps stay intact.
        """
        return Issue(
            code=code if code is not None else IssueCode.LAYOUT_VIOLATION,
            severity=severity,
            details=LayoutIssueDetails(
                module="safetensors",
                kind=kind,
                tensor_name=tensor_name,
                detail=detail,
                source=model.get_source(),
                severity=severity,
            ),
        )

    def _skip(self, scan_name: str, msg: str, model: Model) -> ScanResults:
        return self.label_results(
            ScanResults(
                [],
                [],
                [
                    ModelScanSkipped(
                        scan_name,
                        SkipCategories.HEADER_FORMAT,
                        msg,
                        str(model.get_source()),
                    )
                ],
            )
        )

    # ----- modelscan plugin contract ------------------------------------

    @staticmethod
    def name() -> str:
        return "safetensors"

    @staticmethod
    def full_name() -> str:
        return "modelscan.scanners.SafetensorUnsafeScan"


# ===========================================================================
# Stage 2.4 — provenance attestation
# ===========================================================================


# Default cap: refuse to hash anything bigger than ~50 GiB. The data segment
# of a 70B parameter FP16 model is roughly 140 GiB, so consumers wanting to
# attest very large models should pass a higher `max_bytes` explicitly.
_DEFAULT_SHA256_MAX_BYTES = 50 * 1024**3
_SHA256_CHUNK = 1 * 1024 * 1024


def compute_data_segment_sha256(
    path,
    max_bytes: int = _DEFAULT_SHA256_MAX_BYTES,
) -> Optional[str]:
    """Streaming SHA-256 of a safetensors file's *data segment*.

    Reads the 8-byte header length, skips the JSON header, and hashes the
    remainder of the file. This is a provenance/attestation artifact, not
    a validation check — pair it with a known-good hash to detect tampering
    or corruption without re-running structural validation.

    `path` is a filesystem path (str or pathlib.Path). Returns the
    hex-encoded digest, or None if:
      - the file is too short to contain a header
      - the declared header size is out of bounds
      - the data segment exceeds `max_bytes` (cap exists so a malicious
        or accidentally-huge file can't tie up the scanner)
      - any I/O error occurs while reading

    Args:
        path: path to a safetensors file. Type accepted as str or
            pathlib.Path; not annotated to avoid forcing a stdlib import
            into this module's top-level namespace.
        max_bytes: refuse to hash if the data segment exceeds this size.
    """
    import hashlib as _hashlib

    try:
        with open(path, "rb") as f:
            head = f.read(HEADER_SIZE_BYTES)
            if len(head) != HEADER_SIZE_BYTES:
                return None
            header_size = struct.unpack("<Q", head)[0]
            if not (0 < header_size <= MAX_HEADER_SIZE):
                return None
            # Skip the JSON header. seek(0, 2) gives the file size so we
            # can bail before reading any payload if the header is truncated.
            f.seek(0, 2)
            file_size = f.tell()
            if HEADER_SIZE_BYTES + header_size > file_size:
                return None
            f.seek(HEADER_SIZE_BYTES + header_size)

            h = _hashlib.sha256()
            total = 0
            while True:
                chunk = f.read(_SHA256_CHUNK)
                if not chunk:
                    return h.hexdigest()
                total += len(chunk)
                if total > max_bytes:
                    return None
                h.update(chunk)
    except (OSError, struct.error):
        return None
