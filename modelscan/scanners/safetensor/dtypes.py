"""Safetensors dtype table and size accounting.

Mirrors the dtype enum in huggingface/safetensors `safetensors/src/tensor.rs`
(commit ~main, dtypes section). Every dtype the upstream Rust loader accepts
is included here, with its bit-width. Sub-byte dtypes (BOOL=1, F4=4, F6_*=6)
require alignment-aware size accounting — see `expected_bytes` below.

Reference: https://github.com/huggingface/safetensors/blob/main/safetensors/src/tensor.rs
"""

from typing import Optional


# Canonical (upstream) bit-widths. Keys are the exact strings emitted by
# serde for the Dtype enum — uppercase. Lookup is case-insensitive via
# `normalize()` so historical lowercase tests keep working.
DTYPE_BITS = {
    "BOOL": 1,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "F64": 64,
    "I64": 64,
    "U64": 64,
}


def normalize(dtype: str) -> Optional[str]:
    """Case-insensitive lookup returning the canonical uppercase name, or
    None if the dtype is not in the upstream allowlist.
    """
    if not isinstance(dtype, str):
        return None
    upper = dtype.upper()
    return upper if upper in DTYPE_BITS else None


def bitsize(dtype: str) -> Optional[int]:
    """Bit-width of the given dtype, or None if unknown."""
    canon = normalize(dtype)
    return DTYPE_BITS[canon] if canon is not None else None


def expected_bytes(dtype: str, nelements: int) -> Optional[int]:
    """Number of bytes a packed buffer of `nelements` values of `dtype` occupies.

    Returns None when:
      - dtype is unknown
      - nelements is negative
      - the packed bit-count is not a whole number of bytes (sub-byte
        misalignment — upstream Rust rejects this as `MisalignedSlice`)
    """
    if nelements < 0:
        return None
    bits = bitsize(dtype)
    if bits is None:
        return None
    total_bits = nelements * bits
    if total_bits % 8 != 0:
        return None
    return total_bits // 8
