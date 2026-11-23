#!/usr/bin/env python3
"""Inspect GGUF headers and tensor descriptors without reading weight payloads."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


UINT64_MAX = (1 << 64) - 1
MAX_STRING_BYTES = 1 << 20
MAX_DIRECTORY_BYTES = 256 << 20
MAX_METADATA_ITEMS = 100_000
MAX_ARRAY_ITEMS = 1_000_000
MAX_INLINE_ARRAY_ITEMS_TOTAL = 1_000_000
MAX_TENSORS = 100_000
MAX_DIMS = 4
INLINE_ARRAY_ITEMS = 256

GGUF_VALUE_NAMES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

GGUF_SCALARS = {
    0: ("uint8", "<B"),
    1: ("int8", "<b"),
    2: ("uint16", "<H"),
    3: ("int16", "<h"),
    4: ("uint32", "<I"),
    5: ("int32", "<i"),
    6: ("float32", "<f"),
    7: ("bool", "<B"),
    10: ("uint64", "<Q"),
    11: ("int64", "<q"),
    12: ("float64", "<d"),
}

# (elements per block, bytes per block), indexed by GGML type id.
# This reader only accepts types used by the supported Inkling 2 Mini GGUF.
GGML_TYPES = {
    0: ("f32", 1, 4),
    1: ("f16", 1, 2),
    8: ("q8_0", 32, 34),
    20: ("iq4_nl", 32, 18),
}


class GGUFFormatError(ValueError):
    """The GGUF header or descriptor directory is malformed or unsupported."""


class ValidationError(ValueError):
    """The GGUF is structurally valid but does not match the Inkling contract."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dimensions: tuple[int, ...]
    ggml_type: int
    type_name: str
    relative_offset: int
    absolute_offset: int
    nbytes: int


@dataclass(frozen=True)
class GGUFInfo:
    path: str
    version: int
    metadata: dict[str, object]
    tensors: tuple[TensorInfo, ...]
    data_offset: int
    file_size: int
    tensor_type_counts: dict[str, int]


class _Reader:
    def __init__(self, stream: BinaryIO, file_size: int):
        self.stream = stream
        self.file_size = file_size

    @property
    def position(self) -> int:
        return self.stream.tell()

    @property
    def remaining(self) -> int:
        return self.file_size - self.position

    def read_exact(self, count: int, what: str) -> bytes:
        if count < 0 or count > self.remaining:
            raise GGUFFormatError(f"truncated {what} at byte {self.position}")
        data = self.stream.read(count)
        if len(data) != count:
            raise GGUFFormatError(f"truncated {what} at byte {self.position}")
        return data

    def skip(self, count: int, what: str) -> None:
        if count < 0 or count > self.remaining:
            raise GGUFFormatError(f"{what} extends past end of file")
        self.stream.seek(self.position + count)

    def unpack(self, fmt: str, what: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read_exact(size, what))[0]

    def string(self, what: str) -> str:
        length = self.unpack("<Q", f"{what} length")
        if length > MAX_STRING_BYTES:
            raise GGUFFormatError(f"{what} length {length} exceeds the safety limit")
        raw = self.read_exact(length, what)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFFormatError(f"{what} is not valid UTF-8") from exc


def _checked_mul(left: int, right: int, what: str) -> int:
    if right < 0 or (left and right > UINT64_MAX // left):
        raise GGUFFormatError(f"{what} overflows uint64")
    return left * right


def _read_scalar(reader: _Reader, kind: int, what: str):
    if kind == 8:
        return reader.string(what)
    try:
        name, fmt = GGUF_SCALARS[kind]
    except KeyError as exc:
        raise GGUFFormatError(f"unsupported {what} type {kind}") from exc
    value = reader.unpack(fmt, what)
    if name == "bool":
        if value not in (0, 1):
            raise GGUFFormatError(f"{what} has invalid boolean value {value}")
        return bool(value)
    return value


def _skip_array_item(reader: _Reader, kind: int, what: str) -> None:
    if kind == 8:
        length = reader.unpack("<Q", f"{what} string length")
        if length > MAX_STRING_BYTES:
            raise GGUFFormatError(f"{what} string length {length} exceeds the safety limit")
        reader.skip(length, what)
        return
    try:
        name, fmt = GGUF_SCALARS[kind]
    except KeyError as exc:
        raise GGUFFormatError(f"unsupported {what} array element type {kind}") from exc
    size = struct.calcsize(fmt)
    if name == "bool":
        value = reader.unpack(fmt, what)
        if value not in (0, 1):
            raise GGUFFormatError(f"{what} has invalid boolean value {value}")
    else:
        reader.skip(size, what)


def _read_array(reader: _Reader, key: str, inline_budget: list[int]) -> object:
    item_type = reader.unpack("<I", f"metadata array {key} element type")
    count = reader.unpack("<Q", f"metadata array {key} length")
    if item_type == 9:
        raise GGUFFormatError(f"metadata array {key} has a nested array type")
    if item_type not in GGUF_SCALARS and item_type != 8:
        raise GGUFFormatError(f"metadata array {key} has unsupported element type {item_type}")
    if count > MAX_ARRAY_ITEMS:
        raise GGUFFormatError(f"metadata array {key} length {count} exceeds the safety limit")

    capture = count <= INLINE_ARRAY_ITEMS and count <= inline_budget[0]
    if not capture:
        if item_type in GGUF_SCALARS:
            size = struct.calcsize(GGUF_SCALARS[item_type][1])
            byte_count = _checked_mul(count, size, f"metadata array {key} size")
        elif count > reader.remaining // 8:
            raise GGUFFormatError(f"metadata array {key} cannot fit in the remaining file")
        else:
            byte_count = None

        if item_type in GGUF_SCALARS:
            if item_type == 7:
                remaining = byte_count
                while remaining:
                    chunk = reader.read_exact(min(remaining, 65536),
                                              f"metadata array {key}")
                    if any(value not in (0, 1) for value in chunk):
                        raise GGUFFormatError(
                            f"metadata array {key} contains an invalid boolean"
                        )
                    remaining -= len(chunk)
            else:
                reader.skip(byte_count, f"metadata array {key}")
            return {
                "type": "array",
                "length": count,
                "element_type": GGUF_VALUE_NAMES.get(item_type, "unknown"),
                "items_omitted": True,
            }

    values = [] if capture else None
    for index in range(count):
        label = f"metadata array {key}[{index}]"
        if capture:
            values.append(_read_scalar(reader, item_type, label))
        else:
            _skip_array_item(reader, item_type, label)
        if reader.position > MAX_DIRECTORY_BYTES:
            raise GGUFFormatError("GGUF metadata directory exceeds the safety limit")
    if capture:
        inline_budget[0] -= count
        return values
    return {
        "type": "array",
        "length": count,
        "element_type": GGUF_VALUE_NAMES.get(item_type, "unknown"),
        "items_omitted": True,
    }


def _read_metadata_value(reader: _Reader, kind: int, key: str,
                         inline_budget: list[int]) -> object:
    if kind == 9:
        return _read_array(reader, key, inline_budget)
    if kind not in GGUF_VALUE_NAMES:
        raise GGUFFormatError(f"metadata key {key} has unknown GGUF value type {kind}")
    return _read_scalar(reader, kind, f"metadata value {key}")


def _tensor_nbytes(dimensions: tuple[int, ...], ggml_type: int) -> tuple[str, int]:
    try:
        name, block_elements, block_bytes = GGML_TYPES[ggml_type]
    except KeyError as exc:
        raise GGUFFormatError(f"unsupported GGML tensor type {ggml_type}") from exc
    if not dimensions or any(dim == 0 for dim in dimensions):
        raise GGUFFormatError("tensor dimensions must be non-zero")
    elements = 1
    for dim in dimensions:
        elements = _checked_mul(elements, dim, "tensor element count")
    if dimensions[0] % block_elements:
        raise GGUFFormatError(
            f"tensor first dimension {dimensions[0]} is not divisible by {block_elements} "
            f"for {name}"
        )
    row_bytes = _checked_mul(dimensions[0] // block_elements, block_bytes,
                             "tensor row size")
    rows = elements // dimensions[0]
    return name, _checked_mul(row_bytes, rows, "tensor byte size")


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def parse_gguf(stream: BinaryIO, path: str = "<stream>") -> GGUFInfo:
    """Read and validate a GGUF v3 directory, leaving tensor payloads untouched."""
    try:
        stream.seek(0, 2)
        file_size = stream.tell()
        stream.seek(0)
    except (OSError, AttributeError) as exc:
        raise GGUFFormatError("input must be a seekable binary stream") from exc
    reader = _Reader(stream, file_size)

    if reader.read_exact(4, "GGUF magic") != b"GGUF":
        raise GGUFFormatError("invalid GGUF magic")
    version = reader.unpack("<I", "GGUF version")
    if version != 3:
        raise GGUFFormatError(f"unsupported GGUF version {version}; expected 3")
    tensor_count = reader.unpack("<Q", "tensor count")
    metadata_count = reader.unpack("<Q", "metadata count")
    if tensor_count > MAX_TENSORS:
        raise GGUFFormatError(f"tensor count {tensor_count} exceeds the safety limit")
    if metadata_count > MAX_METADATA_ITEMS:
        raise GGUFFormatError(f"metadata count {metadata_count} exceeds the safety limit")

    metadata: dict[str, object] = {}
    inline_budget = [MAX_INLINE_ARRAY_ITEMS_TOTAL]
    for index in range(metadata_count):
        key = reader.string(f"metadata key {index}")
        if key in metadata:
            raise GGUFFormatError(f"duplicate metadata key {key}")
        kind = reader.unpack("<I", f"metadata type for {key}")
        metadata[key] = _read_metadata_value(reader, kind, key, inline_budget)
        if reader.position > MAX_DIRECTORY_BYTES:
            raise GGUFFormatError("GGUF metadata directory exceeds the safety limit")

    alignment = metadata.get("general.alignment", 32)
    if isinstance(alignment, bool) or not isinstance(alignment, int):
        raise GGUFFormatError("general.alignment must be an unsigned integer")
    if alignment < 1 or alignment > (1 << 20) or alignment & (alignment - 1):
        raise GGUFFormatError(f"invalid GGUF alignment {alignment}")

    tensors: list[TensorInfo] = []
    names: set[str] = set()
    for index in range(tensor_count):
        name = reader.string(f"tensor name {index}")
        if name in names:
            raise GGUFFormatError(f"duplicate tensor name {name}")
        names.add(name)
        ndim = reader.unpack("<I", f"tensor {name} dimension count")
        if ndim < 1 or ndim > MAX_DIMS:
            raise GGUFFormatError(f"tensor {name} has unsupported rank {ndim}")
        dimensions = tuple(reader.unpack("<Q", f"tensor {name} dimension")
                           for _ in range(ndim))
        ggml_type = reader.unpack("<I", f"tensor {name} type")
        relative_offset = reader.unpack("<Q", f"tensor {name} offset")
        type_name, nbytes = _tensor_nbytes(dimensions, ggml_type)
        if relative_offset % alignment:
            raise GGUFFormatError(
                f"tensor {name} offset {relative_offset} is not aligned to {alignment}"
            )
        tensors.append(TensorInfo(
            name=name,
            dimensions=dimensions,
            ggml_type=ggml_type,
            type_name=type_name,
            relative_offset=relative_offset,
            absolute_offset=0,
            nbytes=nbytes,
        ))
        if reader.position > MAX_DIRECTORY_BYTES:
            raise GGUFFormatError("GGUF tensor directory exceeds the safety limit")

    data_offset = _align(reader.position, alignment)
    if data_offset > file_size:
        raise GGUFFormatError("tensor data section begins past end of file")

    spans = []
    resolved = []
    for tensor in tensors:
        if tensor.relative_offset > UINT64_MAX - data_offset:
            raise GGUFFormatError(f"tensor {tensor.name} absolute offset overflows uint64")
        absolute_offset = data_offset + tensor.relative_offset
        if tensor.nbytes > UINT64_MAX - absolute_offset:
            raise GGUFFormatError(f"tensor {tensor.name} end offset overflows uint64")
        end = absolute_offset + tensor.nbytes
        if end > file_size:
            raise GGUFFormatError(f"tensor {tensor.name} extends past end of file")
        spans.append((absolute_offset, end, tensor.name))
        resolved.append(TensorInfo(
            name=tensor.name,
            dimensions=tensor.dimensions,
            ggml_type=tensor.ggml_type,
            type_name=tensor.type_name,
            relative_offset=tensor.relative_offset,
            absolute_offset=absolute_offset,
            nbytes=tensor.nbytes,
        ))
    spans.sort()
    for previous, current in zip(spans, spans[1:]):
        if current[0] < previous[1]:
            raise GGUFFormatError(
                f"tensor ranges overlap: {previous[2]} and {current[2]}"
            )

    counts: dict[str, int] = {}
    for tensor in resolved:
        counts[tensor.type_name] = counts.get(tensor.type_name, 0) + 1
    return GGUFInfo(
        path=path,
        version=version,
        metadata=metadata,
        tensors=tuple(resolved),
        data_offset=data_offset,
        file_size=file_size,
        tensor_type_counts=dict(sorted(counts.items())),
    )


_CONTROLLER_SUFFIXES = (
    "loop_msg_query",
    "loop_q_ctrl_down", "loop_q_ctrl_gain", "loop_q_ctrl_forget",
    "loop_k_ctrl_down", "loop_k_ctrl_gain", "loop_k_ctrl_forget",
    "loop_v_ctrl_down", "loop_v_ctrl_gain", "loop_v_ctrl_forget",
    "loop_ffn_ctrl_down", "loop_ffn_ctrl_gain", "loop_ffn_ctrl_forget",
)
_EXPERT_SUFFIXES = ("ffn_gate_up_exps", "ffn_down_exps")


def _require_metadata(metadata: dict[str, object], key: str, expected_type):
    if key not in metadata:
        raise ValidationError(f"required metadata key {key} is missing")
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, expected_type):
        raise ValidationError(f"metadata key {key} has the wrong type")
    return value


def validate_inkling2(info: GGUFInfo) -> dict[str, object]:
    """Validate the pinned Inkling 2 Mini layout and return a concise report."""
    metadata = info.metadata
    architecture = metadata.get("general.architecture")
    if architecture != "inkling2_loop":
        raise ValidationError(
            f"general.architecture must be 'inkling2_loop', got {architecture!r}"
        )
    prefix = str(architecture)
    expected = {
        "block_count": 24,
        "loop.start": 6,
        "loop.end": 18,
        "loop.count": 3,
        "loop.gate_rank": 64,
        "attn_res.block_size": 6,
        "attention.sliding_window": 1024,
        "shortconv.l_cache": 4,
    }
    for suffix, wanted in expected.items():
        actual = _require_metadata(metadata, f"{prefix}.{suffix}", int)
        if actual != wanted:
            raise ValidationError(
                f"{prefix}.{suffix} must be {wanted}, got {actual}"
            )
    varying_key = f"{prefix}.loop.varying"
    if varying_key in metadata and metadata[varying_key] is not False:
        raise ValidationError(f"{varying_key} must be false for the fixed three-pass loop")

    pattern_key = f"{prefix}.attention.sliding_window_pattern"
    pattern = metadata.get(pattern_key)
    if not isinstance(pattern, list) or len(pattern) != 24 or any(
        not isinstance(value, bool) for value in pattern
    ):
        raise ValidationError(f"{pattern_key} must be an array of 24 booleans")
    full_layers = [index for index, is_sliding in enumerate(pattern) if not is_sliding]
    expected_full = [5, 11, 17, 23]
    if full_layers != expected_full:
        raise ValidationError(
            f"{pattern_key} full-attention layers must be {expected_full}, got {full_layers}"
        )

    by_name = {tensor.name: tensor for tensor in info.tensors}
    errors = []
    controller_count = 0
    for layer in range(6, 18):
        for suffix in _CONTROLLER_SUFFIXES:
            name = f"blk.{layer}.{suffix}.weight"
            tensor = by_name.get(name)
            if tensor is None:
                errors.append(f"required controller tensor {name} is missing")
            else:
                controller_count += 1
                if tensor.ggml_type != 0:
                    errors.append(f"controller tensor {name} must be F32")
    unexpected_controllers = [
        tensor.name for tensor in info.tensors
        if ".loop_" in tensor.name and tensor.name not in {
            f"blk.{layer}.{suffix}.weight"
            for layer in range(6, 18) for suffix in _CONTROLLER_SUFFIXES
        }
    ]
    if unexpected_controllers:
        errors.append("unexpected loop controller tensors: " + ", ".join(unexpected_controllers[:4]))
    if controller_count != 156:
        errors.append(f"expected 156 controller tensors, found {controller_count}")

    expert_types: dict[str, int] = {"iq4_nl": 0, "q8_0": 0}
    for layer in range(24):
        wanted_type = 20 if layer < 20 else 8
        wanted_name = "iq4_nl" if wanted_type == 20 else "q8_0"
        for suffix in _EXPERT_SUFFIXES:
            name = f"blk.{layer}.{suffix}.weight"
            tensor = by_name.get(name)
            if tensor is None:
                errors.append(f"required expert tensor {name} is missing")
            elif tensor.ggml_type != wanted_type:
                errors.append(f"expert tensor {name} must use {wanted_name}")
            else:
                expert_types[wanted_name] += 1
    if not info.tensor_type_counts.get("iq4_nl"):
        errors.append("GGUF contains no IQ4_NL tensors")
    if not info.tensor_type_counts.get("q8_0"):
        errors.append("GGUF contains no Q8_0 tensors")
    if not info.tensor_type_counts.get("f32"):
        errors.append("GGUF contains no F32 tensors")
    if errors:
        raise ValidationError("; ".join(errors))

    core_length = 18 - 6
    visits = 24 + (3 - 1) * core_length
    full_visits = sum(not pattern[layer] for layer in range(24)) + 2 * sum(
        not pattern[layer] for layer in range(6, 18)
    )
    return {
        "architecture": architecture,
        "physical_layers": 24,
        "logical_visits": visits,
        "loop_core": [6, 18],
        "passes": 3,
        "gate_rank": 64,
        "full_attention_layers": full_layers,
        "full_attention_visits": full_visits,
        "sliding_attention_visits": visits - full_visits,
        "controller_tensors": controller_count,
        "expert_tensor_types": expert_types,
    }


def _tensor_dict(tensor: TensorInfo) -> dict[str, object]:
    return {
        "name": tensor.name,
        "shape": list(tensor.dimensions),
        "ggml_type": tensor.ggml_type,
        "type": tensor.type_name,
        "relative_offset": tensor.relative_offset,
        "absolute_offset": tensor.absolute_offset,
        "bytes": tensor.nbytes,
    }


def _type_summary(info: GGUFInfo) -> str:
    return ", ".join(f"{name}={count}" for name, count in info.tensor_type_counts.items()) or "none"


def _summary_dict(info: GGUFInfo, validation: dict[str, object] | None = None):
    report = {
        "path": info.path,
        "version": info.version,
        "architecture": info.metadata.get("general.architecture"),
        "metadata_count": len(info.metadata),
        "tensor_count": len(info.tensors),
        "tensor_types": info.tensor_type_counts,
        "tensor_data_offset": info.data_offset,
        "file_size": info.file_size,
        "tensor_payloads_read": False,
        "metadata": info.metadata,
        "tensors": [_tensor_dict(tensor) for tensor in info.tensors],
    }
    if validation is not None:
        report["inkling2_validation"] = validation
    return report


def _print_summary(info: GGUFInfo, validation: dict[str, object] | None = None) -> None:
    print(f"GGUF version: {info.version}")
    print(f"Architecture: {info.metadata.get('general.architecture', 'unknown')}")
    print(f"Metadata entries: {len(info.metadata)}")
    print(f"Tensors: {len(info.tensors)}")
    print(f"Tensor types: {_type_summary(info)}")
    print(f"Tensor data offset: {info.data_offset}")
    if validation is not None:
        print("Inkling 2 Mini validation: PASS")
        for key, value in validation.items():
            print(f"{key}: {value}")
    print("Tensor payloads read: no")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect GGUF metadata and tensor descriptors without reading weight payloads."
    )
    parser.add_argument("gguf", type=Path, help="GGUF file to inspect")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--summary", action="store_true", help="print a concise summary (default)")
    modes.add_argument("--tensors", action="store_true", help="list tensor descriptors")
    modes.add_argument("--json", action="store_true", help="emit a JSON report")
    modes.add_argument("--validate-inkling2", action="store_true",
                       help="validate the pinned Inkling 2 Mini GGUF contract")
    args = parser.parse_args(argv)

    try:
        with args.gguf.open("rb") as source:
            info = parse_gguf(source, str(args.gguf))
        validation = validate_inkling2(info) if args.validate_inkling2 else None
    except (OSError, GGUFFormatError, ValidationError) as exc:
        print(f"inkling2_inspect: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(_summary_dict(info, validation), indent=2, sort_keys=True))
    elif args.tensors:
        for tensor in info.tensors:
            shape = "x".join(str(dim) for dim in tensor.dimensions)
            print(f"{tensor.name}\t{shape}\t{tensor.type_name}\t"
                  f"{tensor.absolute_offset}\t{tensor.nbytes}")
        print("Tensor payloads read: no")
    else:
        _print_summary(info, validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
