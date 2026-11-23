#!/usr/bin/env python3

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inkling2_inspect import (  # noqa: E402
    GGUFFormatError,
    ValidationError,
    parse_gguf,
    validate_inkling2,
)


SCRIPT = Path(__file__).resolve().parents[1] / "inkling2_inspect.py"

META_UINT8 = 0
META_INT8 = 1
META_UINT16 = 2
META_INT16 = 3
META_UINT32 = 4
META_INT32 = 5
META_FLOAT32 = 6
META_BOOL = 7
META_STRING = 8
META_ARRAY = 9
META_UINT64 = 10
META_INT64 = 11
META_FLOAT64 = 12

GGML_TYPES = {
    0: (1, 4),    # F32
    1: (1, 2),    # F16
    8: (32, 34), # Q8_0
    20: (32, 18), # IQ4_NL
}


def packed_string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def encode_scalar(kind, value):
    formats = {
        META_UINT8: "<B",
        META_INT8: "<b",
        META_UINT16: "<H",
        META_INT16: "<h",
        META_UINT32: "<I",
        META_INT32: "<i",
        META_FLOAT32: "<f",
        META_BOOL: "<?",
        META_UINT64: "<Q",
        META_INT64: "<q",
        META_FLOAT64: "<d",
    }
    if kind == META_STRING:
        return packed_string(value)
    return struct.pack(formats[kind], value)


def metadata_record(key, kind, value):
    encoded = packed_string(key) + struct.pack("<I", kind)
    if kind != META_ARRAY:
        return encoded + encode_scalar(kind, value)
    item_kind, items = value
    return (encoded + struct.pack("<IQ", item_kind, len(items)) +
            b"".join(encode_scalar(item_kind, item) for item in items))


def default_inkling_metadata():
    prefix = "inkling2_loop"
    pattern = [True] * 24
    for layer in (5, 11, 17, 23):
        pattern[layer] = False
    return [
        ("general.architecture", META_STRING, prefix),
        ("general.alignment", META_UINT32, 32),
        (f"{prefix}.block_count", META_UINT32, 24),
        (f"{prefix}.loop.start", META_UINT32, 6),
        (f"{prefix}.loop.end", META_UINT32, 18),
        (f"{prefix}.loop.count", META_UINT32, 3),
        (f"{prefix}.loop.gate_rank", META_UINT32, 64),
        (f"{prefix}.attn_res.block_size", META_UINT32, 6),
        (f"{prefix}.attention.sliding_window", META_UINT32, 1024),
        (f"{prefix}.attention.sliding_window_pattern", META_ARRAY,
         (META_BOOL, pattern)),
        (f"{prefix}.shortconv.l_cache", META_UINT32, 4),
    ]


def inkling_tensor_specs():
    tensors = []
    for layer in range(24):
        expert_type = 20 if layer < 20 else 8
        for suffix in ("ffn_gate_up_exps", "ffn_down_exps"):
            tensors.append({
                "name": f"blk.{layer}.{suffix}.weight",
                "dims": (32,),
                "type": expert_type,
            })
    controls = (
        "loop_msg_query",
        "loop_q_ctrl_down", "loop_q_ctrl_gain", "loop_q_ctrl_forget",
        "loop_k_ctrl_down", "loop_k_ctrl_gain", "loop_k_ctrl_forget",
        "loop_v_ctrl_down", "loop_v_ctrl_gain", "loop_v_ctrl_forget",
        "loop_ffn_ctrl_down", "loop_ffn_ctrl_gain", "loop_ffn_ctrl_forget",
    )
    for layer in range(6, 18):
        for suffix in controls:
            tensors.append({
                "name": f"blk.{layer}.{suffix}.weight",
                "dims": (1,),
                "type": 0,
            })
    return tensors


def tensor_nbytes(spec):
    block_elems, block_bytes = GGML_TYPES.get(spec["type"], (1, 0))
    dims = spec["dims"]
    if not dims or dims[0] % block_elems:
        return 0
    rows = 1
    for dim in dims[1:]:
        rows *= dim
    return (dims[0] // block_elems) * block_bytes * rows


def align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def build_gguf(path, metadata=None, tensors=None, file_size_override=None):
    metadata = list(metadata if metadata is not None else [
        ("general.architecture", META_STRING, "synthetic"),
        ("general.alignment", META_UINT32, 32),
    ])
    tensors = [dict(tensor) for tensor in (tensors or [
        {"name": "tensor.weight", "dims": (32,), "type": 20},
    ])]
    alignment = next((value for key, kind, value in metadata
                      if key == "general.alignment" and kind == META_UINT32), 32)

    # Assign compact, correctly aligned offsets unless the case overrides one.
    next_offset = 0
    for spec in tensors:
        if "offset" not in spec:
            next_offset = align(next_offset, alignment)
            spec["offset"] = next_offset
        next_offset = max(next_offset, spec["offset"] + tensor_nbytes(spec))

    kv = b"".join(metadata_record(*record) for record in metadata)
    desc = bytearray()
    for spec in tensors:
        desc += packed_string(spec["name"])
        desc += struct.pack("<I", len(spec["dims"]))
        desc += struct.pack("<" + "Q" * len(spec["dims"]), *spec["dims"])
        desc += struct.pack("<IQ", spec["type"], spec["offset"])
    prefix = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), len(metadata)) + kv + desc
    data_start = align(len(prefix), alignment)
    payload_end = max((spec["offset"] + tensor_nbytes(spec) for spec in tensors), default=0)
    file_size = data_start + payload_end
    if file_size_override is not None:
        file_size = file_size_override
    with open(path, "wb") as out:
        out.write(prefix)
        out.truncate(file_size)
    return data_start


class TrackingReader:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.read_ranges = []

    def read(self, size=-1):
        start = self.wrapped.tell()
        data = self.wrapped.read(size)
        self.read_ranges.append((start, start + len(data)))
        return data

    def seek(self, *args):
        return self.wrapped.seek(*args)

    def tell(self):
        return self.wrapped.tell()

    def close(self):
        return self.wrapped.close()


class GGUFInspectorTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def write(self, **kwargs):
        path = self.root / "synthetic.gguf"
        data_start = build_gguf(path, **kwargs)
        return path, data_start

    def parse_path(self, path):
        with open(path, "rb") as source:
            return parse_gguf(source, str(path))

    def test_reads_header_metadata_and_tensor_descriptors_without_payload(self):
        metadata = [
            ("general.architecture", META_STRING, "test"),
            ("general.alignment", META_UINT32, 32),
            ("test.enabled", META_BOOL, True),
            ("test.scalar", META_FLOAT64, 1.25),
            ("test.names", META_ARRAY, (META_STRING, ["a", "b"])),
            ("test.ids", META_ARRAY, (META_UINT32, [3, 5, 8])),
        ]
        tensors = [
            {"name": "q4nl", "dims": (32,), "type": 20},
            {"name": "q8", "dims": (32,), "type": 8},
            {"name": "norm", "dims": (3,), "type": 0},
            {"name": "half", "dims": (4,), "type": 1},
        ]
        path, data_start = self.write(metadata=metadata, tensors=tensors)
        with open(path, "rb") as raw:
            tracking = TrackingReader(raw)
            info = parse_gguf(tracking, str(path))
        self.assertEqual(info.version, 3)
        self.assertEqual(info.metadata["test.enabled"], True)
        self.assertEqual(info.metadata["test.scalar"], 1.25)
        self.assertEqual(info.metadata["test.names"], ["a", "b"])
        self.assertEqual(info.metadata["test.ids"], [3, 5, 8])
        self.assertEqual([t.type_name for t in info.tensors],
                         ["iq4_nl", "q8_0", "f32", "f16"])
        self.assertEqual([t.nbytes for t in info.tensors], [18, 34, 12, 8])
        self.assertTrue(all(end <= data_start for _, end in tracking.read_ranges))

    def test_rejects_bad_magic_and_unsupported_version(self):
        path = self.root / "bad.gguf"
        path.write_bytes(b"NOPE" + struct.pack("<IQQ", 3, 0, 0))
        with open(path, "rb") as source, self.assertRaises(GGUFFormatError):
            parse_gguf(source)
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 99, 0, 0))
        with open(path, "rb") as source, self.assertRaises(GGUFFormatError):
            parse_gguf(source)

    def test_rejects_truncated_header_and_metadata(self):
        path = self.root / "truncated.gguf"
        path.write_bytes(b"GGUF\x03\x00")
        with open(path, "rb") as source, self.assertRaises(GGUFFormatError):
            parse_gguf(source)
        path.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 0, 1) +
                         struct.pack("<Q", 7) + b"bad")
        with open(path, "rb") as source, self.assertRaises(GGUFFormatError):
            parse_gguf(source)

    def test_rejects_truncated_metadata_array(self):
        path = self.root / "truncated-array.gguf"
        raw = (struct.pack("<4sIQQ", b"GGUF", 3, 0, 1) +
               packed_string("numbers") + struct.pack("<I", META_ARRAY) +
               struct.pack("<IQ", META_UINT32, 3) + struct.pack("<I", 1))
        path.write_bytes(raw)
        with open(path, "rb") as source, self.assertRaises(GGUFFormatError):
            parse_gguf(source)

    def test_summarizes_large_arrays_and_rejects_invalid_boolean_values(self):
        metadata = [
            ("general.architecture", META_STRING, "test"),
            ("general.alignment", META_UINT32, 32),
            ("test.large", META_ARRAY,
             (META_UINT32, list(range(300)))),
        ]
        path, _ = self.write(metadata=metadata)
        info = self.parse_path(path)
        self.assertEqual(info.metadata["test.large"], {
            "type": "array", "length": 300, "element_type": "uint32",
            "items_omitted": True,
        })

        path = self.root / "invalid-bool.gguf"
        raw = (struct.pack("<4sIQQ", b"GGUF", 3, 0, 1) +
               packed_string("flags") + struct.pack("<I", META_ARRAY) +
               struct.pack("<IQ", META_BOOL, 1) + b"\x02")
        path.write_bytes(raw)
        with open(path, "rb") as source, self.assertRaises(GGUFFormatError):
            parse_gguf(source)

    def test_rejects_impossible_string_array_lengths(self):
        path = self.root / "large-length.gguf"
        raw = (struct.pack("<4sIQQ", b"GGUF", 3, 0, 1) +
               packed_string("k") + struct.pack("<I", META_STRING) +
               struct.pack("<Q", 1 << 62))
        path.write_bytes(raw)
        with open(path, "rb") as source, self.assertRaises(GGUFFormatError):
            parse_gguf(source)

    def test_rejects_duplicate_metadata_keys(self):
        metadata = [
            ("general.architecture", META_STRING, "test"),
            ("general.architecture", META_STRING, "test"),
        ]
        path, _ = self.write(metadata=metadata)
        with open(path, "rb") as source, self.assertRaises(GGUFFormatError):
            parse_gguf(source)

    def test_rejects_duplicate_tensor_names(self):
        tensors = [
            {"name": "same", "dims": (32,), "type": 20},
            {"name": "same", "dims": (32,), "type": 20},
        ]
        path, _ = self.write(tensors=tensors)
        with open(path, "rb") as source, self.assertRaises(GGUFFormatError):
            parse_gguf(source)

    def test_rejects_zero_or_excessive_tensor_dimensions(self):
        for dims in ((0,), (1, 1, 1, 1, 1)):
            with self.subTest(dims=dims):
                path, _ = self.write(tensors=[{
                    "name": "bad", "dims": dims, "type": 0,
                }])
                with open(path, "rb") as source, self.assertRaises(GGUFFormatError):
                    parse_gguf(source)

    def test_rejects_overflowing_tensor_size_and_partial_quant_row(self):
        path, _ = self.write(tensors=[{
            "name": "overflow", "dims": (1 << 63, 4), "type": 0,
        }], file_size_override=4096)
        with open(path, "rb") as source, self.assertRaises(GGUFFormatError):
            parse_gguf(source)
        path, _ = self.write(tensors=[{
            "name": "partial", "dims": (33,), "type": 20,
        }])
        with open(path, "rb") as source, self.assertRaises(GGUFFormatError):
            parse_gguf(source)

    def test_rejects_unknown_tensor_type(self):
        path, _ = self.write(tensors=[{
            "name": "unknown", "dims": (32,), "type": 65535,
        }])
        with open(path, "rb") as source, self.assertRaises(GGUFFormatError):
            parse_gguf(source)

    def test_rejects_unaligned_overlapping_and_out_of_file_tensors(self):
        unaligned, _ = self.write(tensors=[{
            "name": "unaligned", "dims": (32,), "type": 20, "offset": 1,
        }])
        with open(unaligned, "rb") as source, self.assertRaises(GGUFFormatError):
            parse_gguf(source)

        overlap, _ = self.write(tensors=[
            {"name": "first", "dims": (32,), "type": 20, "offset": 0},
            {"name": "second", "dims": (32,), "type": 20, "offset": 0},
        ])
        with open(overlap, "rb") as source, self.assertRaises(GGUFFormatError):
            parse_gguf(source)

        outside, data_start = self.write(tensors=[{
            "name": "outside", "dims": (32,), "type": 20, "offset": 32,
        }])
        with open(outside, "r+b") as out:
            out.truncate(data_start + 32)
        with open(outside, "rb") as source, self.assertRaises(GGUFFormatError):
            parse_gguf(source)

    def test_validates_inkling_contract_and_controller_inventory(self):
        path, _ = self.write(metadata=default_inkling_metadata(),
                             tensors=inkling_tensor_specs())
        summary = validate_inkling2(self.parse_path(path))
        self.assertEqual(summary["logical_visits"], 48)
        self.assertEqual(summary["full_attention_layers"], [5, 11, 17, 23])
        self.assertEqual(summary["full_attention_visits"], 8)
        self.assertEqual(summary["sliding_attention_visits"], 40)
        self.assertEqual(summary["controller_tensors"], 156)
        self.assertEqual(summary["expert_tensor_types"], {
            "iq4_nl": 40, "q8_0": 8,
        })

    def test_validation_rejects_changed_loop_or_layer_schedule(self):
        for key, kind, value in (
            ("inkling2_loop.loop.count", META_UINT32, 2),
            ("inkling2_loop.loop.start", META_UINT32, 5),
            ("inkling2_loop.attn_res.block_size", META_UINT32, 3),
            ("inkling2_loop.attention.sliding_window_pattern",
             META_ARRAY, (META_BOOL, [True] * 24)),
        ):
            with self.subTest(key=key):
                metadata = [row for row in default_inkling_metadata()
                            if row[0] != key]
                metadata.append((key, kind, value))
                path, _ = self.write(metadata=metadata,
                                     tensors=inkling_tensor_specs())
                with self.assertRaises(ValidationError):
                    validate_inkling2(self.parse_path(path))

    def test_validation_rejects_non_f32_controller_and_wrong_expert_quant(self):
        tensors = inkling_tensor_specs()
        tensors[48] = {**tensors[48], "type": 1}
        path, _ = self.write(metadata=default_inkling_metadata(), tensors=tensors)
        with self.assertRaises(ValidationError):
            validate_inkling2(self.parse_path(path))

        tensors = inkling_tensor_specs()
        tensors[0] = {**tensors[0], "type": 8}
        path, _ = self.write(metadata=default_inkling_metadata(), tensors=tensors)
        with self.assertRaises(ValidationError):
            validate_inkling2(self.parse_path(path))

    def test_summary_json_does_not_read_tensor_payloads(self):
        path, _ = self.write(metadata=default_inkling_metadata(),
                             tensors=inkling_tensor_specs())
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--json"],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertFalse(report["tensor_payloads_read"])
        self.assertEqual(report["tensor_count"], 204)

    def test_summary_text_states_payloads_were_not_read(self):
        path, _ = self.write()
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--summary"],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Tensor payloads read: no", result.stdout)

    def test_tensor_listing_and_inkling_validation_cli_modes(self):
        path, _ = self.write(metadata=default_inkling_metadata(),
                             tensors=inkling_tensor_specs())
        listing = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--tensors"],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(listing.returncode, 0, listing.stderr)
        self.assertIn("blk.0.ffn_gate_up_exps.weight", listing.stdout)
        self.assertIn("Tensor payloads read: no", listing.stdout)

        validation = subprocess.run(
            [sys.executable, str(SCRIPT), str(path), "--validate-inkling2"],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(validation.returncode, 0, validation.stderr)
        self.assertIn("Inkling 2 Mini validation: PASS", validation.stdout)
        self.assertIn("logical_visits: 48", validation.stdout)


if __name__ == "__main__":
    unittest.main()
