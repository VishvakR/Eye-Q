"""Audit how much of an ONNX graph is actually quantized.

The repo names every file ``*_q4``, but a filename is not evidence. This
walks the initializers and reports bytes per dtype, so a component that
shipped unquantized shows up as float32 mass rather than being taken on
trust.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import onnx

# Bytes per element, by TensorProto dtype. INT4/UINT4 are half-byte.
_WIDTH = {
    onnx.TensorProto.FLOAT: 4.0,
    onnx.TensorProto.FLOAT16: 2.0,
    onnx.TensorProto.BFLOAT16: 2.0,
    onnx.TensorProto.INT8: 1.0,
    onnx.TensorProto.UINT8: 1.0,
    onnx.TensorProto.INT32: 4.0,
    onnx.TensorProto.INT64: 8.0,
}
for _name in ("INT4", "UINT4"):
    if hasattr(onnx.TensorProto, _name):
        _WIDTH[getattr(onnx.TensorProto, _name)] = 0.5


def audit(path: str) -> None:
    """Report initializer bytes grouped by dtype, plus quantization op counts."""
    model = onnx.load(path, load_external_data=False)

    by_dtype: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])  # [count, bytes]
    for init in model.graph.initializer:
        name = onnx.TensorProto.DataType.Name(init.data_type)
        n = 1
        for d in init.dims:
            n *= d
        by_dtype[name][0] += 1
        by_dtype[name][1] += n * _WIDTH.get(init.data_type, 4.0)

    ops: dict[str, int] = defaultdict(int)
    for node in model.graph.node:
        ops[node.op_type] += 1

    total = sum(v[1] for v in by_dtype.values())
    print(f"\n=== {path.split('/')[-1]} ===")
    print(f"  initializer payload: {total / 1e6:.1f} MB")
    for name, (count, nbytes) in sorted(by_dtype.items(), key=lambda kv: -kv[1][1]):
        share = 100.0 * nbytes / total if total else 0.0
        print(f"    {name:<10} {count:>6} tensors  {nbytes / 1e6:>9.1f} MB  ({share:5.1f}%)")

    quant_ops = {k: v for k, v in ops.items() if "Quantize" in k or "MatMulNBits" in k}
    print(f"  quant-related ops: {quant_ops or 'NONE'}")
    print(f"  MatMul={ops.get('MatMul', 0)}  Gather={ops.get('Gather', 0)}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        audit(p)
