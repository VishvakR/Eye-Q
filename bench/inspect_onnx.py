"""Print the input/output signature of an ONNX graph without loading weights.

Used to build the Moondream ORT driver against the real graph rather than
against assumptions about how the transformers.js export is shaped.
"""

from __future__ import annotations

import sys

import onnx


def describe(path: str) -> None:
    """Dump every graph input/output with dtype and symbolic shape."""
    model = onnx.load(path, load_external_data=False)
    g = model.graph

    def fmt(v) -> str:
        t = v.type.tensor_type
        dims = [
            (d.dim_param or str(d.dim_value)) if (d.dim_param or d.dim_value) else "?"
            for d in t.shape.dim
        ]
        dtype = onnx.TensorProto.DataType.Name(t.elem_type)
        return f"    {v.name:<40} {dtype:<8} [{', '.join(dims)}]"

    print(f"\n=== {path} ===")
    print(f"  ir_version={model.ir_version} opset={[o.version for o in model.opset_import]}")
    print("  INPUTS:")
    for v in g.input:
        print(fmt(v))
    print("  OUTPUTS:")
    for v in g.output:
        print(fmt(v))
    print(f"  (inputs={len(g.input)} outputs={len(g.output)} nodes={len(g.node)})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        describe(p)
