from torch.onnx import register_custom_op_symbolic

import torch
import torch._C._onnx as _C_onnx

try:
    from torch.onnx import _type_utils
except ImportError:
    _type_utils = None

from torch.onnx import (
    symbolic_helper,
    symbolic_opset9 as opset9,
)


def _fix_type_utils_check(scale):
    if _type_utils is not None:
        return (
            _type_utils.JitScalarType.from_value(scale, _type_utils.JitScalarType.UNDEFINED)
            != _type_utils.JitScalarType.FLOAT
        )
    return False


def fake_quantize_per_tensor_affine(
    g,
    inputs,
    scale,
    zero_point,
    quant_min=-128,
    quant_max=127,
):
    if quant_min == 0:
        zero_point = g.op('Cast', zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op('Cast', zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    if _fix_type_utils_check(scale):
        scale = g.op('Cast', scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    quantized = g.op('QuantizeLinear', inputs, scale, zero_point)
    if (quant_min, quant_max) == (0, 127):
        quantized = g.op(
            'Clip',
            quantized,
            opset9.unused(g),
            g.op('Constant', value_t=torch.tensor(127, dtype=torch.uint8)),
        )
    return g.op('DequantizeLinear', quantized, scale, zero_point)


def fake_quantize_per_channel_affine(
    g,
    inputs,
    scale,
    zero_point,
    axis,
    quant_min=-128,
    quant_max=127,
):
    if quant_min == 0:
        zero_point = g.op('Cast', zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op('Cast', zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    quantized = g.op('QuantizeLinear', inputs, scale, zero_point, axis_i=axis)
    if (quant_min, quant_max) == (0, 127):
        quantized = g.op(
            'Clip',
            quantized,
            opset9.unused(g),
            g.op('Constant', value_t=torch.tensor(127, dtype=torch.uint8)),
        )
    return g.op('DequantizeLinear', quantized, scale, zero_point, axis_i=axis)


def _fake_quantize_learnable_per_tensor_affine(
    g,
    inputs,
    scale,
    zero_point,
    quant_min=-128,
    quant_max=127,
    grad_factor=1.0,
):
    if quant_min == 0:
        zero_point = g.op('Cast', zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op('Cast', zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    if _fix_type_utils_check(scale):
        scale = g.op('Cast', scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    quantized = g.op('QuantizeLinear', inputs, scale, zero_point)
    if (quant_min, quant_max) == (0, 127):
        quantized = g.op(
            'Clip',
            quantized,
            opset9.unused(g),
            g.op('Constant', value_t=torch.tensor(127, dtype=torch.uint8)),
        )
    return g.op('DequantizeLinear', quantized, scale, zero_point)


def _fake_quantize_learnable_per_channel_affine(
    g,
    inputs,
    scale,
    zero_point,
    axis,
    quant_min=-128,
    quant_max=127,
    grad_factor=1.0,
):
    if quant_min == 0:
        zero_point = g.op('Cast', zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op('Cast', zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    # Try to get axis value from the node
    axis_val = 0  # default
    try:
        # Try using _get_tensor_int_value if available
        if hasattr(symbolic_helper, '_get_tensor_int_value'):
            axis_val = symbolic_helper._get_tensor_int_value(axis)
        elif hasattr(symbolic_helper, '_maybe_get_const'):
            axis_val = symbolic_helper._maybe_get_const(axis, 'i')
        else:
            # Fallback: get from node's output
            axis_val = 0
    except:
        axis_val = 0
    quantized = g.op('QuantizeLinear', inputs, scale, zero_point, axis_i=axis_val)
    if (quant_min, quant_max) == (0, 127):
        quantized = g.op(
            'Clip',
            quantized,
            opset9.unused(g),
            g.op('Constant', value_t=torch.tensor(127, dtype=torch.uint8)),
        )
    return g.op('DequantizeLinear', quantized, scale, zero_point, axis_i=axis_val)


# Register custom symbolic functions for PyTorch 2.11+
register_custom_op_symbolic('aten::fake_quantize_per_tensor_affine', fake_quantize_per_tensor_affine, 13)
register_custom_op_symbolic('aten::fake_quantize_per_channel_affine', fake_quantize_per_channel_affine, 13)
register_custom_op_symbolic('aten::_fake_quantize_learnable_per_tensor_affine', _fake_quantize_learnable_per_tensor_affine, 13)
register_custom_op_symbolic('aten::_fake_quantize_learnable_per_channel_affine', _fake_quantize_learnable_per_channel_affine, 13)