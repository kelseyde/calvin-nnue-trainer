import torch

MIN_INT8 = -128
MAX_INT8 = 127
RANGE_INT8 = MAX_INT8 - MIN_INT8 + 1 # 256

MIN_INT16 = -32768
MAX_INT16 = 32767
RANGE_INT16 = MAX_INT16 - MIN_INT16 + 1  # 65536

MIN_INT32 = -2147483648
MAX_INT32 = 2147483647
RANGE_INT32 = MAX_INT32 - MIN_INT32 + 1  # 4294967296


def quantize_int8(tensor):
    """Quantize a float32 tensor to int8."""
    return torch.round(tensor * MAX_INT8)


def quantize_int16(tensor):
    """Quantize a float32 tensor to int16."""
    return torch.round(tensor * MAX_INT16)


def quantize_int32(tensor):
    """Quantize a float32 tensor to int32."""
    return torch.round(tensor * MAX_INT32)


def dequantize_int8(tensor):
    """Dequantize an int8 tensor to float32."""
    return tensor / MAX_INT8


def dequantize_int16(tensor):
    """Dequantize an int16 tensor to float32."""
    return tensor / MAX_INT16


def dequantize_int32(tensor):
    """Dequantize an int32 tensor to float32."""
    return tensor / MAX_INT32

