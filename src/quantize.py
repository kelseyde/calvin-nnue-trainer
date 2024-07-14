import numpy as np

QA = 255
QB = 64
QAB = QA * QB


def quantize(input_weights, input_biases, output_weights, output_bias):
    quant_w0 = np.round(input_weights * QA).astype(np.int16)
    quant_b0 = np.round(input_biases * QA).astype(np.int16)
    quant_w1 = np.round(output_weights * QB).astype(np.int16)
    quant_b1 = np.round(output_bias * QAB).astype(np.int16)
    return quant_w0, quant_b0, quant_w1, quant_b1


def dequantize(input_weights, input_biases, output_weights, output_bias):
    dequant_w0 = input_weights / QA
    dequant_b0 = input_biases / QA
    dequant_w1 = output_weights / QB
    dequant_b1 = output_bias / QAB
    return dequant_w0, dequant_b0, dequant_w1, dequant_b1

