# Theory (for 8-bit quantization) #

See this excellent Medium article for more info: https://medium.com/@luis.vasquez.work.log/zero-point-quantization-how-do-we-get-those-formulas-4155b51a60d6

    (1) int8 = scale factor * float32 + offset

how to find scaling factor and offset?

we know that min(x) must map to min(int8) (-128) and max(x) must map to max(int8) (127)

from (1) we can therefore derive:

    (2) -128 = scale factor * min(x) + offset
    (3) 127 = scale factor * max(x) + offset

if we subtract (2) from (3) we get:

    (4) 255 = scale factor * (max(x) - min(x))

if we solve (4) for scale factor we get:

    (5) scale factor = 255 / (max(x) - min(x))

to obtain the offset we can again use (2):

    (2) -128 = scale factor * min(x) + offset

and plugging in the scale factor from (5) we get:

    (6) offset = -(255 / (max(x) - min(x))) * min(x) - 128

### Dealing with rounding ###

since we are projecting to the integer space we need to round the result:

    (7) offset = round(scale factor * min(x)) - 128
    (8) offset = round(-(255 / (max(x) - min(x))) * min(x)) - 128

formula (8) guarantees the offset will be an integer. next, we round the right-hand side:

    (9) int8 = round(scale factor * float32 + offset)
    (10) int8 = round(scale factor * float32) + offset

### De-quantization ###

after quantization, we can't obtain back the original un-quantized values; that precision is lost.
however, we can obtain an approximation of the original values.

if we re-arrange formula (1):

    (1) int8 = scale factor * float32 + offset

we get:

    (11) float32 = (int8 - offset) / scale factor

### 'Zero-point' quantization ###

the zero value carries special meaning in many machine learning models. however, after quantization, it is not obvious
which quantized value corresponds to the original float32 zero value.

luckily, calculating zero-point is easy: we just need to plug in float32 = 0 to formula (1)!

    i.e., zp = scale factor * 0 + offset

thus, we can say that the zero-point is the offset of the transformation!

    (12) zp = offset
    (13) zp = -round(scale factor * min(x)) - 128

Replacing this in the equations for quantization and de-quantization (formulas 10 and 11), we obtain the usual formulas:

    (14) int8 = round(scale factor * float32) + zp
    (15) float32 = (int8 - zp) / scale factor
