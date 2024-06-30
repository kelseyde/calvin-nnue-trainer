import numpy as np
import network
import piece


def fen_to_features(fen):
    """
    Convert FEN string to a vector of input features for the neural network.
    """
    input_features = np.zeros(network.INPUT_LAYER_SIZE)

    fen_parts = fen.split()
    piece_placement, active_color = fen_parts[:2]

    rank = 7
    file = 0
    for char in piece_placement:
        if char == '/':
            rank -= 1
            file = 0
        elif char.isdigit():
            file += int(char)
        else:
            piece_index = piece.CHAR_TO_PIECE_MAP[char].value
            colour_index = 1 if char.isupper() else 0
            square = square_index(file, rank)
            feature_index = compute_feature_index(piece_index, colour_index, square)
            input_features[feature_index] = 1
            file += 1

    return input_features


def compute_feature_index(piece_index, colour_index, square):
    piece_stride = network.PIECE_STRIDE
    colour_stride = network.COLOUR_STRIDE
    flip_mask = 0x38
    is_white = colour_index == 1
    if is_white:
        return (colour_index ^ 1) * colour_stride + piece_index * piece_stride + square
    else:
        return colour_index * colour_stride + piece_index * piece_stride + square ^ flip_mask


def square_index(file, rank):
    return 8 * rank + file