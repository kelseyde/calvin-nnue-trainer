import numpy as np
import torch

CHAR_TO_PIECE_MAP = {
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
}

COLOUR_STRIDE = 64 * 6
PIECE_STRIDE = 64


def fen_to_features(fen):
    fen_parts = fen.split()
    piece_placement, active_color = fen_parts[:2]
    stm = 1 if active_color == 'w' else 0
    white_features = np.zeros(768)
    black_features = np.zeros(768)

    rank, file = 7, 0
    for char in piece_placement:
        if char == '/':
            rank, file = rank - 1, 0
        elif char.isdigit():
            file += int(char)
        else:
            piece_idx = CHAR_TO_PIECE_MAP[char]
            square_idx = 8 * rank + file
            is_white = char.isupper()
            white_features = update_features(white_features, square_idx, piece_idx, is_white, True)
            black_features = update_features(black_features, square_idx, piece_idx, is_white, False)
            file += 1

    white_features = torch.tensor(white_features, dtype=torch.float32)
    black_features = torch.tensor(black_features, dtype=torch.float32)
    stm = torch.tensor([stm], dtype=torch.int8)
    stm_features = white_features if stm == 1 else black_features
    nstm_features = black_features if stm == 1 else white_features

    return stm_features, nstm_features, stm


def update_features(features, square_idx, piece_idx, is_white_piece, is_white_perspective):
    colour_idx = 0 if is_white_piece == is_white_perspective else 1
    if not is_white_perspective:
        square_idx ^= 56
    feature_idx = colour_idx * COLOUR_STRIDE + piece_idx * PIECE_STRIDE + square_idx
    features[feature_idx] = 1
    return features
