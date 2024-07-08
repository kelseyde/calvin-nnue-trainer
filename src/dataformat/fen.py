import numpy as np

CHAR_TO_PIECE_MAP = {
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
}


def fen_to_features(fen):
    white_features = np.zeros((6, 64))
    black_features = np.zeros((6, 64))

    fen_parts = fen.split()
    piece_placement, active_color = fen_parts[:2]

    rank, file = 7, 0
    for char in piece_placement:
        if char == '/':
            rank, file = rank - 1, 0
        elif char.isdigit():
            file += int(char)
        else:
            piece_index = CHAR_TO_PIECE_MAP[char]
            is_white = char.isupper()
            square = square_index(file, rank, is_white)
            if is_white:
                white_features[piece_index][square] = 1
            else:
                black_features[piece_index][square] = 1
            file += 1

    input_features = np.concatenate([white_features, black_features]
                                    if active_color == 'w' else
                                    [black_features, white_features])
    return input_features.flatten()


def square_index(file, rank, is_white):
    if is_white:
        return 8 * rank + file
    else:
        # Flip the rank, not the file
        return 8 * (7 - rank) + file
