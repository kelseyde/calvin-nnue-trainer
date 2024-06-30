from enum import Enum


class Piece(Enum):
    P = 0
    N = 1
    B = 2
    R = 3
    Q = 4
    K = 5

    def __int__(self):
        return self.value


CHAR_TO_PIECE_MAP = {
    'p': Piece.P, 'r': Piece.R, 'n': Piece.N, 'b': Piece.B, 'q': Piece.Q, 'k': Piece.K,
    'P': Piece.P, 'R': Piece.R, 'N': Piece.N, 'B': Piece.B, 'Q': Piece.Q, 'K': Piece.K,
}