import struct

# Constants
PIECE_MAP = {
    0: 'P',  1: 'p',  2: 'N',  3: 'n',  4: 'B',  5: 'b',  6: 'R',  7: 'r',  8: 'Q',  9: 'q',
    10: 'K', 11: 'k', 12: 'P_EP', 13: 'R_CASTLE_W', 14: 'R_CASTLE_B', 15: 'k_black_to_move'
}

# Constants for masks and shifts
SQUARE_MASK = 0b111111
PROMOTED_PIECE_TYPE_MASK = 0b11
MOVE_TYPE_MASK = 0b11

# Constants for the shifts
MOVE_TYPE_SHIFT = 14
FROM_SQUARE_SHIFT = 8
TO_SQUARE_SHIFT = 2

# Piece map for promotion
PROMOTION_PIECE_MAP = {
    0: 'N',  # Knight
    1: 'B',  # Bishop
    2: 'R',  # Rook
    3: 'Q'   # Queen
}


def signed_to_unsigned(a):
    r = struct.unpack('<H', struct.pack('<h', a))[0]
    if r & 0x8000:
        r ^= 0x7FFF  # flip value bits if negative
    r = (r << 1) | (r >> 15)  # store sign bit at bit 0
    return r


def parse_position(data):
    # Unpack the occupancy bitboard (8 bytes, big-endian)
    occupancy = struct.unpack('>Q', data[:8])[0]

    # Read the packed state (16 bytes)
    packed_state = data[8:24]

    # Initialize the board representation
    board = [['.' for _ in range(8)] for _ in range(8)]

    # Iterate over each bit in the occupancy bitboard
    bit_pos = 0
    nibble_index = 0
    for rank in range(8):
        for file in range(8):
            if occupancy & (1 << bit_pos):
                if nibble_index < len(packed_state):
                    nibble = (packed_state[nibble_index // 2] >> (4 * (nibble_index % 2))) & 0xF
                    piece = PIECE_MAP.get(nibble, '.')
                    board[7 - rank][file] = piece  # Flip rank for correct board orientation
                    nibble_index += 1
            bit_pos += 1

    # Convert board to a single string
    board_str = '\n'.join([''.join(rank) for rank in board])

    return board_str


def parse_move(data):
    # Unpack the 2 bytes of data into an integer
    packed_move = struct.unpack('>H', data)[0]

    # Extract move type
    move_type = (packed_move >> MOVE_TYPE_SHIFT) & MOVE_TYPE_MASK

    # Extract from square
    from_square = (packed_move >> FROM_SQUARE_SHIFT) & SQUARE_MASK

    # Extract to square
    to_square = (packed_move >> TO_SQUARE_SHIFT) & SQUARE_MASK

    # Extract promoted piece if it's a promotion move
    promoted_piece = None
    if move_type == 2:  # Assuming 2 represents Promotion
        promoted_piece_code = packed_move & PROMOTED_PIECE_TYPE_MASK
        promoted_piece = PROMOTION_PIECE_MAP.get(promoted_piece_code, None)

    return move_type, from_square, to_square, promoted_piece

def parse_score(data):
    score, = struct.unpack('>H', data[:2])
    return score

def parse_ply_and_result(data):
    ply_and_result, = struct.unpack('>H', data[:2])
    return ply_and_result

def parse_rule50(data):
    rule50, = struct.unpack('>H', data[:2])
    return rule50

def parse_movetext(data):
    count, = struct.unpack('>H', data[:2])
    moves_scores = []
    offset = 2
    for _ in range(count):
        # Placeholder for parsing the move and score
        encoded_move = data[offset:offset+2]
        encoded_score = data[offset+2:offset+4]
        moves_scores.append((encoded_move, encoded_score))
        offset += 4  # assuming 4 bytes for move and score
    return moves_scores

def parse_stem(data):
    pos = parse_position(data[:24])
    move = parse_move(data[24:26])
    score = parse_score(data[26:28])
    ply_and_result = parse_ply_and_result(data[28:30])
    rule50 = parse_rule50(data[30:32])
    return pos, move, score, ply_and_result, rule50

def extract_chess_positions(file_path):
    with open(file_path, 'rb') as f:
        while True:
            block_header = f.read(4)
            if not block_header:
                break
            if block_header != b'BINP':
                raise ValueError("Invalid block header")

            while True:
                stem_data = f.read(32)
                if not stem_data:
                    break

                pos, move, score, ply_and_result, rule50 = parse_stem(stem_data)
                print(f"Position: {pos}, Move: {move}, Score: {score}, Ply and Result: {ply_and_result}, Rule50: {rule50}")

                movetext_data_length = struct.unpack('>H', f.read(2))[0] * 4  # assuming each move and score pair is 4 bytes
                movetext_data = f.read(movetext_data_length)
                movetext = parse_movetext(movetext_data)
                print(f"Movetext: {movetext}")

if __name__ == "__main__":
    binpack_file_path = "../datasets/training_data.binpack"
    extract_chess_positions(binpack_file_path)