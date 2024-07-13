import numpy as np
import struct

CHAR_TO_PIECE_MAP = {
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
}

INDEX_TO_PIECE = {v: k for k, v in CHAR_TO_PIECE_MAP.items()}
INDEX_TO_PIECE.update({v + 6: k.upper() for k, v in CHAR_TO_PIECE_MAP.items() if k.islower()})

def fen_to_bitboards(fen):
    white_pieces = np.uint64(0)
    black_pieces = np.uint64(0)
    piece_bitboards = [np.uint64(0) for _ in range(6)]

    fen_parts = fen.split()
    board, stm = fen_parts[0], fen_parts[1]

    rank, file = 7, 0
    for char in board:
        if char.isdigit():
            file += int(char)
        elif char == '/':
            rank -= 1
            file = 0
        else:
            square_index = 8 * rank + file
            piece_index = CHAR_TO_PIECE_MAP[char]
            if char.isupper():
                white_pieces |= np.uint64(1) << np.uint64(square_index)
            else:
                black_pieces |= np.uint64(1) << np.uint64(square_index)
            piece_bitboards[piece_index] |= np.uint64(1) << np.uint64(square_index)
            file += 1

    return white_pieces, black_pieces, piece_bitboards, stm

def bitboards_to_fen(white_pieces, black_pieces, piece_bitboards, stm):
    board = ''
    for rank in range(7, -1, -1):
        empty_count = 0
        for file in range(8):
            square_index = 8 * rank + file
            piece_char = None
            if white_pieces & (np.uint64(1) << np.uint64(square_index)):
                for i, bitboard in enumerate(piece_bitboards):
                    if bitboard & (np.uint64(1) << np.uint64(square_index)):
                        piece_char = INDEX_TO_PIECE[i + 6]
                        break
            elif black_pieces & (np.uint64(1) << np.uint64(square_index)):
                for i, bitboard in enumerate(piece_bitboards):
                    if bitboard & (np.uint64(1) << np.uint64(square_index)):
                        piece_char = INDEX_TO_PIECE[i]
                        break
            if piece_char:
                if empty_count > 0:
                    board += str(empty_count)
                    empty_count = 0
                board += piece_char
            else:
                empty_count += 1
        if empty_count > 0:
            board += str(empty_count)
        if rank > 0:
            board += '/'

    stm_char = 'w' if stm == 'w' else 'b'
    return board + ' ' + stm_char

def encode_position(fen, centipawns, wdl):
    white_pieces, black_pieces, piece_bitboards, stm = fen_to_bitboards(fen)
    stm_byte = b'\x01' if stm == 'w' else b'\x00'
    centipawn_bytes = struct.pack('h', centipawns)
    wdl_byte = struct.pack('B', int(wdl * 255))

    encoded = white_pieces.tobytes() + black_pieces.tobytes()
    for bitboard in piece_bitboards:
        encoded += bitboard.tobytes()
    return encoded + stm_byte + centipawn_bytes + wdl_byte

def decode_position(encoded):
    white_pieces = np.frombuffer(encoded[:8], dtype=np.uint64)[0]
    black_pieces = np.frombuffer(encoded[8:16], dtype=np.uint64)[0]
    piece_bitboards = [np.frombuffer(encoded[16+i*8:24+i*8], dtype=np.uint64)[0] for i in range(6)]
    stm = 'w' if encoded[64] == 1 else 'b'
    centipawns = struct.unpack('h', encoded[65:67])[0]
    wdl = struct.unpack('B', encoded[67:68])[0] / 255

    fen = bitboards_to_fen(white_pieces, black_pieces, piece_bitboards, stm)
    return fen, centipawns, wdl

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'wb') as outfile:
        for line in infile:
            fen, cp, wdl = line.strip().split('|')
            encoded = encode_position(fen.strip(), int(cp.strip()), float(wdl.strip()))
            outfile.write(encoded)
            print(f"Original size: {len(line.encode('utf-8'))} bytes, Encoded size: {len(encoded)} bytes")

# Example usage
input_file = "../../datasets/training_data_tst.txt"
output_file = "../../datasets/training_data_tst.bin"
process_file(input_file, output_file)