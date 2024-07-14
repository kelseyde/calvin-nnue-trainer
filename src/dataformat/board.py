import struct

import numpy as np

CHAR_TO_PIECE_MAP = {
    'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
}
EMPTY_PIECE = 6
COLOUR_STRIDE = 64 * 6
PIECE_STRIDE = 64


class PackedBoard:

    def __init__(self, occ, pcs, cp, wdl, stm):
        self.occ = occ          # 64 bits
        self.pcs = pcs          # 32x4 = 128 bits
        self.cp = cp            # 16 bits
        self.wdl = wdl          # 2 bits, will be packed with stm
        self.stm = stm          # 1 bit, will be packed with wdl

    def to_bytes(self):
        """
        Convert the packed board to a byte string.
        """
        pcs_bytes = bytes(self.pcs)
        stm_wdl_byte = (self.stm << 2) | self.wdl
        cp_int = struct.unpack('>h', self.cp)[0]
        packed = struct.pack('>Q32shB', self.occ, pcs_bytes, cp_int, stm_wdl_byte)
        return packed

    def to_features(self):
        black_features = np.zeros(768, dtype=np.float32)
        white_features = np.zeros(768, dtype=np.float32)
        occ = self.occ
        idx = 0

        while occ:
            sq = lsb(occ)
            piece = self.pcs[idx]
            idx += 1
            piece_idx, colour = decode_piece(piece)
            is_white_piece = colour == 1
            w_sq, b_sq = sq, sq ^ 56
            w_colour_offset = 0 if is_white_piece else COLOUR_STRIDE
            b_colour_offset = COLOUR_STRIDE if is_white_piece else 0
            w_ft_idx = w_colour_offset + piece_idx * 64 + w_sq
            b_ft_idx = b_colour_offset + piece_idx * 64 + b_sq
            white_features[w_ft_idx] = 1
            black_features[b_ft_idx] = 1
            occ = pop_bit(occ)

        stm_features = white_features if self.stm == 1 else black_features
        nstm_features = black_features if self.stm == 1 else white_features

        input_data = np.array([stm_features, nstm_features], dtype=np.float32)
        output_data = np.array([decode_wdl(self.wdl), decode_cp(self.cp)], dtype=np.float32)

        return input_data, output_data


    @staticmethod
    def from_bytes(data: bytes):
        """
        Convert a byte string to a packed board.
        """
        occ = int.from_bytes(data[:8], 'big')
        pcs = list(data[8:40])
        cp = data[40:42]
        stm = (data[42] >> 2) & 0b1
        wdl = data[42] & 0b11
        return PackedBoard(occ, pcs, cp, wdl, stm)

    @staticmethod
    def from_labelled(labelled: str):
        fen, cp, wdl = labelled.split('|')
        fen_parts = fen.split()
        fen, active_color = fen_parts[:2]
        stm = 1 if active_color == 'w' else 0

        cp = encode_centipawns(parse_score(cp, stm))
        wdl = encode_wdl(parse_result(wdl, stm))
        occ = 0
        pcs = []
        rank_pcs = []

        rank, file = 7, 0
        for char in fen:
            if char == '/':
                pcs = rank_pcs + pcs
                rank_pcs = []
                rank, file = rank - 1, 0
            elif char.isdigit():
                file += int(char)
            else:
                colour = 1 if char.isupper() else 0
                type = CHAR_TO_PIECE_MAP[char.lower()]
                square_idx = 8 * rank + file
                occ |= 1 << square_idx
                rank_pcs.append(encode_piece(type, colour))
                file += 1
        pcs = rank_pcs + pcs

        while len(pcs) < 32:
            pcs.append(encode_piece(EMPTY_PIECE, 0))
        return PackedBoard(occ, pcs, cp, wdl, stm)


def encode_piece(piece_type, colour):
    """
    Encode piece type and color into a single integer using 4 bits.
    """
    return (colour << 3) | piece_type


def decode_piece(piece):
    """
    Decode the piece type and color from the encoded integer.
    """
    piece_type = piece & 0b111  # Last 3 bits
    colour = (piece >> 3) & 0b1  # 4th bit
    return piece_type, colour


def encode_wdl(wdl):
    """
    Encodes a WDL value into a byte.
    1.0 -> 00, 0.5 -> 01, 0.0 -> 10
    """
    if wdl == 1.0:
        return 0b00
    elif wdl == 0.5:
        return 0b01
    elif wdl == 0.0:
        return 0b10
    else:
        raise ValueError("Invalid WDL value")


def decode_wdl(byte):
    """
    Decodes a byte into a WDL value.
    00 -> 1.0, 01 -> 0.5, 10 -> 0.0
    """
    if byte == 0b00:
        return 1.0
    elif byte == 0b01:
        return 0.5
    elif byte == 0b10:
        return 0.0
    else:
        raise ValueError("Invalid encoded WDL byte")


def encode_centipawns(score):
    """
    Encodes a centipawn score into 16 bits.
    """
    # Ensure the score is within the range of a signed 16-bit integer
    if score < -32768 or score > 32767:
        raise ValueError("Score out of range for 16-bit signed integer")

    # Convert the score to a 16-bit signed integer
    encoded = struct.pack('>h', score)
    return encoded


def decode_cp(encoded):
    """
    Decodes a 16-bit encoded centipawn score.
    """
    # Convert the 16-bit signed integer back to a score
    score = struct.unpack('>h', encoded)[0]
    return score


def parse_result(result, stm):
    if '1-0' in result or "1.0" in result:
        return 1.0 if stm == 1 else 0.0
    elif '0-1' in result or "0.0" in result:
        return 0.0 if stm == 1 else 1.0
    elif '1/2-1/2' in result or "0.5" in result:
        return 0.5
    else:
        return None


def parse_score(score, stm):
    return int(score) if stm == 1 else -int(score)


def lsb(bb):
    return (bb & -bb).bit_length() - 1


def pop_bit(bb):
    return bb & (bb - 1)


# lab = "r1bqk2r/ppp2ppp/8/4b3/8/2P3P1/P3PPBP/R1BQK2R w KQkq - 0 10 | 5 | 0.5"
# print("Original length:", len(lab.encode('utf-8')))
# pb = PackedBoard.from_labelled(lab)
# pb_bytes = pb.to_bytes()
# print("Encoded length:", len(pb_bytes))
# pb2 = PackedBoard.from_bytes(pb_bytes)
# print("Re-encoded length:", len(pb2.to_bytes()))
#
# # Debug prints to check the pcs arrays
# print("Original pcs:", pb.pcs)
# print("Re-encoded pcs:", pb2.pcs)
#
# assert pb.occ == pb2.occ, "Occupancies do not match!"
# assert pb.pcs == pb2.pcs, "Pieces arrays do not match!"
# assert pb.cp == pb2.cp, "Centipawn scores do not match!"
# assert pb.wdl == pb2.wdl, "WDL values do not match!"
# assert pb.stm == pb2.stm, "Side to move values do not match!"
#
# b_features = pb.to_features()
# b2_features = pb2.to_features()
# fen_features = epd.parse_labelled_position(lab)
#
# print("Original features:", b_features)
# print("FEN features:", fen_features)
# print(len(b_features))
# print(len(fen_features))
# assert torch.equal(torch.tensor(b_features[0]), fen_features[0]), "Input tensors do not match!"
# assert torch.equal(torch.tensor(b_features[1]), fen_features[1]), "Output tensors do not match!"
#
# print("Test passed.")