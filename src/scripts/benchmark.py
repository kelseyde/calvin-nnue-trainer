import torch

from src import wdl
from src.dataformat import fen
from src.model import NNUE

NET_FILE = "../../../calvin-chess-engine/src/main/resources/nnue/256HL-3B5083B8.nnue"
model = NNUE.load(NET_FILE)


def benchmark(name, fen_str):
    input_data = fen.fen_to_features(fen_str)
    eval = model(torch.stack([input_data[0]]))
    cp = wdl.nnue_output_to_cp(eval.item())
    print(f"{name} {fen_str} nnue: {cp}")


startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
lostpos = "rnbqkbnr/pppppppp/8/8/8/8/8/3QK3 w kq - 0 1"
wonpos = "rn2k1nr/ppp2ppp/8/4P3/2P3b1/8/PP1B1KPP/RN1q1BR1 b kq - 1 10"

benchmark("startpos", startpos)
benchmark("lostpos", lostpos)
benchmark("wonpos", wonpos)

