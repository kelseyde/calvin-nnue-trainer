import unittest

import torch

from src import wdl
from src.dataformat.fen import fen_to_features
from src.model import NNUE


class ModelTest(unittest.TestCase):

    def test_save_and_load_give_same_output(self):

        model = NNUE(input_size=768, hidden_size=256)
        startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        stm_features, nstm_features, stm = fen_to_features(startpos)

        output1 = model(torch.tensor([[stm_features, nstm_features]], dtype=torch.float32))

        model.save("../nets/test_model.nnue")

        model = NNUE.load("../nets/test_model.nnue", input_size=768, hidden_size=256)
        output2 = model(torch.tensor([[stm_features, nstm_features]], dtype=torch.float32))

        cp1 = wdl.nnue_output_to_cp(output1.item())
        cp2 = wdl.nnue_output_to_cp(output2.item())
        self.assertAlmostEqual(cp1, cp2, delta=2)

    @staticmethod
    def load_features(fen):
        stm_features, nstm_features, stm = fen_to_features(fen)
        stm_features = torch.tensor([stm_features], dtype=torch.float32)
        nstm_features = torch.tensor([nstm_features], dtype=torch.float32)
        stm = torch.tensor([stm], dtype=torch.float32)
        return stm_features, nstm_features, stm
