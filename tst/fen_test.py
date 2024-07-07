import unittest

import torch

from src import wdl, quantize
from src.dataformat.fen import fen_to_features
from src.model import NNUE


class FenTest(unittest.TestCase):

    def test_fen_to_features(self):
        fen = "4b3/2k4P/2p4R/3r4/4K3/8/8/N7 w - - 0 1"
        features = fen_to_features(fen)
        print("total length: ", len(features))
        active_features = [i for i, x in enumerate(features) if x == 1]
        num_active_features = len(active_features)
        print(active_features)
        self.assertEqual(8, num_active_features)
        self.assertEqual(1, features[64])
        self.assertEqual(1, features[239])
        self.assertEqual(1, features[348])
        self.assertEqual(1, features[55])
        self.assertEqual(1, features[516])
        self.assertEqual(1, features[714])
        self.assertEqual(1, features[402])
        self.assertEqual(1, features[603])
        #
        # model = NNUE.load("/Users/kelseyde/git/dan/calvin/calvin-nnue-trainer/nets/first_attempt.nnue")
        # score = model(torch.tensor(features, dtype=torch.float32))
        # print(score)


    def test_eval_is_symmetrical(self):
        fen = "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
        features = fen_to_features(fen)

        reverse_fen = "rnbqk2r/pppp1ppp/5n2/4p3/1b2P3/P1N5/1PPP1PPP/R1BQKBNR b KQkq - 0 1"
        reverse_features = fen_to_features(reverse_fen)

        model = NNUE.load("/Users/kelseyde/git/dan/calvin/calvin-nnue-trainer/nets/small_net.nnue", input_size=768, hidden_size=2)

        score = model(torch.tensor(features, dtype=torch.float32))
        reverse_score = model(torch.tensor(reverse_features, dtype=torch.float32))
        print("score: ", score.item())
        print("reverse_score: ", reverse_score.item())
        self.assertEqual(score.item(), reverse_score.item())


    def test_startpos(self):
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        features = fen_to_features(fen)
        model = NNUE.load("/Users/kelseyde/git/dan/calvin/calvin-nnue-trainer/nets/yukon_ho_1.nnue", input_size=768, hidden_size=256)

        score = model(torch.tensor(features, dtype=torch.float32))
        print(score.item())
        print("cp: ", wdl.wdl_to_cp(score.item()))

        fen = "r1bqkbnr/1ppp1ppp/2n5/1p6/4P2P/5NPR/P1P1KP2/q1BQ4 b kq - 0 9"
        features = fen_to_features(fen)
        model = NNUE.load("/Users/kelseyde/git/dan/calvin/calvin-nnue-trainer/nets/yukon_ho_1.nnue", input_size=768, hidden_size=256)

        score = model(torch.tensor(features, dtype=torch.float32))
        print(score.item())
        print("cp: ", wdl.wdl_to_cp(score.item()))

        fen = "4k3/3ppp2/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1"
        features = fen_to_features(fen)
        model = NNUE.load("/Users/kelseyde/git/dan/calvin/calvin-nnue-trainer/nets/yukon_ho_1.nnue", input_size=768, hidden_size=256)

        score = model(torch.tensor(features, dtype=torch.float32))
        print(score.item())
        print("cp: ", wdl.wdl_to_cp(score.item()))


    def test_ruy_lopez(self):
        fen = "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
        features = fen_to_features(fen)
        model = NNUE.load("/Users/kelseyde/git/dan/calvin/calvin-nnue-trainer/nets/first_attempt.nnue")

        score = model(torch.tensor(features, dtype=torch.float32))
        print(score.item())
        print("cp: ", wdl.wdl_to_cp(score.item()))
        print("qcp: ", quantize.quantize_int16(wdl.wdl_to_cp(score.item())))