import unittest
import torch
import numpy as np

from src import model, wdl


class LossTest(unittest.TestCase):

    def setUp(self):
        # Initialize the parameters for the loss function
        self.scale = 400.0
        self.lambda_ = 0.5
        self.model = model.NNUE()

    def test_draw_prediction_matches_expected(self):
        prediction = torch.tensor([[wdl.wdl_to_nnue_output(0.5)], [wdl.wdl_to_nnue_output(0.5)]], dtype=torch.float32)
        targets = torch.tensor([[0.5, 0.0], [0.5, 0.0]], dtype=torch.float32)
        expected_loss = 0.0
        loss = self.model.loss(prediction, targets, self.scale, self.lambda_)
        self.assertEqual(loss.item(), expected_loss)

    def test_win_prediction_matches_expected(self):
        prediction = torch.tensor([[wdl.cp_to_nnue_output(2000)], [wdl.cp_to_nnue_output(2000)]], dtype=torch.float32)
        targets = torch.tensor([[1.0, 2000], [1.0, 2000]], dtype=torch.float32)
        expected_loss = 0.0
        loss = self.model.loss(prediction, targets, self.scale, self.lambda_)
        self.assertAlmostEqual(loss.item(), expected_loss, places=4)

    def test_loss_prediction_matches_expected(self):
        prediction = torch.tensor([[wdl.cp_to_nnue_output(-1000)], [wdl.cp_to_nnue_output(-1000)]], dtype=torch.float32)
        targets = torch.tensor([[0.0, -1000], [0.0, -1000]], dtype=torch.float32)
        expected_loss = 0.00
        loss = self.model.loss(prediction, targets, self.scale, self.lambda_)
        self.assertAlmostEqual(loss.item(), expected_loss, places=4)

    def test_loss_different_cp(self):
        # Case when cp_eval is different
        prediction = torch.tensor([[0.0], [0.0]], dtype=torch.float32)
        targets = torch.tensor([[0.0, 100.0], [1.0, -100.0]], dtype=torch.float32)
        expected_loss = 0.24491865  # Computed manually for this case
        loss = self.model.loss(prediction, targets, self.scale, self.lambda_)
        self.assertAlmostEqual(loss.item(), expected_loss, places=6)

    def test_loss_lambda_zero(self):
        # Case when lambda is zero
        self.lambda_ = 0.0
        prediction = torch.tensor([[0.0], [0.0]], dtype=torch.float32)
        targets = torch.tensor([[0.0, 100.0], [1.0, -100.0]], dtype=torch.float32)
        expected_loss = 0.24491865  # Computed manually for this case
        loss = self.model.loss(prediction, targets, self.scale, self.lambda_)
        self.assertAlmostEqual(loss.item(), expected_loss, places=6)

    def test_loss_lambda_one(self):
        # Case when lambda is one
        self.lambda_ = 1.0
        prediction = torch.tensor([[0.0], [0.0]], dtype=torch.float32)
        targets = torch.tensor([[0.0, 100.0], [1.0, -100.0]], dtype=torch.float32)
        expected_loss = 0.5  # Computed manually for this case
        loss = self.model.loss(prediction, targets, self.scale, self.lambda_)
        self.assertAlmostEqual(loss.item(), expected_loss, places=6)

if __name__ == '__main__':
    unittest.main()